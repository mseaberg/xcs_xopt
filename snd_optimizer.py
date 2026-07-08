"""
Xopt-based optimization for the XCS split-and-delay (SnD) alignment.

This module is intentionally free of any EPICS / hardware imports so that it can
be tested against a simulation (e.g. snd-online-model) without the
beamline control system.

The optimizer talks to the environment through an :class:`SnDBackend`:

* :class:`HardwareBackend` (defined in ``xcs_xopt.py``) drives the real SnD
  motors and reads diode / centroid signals over EPICS.
* :class:`SimulationBackend` (here) wraps a simulation model so the exact same
  optimization code can be exercised offline.

A backend maps a dictionary of *normalized* inputs in ``[low, high]`` to a set
of physics outputs ``{intensity, cx, cy, wx, wy}``.  The normalized -> physical
conversion (using each variable's ``pos_range`` and ``start_pos``) lives in the
base class so it is shared by all backends.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.linalg import hadamard
from scipy.stats import qmc

from xopt import Xopt, Evaluator, VOCS
from xopt.generators.bayesian import (
    ExpectedImprovementGenerator,
    UpperConfidenceBoundGenerator,
)
from xopt.generators.bayesian.models.standard import StandardModelConstructor


# ---------------------------------------------------------------------------
# Backend interface
# ---------------------------------------------------------------------------
class SnDBackend(ABC):
    """Abstract environment the optimizer evaluates against.

    Parameters
    ----------
    variable_names:
        Ordered names of the optimization variables (one per motor/knob).
    pos_range:
        Physical motion range for each variable, used to map a normalized input
        onto a physical setpoint.
    start_pos:
        Physical center position for each variable.
    """

    def __init__(
        self,
        variable_names: Sequence[str],
        pos_range: Dict[str, float],
        start_pos: Dict[str, float],
    ):
        self.variable_names: List[str] = list(variable_names)
        self.pos_range: Dict[str, float] = dict(pos_range)
        self.start_pos: Dict[str, float] = dict(start_pos)

    def set_pos_range(self, motion_range: float) -> None:
        """Set the same physical motion range for every variable."""
        for name in self.variable_names:
            self.pos_range[name] = motion_range

    def refresh_start_pos(self) -> None:
        """Refresh ``start_pos`` from the live environment (no-op by default)."""

    def to_physical(self, input_dict: Dict[str, float]) -> Dict[str, float]:
        """Convert normalized inputs in ``[low, high]`` to physical setpoints."""
        return {
            key: value * self.pos_range[key]
            - self.pos_range[key] / 2.0
            + self.start_pos[key]
            for key, value in input_dict.items()
        }

    def evaluate(self, input_dict: Dict[str, float]) -> Dict[str, float]:
        """Map normalized inputs to ``{intensity, cx, cy, wx, wy}``."""
        return self.measure(self.to_physical(input_dict))

    @abstractmethod
    def measure(self, positions: Dict[str, float]) -> Dict[str, float]:
        """Apply physical ``positions`` and return the measured outputs."""

    @abstractmethod
    def set_target(self) -> Tuple[float, float]:
        """Return the ``(x_target, y_target)`` centroid target."""

    @abstractmethod
    def move_to_start(self) -> None:
        """Return all variables to their ``start_pos``."""


class SimulationBackend(SnDBackend):
    """Backend that evaluates a simulation model instead of real hardware.

    The simulation itself is supplied as ``model_fn``: a callable that takes a
    dict of physical positions ``{variable_name: value}`` and returns a dict
    with keys ``intensity``, ``cx``, ``cy``, ``wx``, ``wy``.  This is the single
    integration point for the simulation (write a small adapter
    that calls the model and reshapes its output into that dict).

    Example
    -------
    >>> def model_fn(positions):
    ...     out = snd_model.run(positions)        # example simulation call
    ...     return {"intensity": out.flux, "cx": out.x, "cy": out.y,
    ...             "wx": out.xw, "wy": out.yw}
    >>> backend = SimulationBackend(
    ...     variable_names=["t1_th1", "t1_chi1"],
    ...     model_fn=model_fn,
    ...     motion_range=50e-6 * 180 / np.pi,
    ... )
    """

    def __init__(
        self,
        variable_names: Sequence[str],
        model_fn: Callable[[Dict[str, float]], Dict[str, float]],
        motion_range: float = 50e-6 * 180 / np.pi,
        start_pos: Optional[Dict[str, float]] = None,
        target: Tuple[float, float] = (0.0, 0.0),
    ):
        names = list(variable_names)
        pos_range = {name: motion_range for name in names}
        start_pos = (
            dict(start_pos) if start_pos is not None else {name: 0.0 for name in names}
        )
        super().__init__(names, pos_range, start_pos)
        self.model_fn = model_fn
        self._target = target

    def measure(self, positions: Dict[str, float]) -> Dict[str, float]:
        return self.model_fn(positions)

    def set_target(self) -> Tuple[float, float]:
        return self._target

    def move_to_start(self) -> None:
        # Nothing physical to move for a simulation.
        return None


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------
class SnDOptimizer:
    """Bayesian / TuRBO optimization of SnD alignment via Xopt 3.x.

    Parameters
    ----------
    backend:
        Environment to evaluate against (hardware or simulation).
    savepath:
        Directory to write optimum CSV / NPZ output. If ``None`` saving is
        skipped.
    intensity_scale:
        Weight applied to the intensity term in the objective.
    """

    def __init__(
        self,
        backend: SnDBackend,
        savepath: Optional[str] = None,
        intensity_scale: float = 1e-4,
        prior_mean_model=None,
    ):
        self.backend = backend
        self.savepath = savepath
        self.intensity_scale = intensity_scale
        # optional torch.nn.Module used as the GP prior mean for objective "f"
        self.prior_mean_model = prior_mean_model

        self.vocs: Optional[VOCS] = None
        self.generator = None
        self.evaluator: Optional[Evaluator] = None
        self.X: Optional[Xopt] = None

        self.n_init = 0
        self.num = 0
        self.x_target = 0.0
        self.y_target = 0.0

        # measurement history, appended in evaluation order (matches X.data order)
        self._bpe: List[float] = []
        self._intensity: List[float] = []

    @property
    def name_list(self) -> List[str]:
        return self.backend.variable_names

    @property
    def detailed_output(self) -> Dict[str, np.ndarray]:
        return {
            "BPE": np.array(self._bpe),
            "Intensity": np.array(self._intensity),
        }

    # -- setup -------------------------------------------------------------
    def set_target(self) -> None:
        """Read the centroid target from the backend."""
        self.x_target, self.y_target = self.backend.set_target()

    def set_vocs(self, low: float = 0.0, high: float = 1.0) -> None:
        self.backend.refresh_start_pos()
        variables = {name: [low, high] for name in self.name_list}
        self.vocs = VOCS(variables=variables, objectives={"f": "MINIMIZE"})
        # The prior mean indexes input columns by position, and the GP feeds
        # columns in vocs.variable_names order. That order MUST equal the
        # backend's variable order (which is paired position-for-position with
        # the sim axis names in User.set_motors). It holds today because VOCS
        # preserves insertion order; if a VOCS implementation ever sorts, this
        # fails loudly instead of silently permuting the physics prior against
        # the motors.
        if list(self.vocs.variable_names) != list(self.name_list):
            raise RuntimeError(
                "VOCS reordered the variables: "
                f"{list(self.vocs.variable_names)} != {list(self.name_list)}. "
                "The prior-mean column mapping is positional and would be "
                "scrambled; refusing to build a mis-mapped optimizer."
            )

    def set_prior_mean(self, module) -> None:
        """Set (or clear) the GP prior-mean module for objective ``f``.

        The module captures fixed quantities (``start_pos``, target, scale), so
        rebuild and set it *after* ``set_target`` and *before* ``initialize_*``.
        """
        self.prior_mean_model = module

    def _gp_constructor(self):
        """Build a StandardModelConstructor wiring the frozen prior mean for ``f``.

        Returns ``None`` when no prior-mean module is set, so the generator uses
        its default model.
        """
        if self.prior_mean_model is None:
            return None
        return StandardModelConstructor(
            mean_modules={"f": self.prior_mean_model},
            trainable_mean_keys=[],
        )

    def _reset_history(self) -> None:
        self.num = 0
        self._bpe = []
        self._intensity = []

    def initialize_BO(self, n_init: int = 64, scale: float = 1e-4) -> None:
        self.n_init = n_init
        self.intensity_scale = scale
        self._reset_history()
        self.set_vocs()
        self.evaluator = Evaluator(function=self.eval_function)
        gp_constructor = self._gp_constructor()
        if gp_constructor is not None:
            self.generator = UpperConfidenceBoundGenerator(
                vocs=self.vocs, gp_constructor=gp_constructor
            )
        else:
            self.generator = UpperConfidenceBoundGenerator(vocs=self.vocs)
        self.X = Xopt(evaluator=self.evaluator, generator=self.generator)

    def initialize_BO_transformed(self, n_init: int = 64, scale: float = 1e-4) -> None:
        self.n_init = n_init
        self.intensity_scale = scale
        self._reset_history()
        self.set_vocs(low=-0.5, high=0.5)
        self.evaluator = Evaluator(function=self.eval_function_transformed)
        self.generator = UpperConfidenceBoundGenerator(vocs=self.vocs)
        self.X = Xopt(evaluator=self.evaluator, generator=self.generator)

    def initialize_turbo(self, n_init: int = 64, scale: float = 1.0) -> None:
        self.n_init = n_init
        self.intensity_scale = scale
        self._reset_history()
        self.set_vocs()
        self.evaluator = Evaluator(function=self.eval_function)
        gp_constructor = self._gp_constructor()
        if gp_constructor is not None:
            self.generator = ExpectedImprovementGenerator(
                vocs=self.vocs,
                turbo_controller="optimize",
                gp_constructor=gp_constructor,
            )
        else:
            self.generator = ExpectedImprovementGenerator(
                vocs=self.vocs, turbo_controller="optimize"
            )
        self.X = Xopt(evaluator=self.evaluator, generator=self.generator)

    # -- objective ---------------------------------------------------------
    def _objective(self, data: Dict[str, float]) -> Dict[str, float]:
        """Compute the scalar objective from a backend output dict."""
        if np.isnan(data["cx"]):
            bpe = np.nan
        else:
            bpe = np.sqrt(
                (data["cx"] - self.x_target) ** 2
                + (data["cy"] - self.y_target) ** 2
            )

        self._bpe.append(bpe)
        self._intensity.append(data["intensity"])
        self.num += 1

        print("BPE: {}".format(bpe))
        print("Intensity: {}".format(data["intensity"]))

        result = -self.intensity_scale * data["intensity"] + bpe / 350
        return {"f": result}

    def eval_function(self, input_dict: Dict[str, float]) -> Dict[str, float]:
        data = self.backend.evaluate(input_dict)
        return self._objective(data)

    def eval_function_transformed(
        self, input_dict: Dict[str, float]
    ) -> Dict[str, float]:
        """Evaluate after a Hadamard rotation of the input space.

        Based on the work published with Aashwin. The inputs are rotated by an
        orthonormal Hadamard matrix and shifted back into ``[0, 1]`` before
        being handed to the backend. Requires the number of variables to be a
        power of two.
        """
        keys = self.name_list
        v = np.array([input_dict[key] for key in keys])
        n = len(keys)
        H = hadamard(n)
        R = H / np.sqrt(n)
        out = R @ v + 0.5
        transformed = {key: out[i] for i, key in enumerate(keys)}
        data = self.backend.evaluate(transformed)
        return self._objective(data)

    # -- runs --------------------------------------------------------------
    def _initial_samples(self, seed: Optional[int] = None) -> pd.DataFrame:
        sampler = qmc.LatinHypercube(d=len(self.name_list), seed=seed)
        xs = sampler.random(n=self.n_init)
        # scale the unit-cube samples into each variable's VOCS bounds
        variables = self.vocs.variables
        return pd.DataFrame(
            {
                name: variables[name].domain[0]
                + xs[:, i] * (variables[name].domain[1] - variables[name].domain[0])
                for i, name in enumerate(self.name_list)
            }
        )

    def run_BO(self, num_iter: int = 150, seed: int = 42) -> pd.DataFrame:
        self.X.evaluate_data(self._initial_samples(seed=seed))

        for i in range(num_iter):
            print(i)
            self.X.step()
            self.X.generator.beta += 0.2
            print("beta={}".format(self.X.generator.beta))

        return self.X.data

    def run_turbo(self, num_iter: int = 150) -> pd.DataFrame:
        self.X.evaluate_data(self._initial_samples())
        print("done with random sampling")

        for i in range(num_iter):
            print(f"Step: {i + 1}")
            # Xopt.step() trains the model and updates the turbo trust region
            # internally; we only read state here for diagnostics.
            self.X.step()
            tc = self.X.generator.turbo_controller
            if tc is not None:
                print(
                    f"trust-region length={tc.length}, best_value={tc.best_value}"
                )

        return self.X.data

    # -- results -----------------------------------------------------------
    def get_optimum_details(
        self, plot: bool = True, move_to_optimum: bool = True
    ) -> None:
        """Find the best sample, optionally move to it, and save/plot results."""
        data = self.X.generator.data

        min_idx = int(np.argmin(data["f"].to_numpy()))
        X_min = data.iloc[min_idx]

        if move_to_optimum:
            print("moving to optimum")
            best_inputs = {name: float(X_min[name]) for name in self.name_list}
            self.backend.evaluate(best_inputs)

        bpe = self.detailed_output["BPE"]
        intensity = self.detailed_output["Intensity"]
        bpe_out = bpe[min_idx] if min_idx < len(bpe) else np.nan
        intensity_out = intensity[min_idx] if min_idx < len(intensity) else np.nan

        if self.savepath is not None:
            timestamp = str(int(datetime.now().timestamp()))
            filename = "optimize_output_{}".format(timestamp)
            data.to_csv(self.savepath + filename + ".csv", index=False)
            np.savez(
                self.savepath + filename + ".npz", bpe=bpe, intensity=intensity
            )

        print("Optimum Inputs: ", X_min)
        print("BPE: ", bpe_out)
        print("Intensity:", intensity_out)

        if plot:
            import matplotlib.pyplot as plt

            y1 = data["f"]
            y1_mins = np.minimum.accumulate(y1)
            idx = np.arange(len(y1_mins))
            fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 7))
            ax0.plot(idx, y1_mins, "k")
            ax0.grid()
            ax0.set_xlabel("Sample Number")
            ax0.set_ylabel("Objective")
            ax1.plot(idx[-50:], y1_mins[-50:], "k")
            ax1.grid()
            ax1.set_ylabel("Objective")
            plt.show()

    def move_to_start(self) -> None:
        print("moving to start")
        self.backend.move_to_start()
