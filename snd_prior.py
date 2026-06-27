"""
Differentiable SND simulator wired up as a GP prior mean for Xopt.

This module bridges the PyTorch wave-optics split-and-delay simulator
(``lcls_beamline_differentiable``) into the Bayesian optimization defined in
``snd_optimizer.py``.  The same simulator is exposed two ways:

* ``model_fn`` -- a plain ``{positions} -> {intensity, cx, cy, wx, wy}`` callable
  for :class:`snd_optimizer.SimulationBackend`, so the *data* the optimizer
  collects comes from the sim.
* :class:`SnDObjectivePriorMean` -- a ``torch.nn.Module`` that returns the scalar
  objective ``f`` directly, used as the GP *prior mean*
  (``StandardModelConstructor(mean_modules={"f": module})``).  Bayesian
  optimization then only has to learn the residual between the physics model and
  the real measurement.

Both paths drive a deep copy of one pristine ``SND`` template per evaluation, so
the stateful/relative motor moves always start from the aligned baseline.

Differentiability notes (verified against the installed sim):

* ``detector.profile`` is a torch tensor that carries grad through the device
  geometry -> wave propagation (torch FFT) -> ``profile.sum()``.
* The sim's own ``get_*_cx/cy`` go through a numpy curve-fit analysis layer that
  is intentionally detached.  We therefore recompute the centroid directly from
  the torch lineouts and the detector coordinate grids
  (``cx = sum(x_lineout * x) / sum(x_lineout)``) so it stays on the gradient path.
  This weighted-mean centroid matches the sim's numpy centroid to ~1e-8 m.
* Widths (``wx/wy``) come from the numpy analysis layer; the objective does not
  use them, they are only filled in to satisfy the backend output contract.
"""

from __future__ import annotations

import copy
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from lcls_beamline_differentiable.models.split_and_delay_motion import SND


DTYPE = torch.float64

# default knobs, aligned to the SND model's RotationAxis attribute names
DEFAULT_AXIS_NAMES: List[str] = [
    "t1_th1", "t1_chi1", "t1_th2", "t1_chi2",
    "t4_th1", "t4_chi1", "t4_th2", "t4_chi2",
]


def _differentiable_readout(
    snd: SND, detector: str
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Read ``(intensity, cx, cy)`` from a propagated ``SND`` keeping autograd.

    ``detector`` selects the diode/screen, e.g. ``"do"`` -> ``snd.delay_branch.do``.
    The centroid is the lineout-weighted mean of the detector coordinate grid,
    which is differentiable, unlike the sim's numpy ``get_<det>_cx/cy``.

    When the beam is fully detuned/clipped away the detector profile is either
    the numpy zero array left by ``reset()`` or a torch tensor of NaNs.  Both are
    treated as a beam-lost readout ``(0, 0, 0)``: zero intensity already makes the
    objective much worse than any beam-on point, so this is the correct penalty.
    """
    det = getattr(snd.delay_branch, detector)
    profile = det.profile

    if not torch.is_tensor(profile):
        # numpy zeros from reset(): beam never reached this detector
        zero = torch.zeros((), dtype=DTYPE)
        return zero, zero, zero

    if not torch.isfinite(profile).all() or float(profile.sum()) <= 0.0:
        zero = torch.zeros((), dtype=DTYPE)
        return zero, zero, zero

    x_grid = torch.as_tensor(np.asarray(det.x), dtype=DTYPE)
    y_grid = torch.as_tensor(np.asarray(det.y), dtype=DTYPE)
    x_lineout = torch.sum(profile, dim=0)
    y_lineout = torch.sum(profile, dim=1)

    cx = torch.sum(x_lineout * x_grid) / torch.sum(x_lineout)
    cy = torch.sum(y_lineout * y_grid) / torch.sum(y_lineout)
    intensity = profile.sum()
    return intensity, cx, cy


class SnDObjectivePriorMean(torch.nn.Module):
    """Frozen physics prior over the SnD alignment objective ``f``.

    Receives inputs in the VOCS-variable space (the normalized ``[0, 1]`` knobs
    the optimizer searches) and returns the raw scalar objective ``f``; Xopt's
    ``StandardModelConstructor`` standardizes that into the GP target space.

    Parameters
    ----------
    snd_template:
        A constructed, aligned ``SND``.  Deep-copied per evaluation so the
        relative/stateful motor moves always start from the same baseline.
    axis_names:
        SND attribute names, in the same order as the optimizer's
        ``variable_names`` (one per knob).
    pos_range, start_pos:
        Physical motion range and center per axis -- **must match the evaluator
        backend** so the prior and the data live in the same normalized space.
        The normalized -> physical map mirrors ``SnDBackend.to_physical``:
        ``phys = x * pos_range - pos_range / 2 + start_pos``.
    x_target, y_target:
        Centroid target used by the beam-position-error term.
    intensity_scale:
        Weight on the intensity term, matching ``SnDOptimizer._objective``.
    detector:
        Which detector to read (default ``"do"``).
    """

    def __init__(
        self,
        snd_template: SND,
        axis_names: Sequence[str],
        pos_range: Dict[str, float],
        start_pos: Dict[str, float],
        x_target: float = 0.0,
        y_target: float = 0.0,
        intensity_scale: float = 1e-4,
        detector: str = "do",
    ):
        super().__init__()
        self._template = snd_template
        self.axis_names = list(axis_names)
        self.detector = detector
        self.intensity_scale = float(intensity_scale)
        self.x_target = float(x_target)
        self.y_target = float(y_target)

        # physical conversion vectors in axis order (constant, no grad)
        self.register_buffer(
            "pos_range_vec",
            torch.tensor([pos_range[n] for n in self.axis_names], dtype=DTYPE),
        )
        self.register_buffer(
            "start_pos_vec",
            torch.tensor([start_pos[n] for n in self.axis_names], dtype=DTYPE),
        )

    def _objective_one(self, phys_row: torch.Tensor) -> torch.Tensor:
        """Evaluate ``f`` for one candidate given physical setpoints (axis order)."""
        snd = copy.deepcopy(self._template)
        for k, name in enumerate(self.axis_names):
            getattr(snd, name).mv(phys_row[k])
        snd.propagate_delay()

        intensity, cx, cy = _differentiable_readout(snd, self.detector)
        bpe = torch.sqrt((cx - self.x_target) ** 2 + (cy - self.y_target) ** 2)
        return -self.intensity_scale * intensity + bpe / 350.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map normalized inputs ``b x n x d`` to objective values ``b x n``.

        The trailing output dim is squeezed, matching Xopt's ``CustomMean``
        contract (cf. the prior-mean test module in xopt): the GP mean must
        broadcast against the ``b x n`` training/candidate batch shape.
        """
        x = x.to(DTYPE)
        lead_shape = x.shape[:-1]
        d = x.shape[-1]
        flat = x.reshape(-1, d)

        phys = flat * self.pos_range_vec - self.pos_range_vec / 2.0 + self.start_pos_vec

        f_vals = [self._objective_one(phys[i]) for i in range(phys.shape[0])]
        f = torch.stack(f_vals)
        return f.reshape(*lead_shape)


def build_snd_sim(
    axis_names: Optional[Sequence[str]] = None,
    backend_pos_range: Optional[Dict[str, float]] = None,
    backend_start_pos: Optional[Dict[str, float]] = None,
    *,
    energy: float = 9500.0,
    delay: float = 280e-3,
    detector: str = "do",
    x_target: float = 0.0,
    y_target: float = 0.0,
    intensity_scale: float = 1e-4,
) -> Tuple[Callable[[Dict[str, float]], Dict[str, float]], SnDObjectivePriorMean]:
    """Build a sim-backed evaluator callable and a matching prior-mean module.

    The evaluator (``model_fn``) and the prior share one ``SND`` template, so the
    data the optimizer collects and the prior it starts from come from the
    *identical* simulator.

    ``backend_pos_range`` / ``backend_start_pos`` should be the same dicts the
    :class:`snd_optimizer.SimulationBackend` uses, keyed by ``axis_names``.  If
    omitted, they default to the aligned baseline (``wm()``) with the standard
    ``50e-6 rad`` angular range, matching the hardware backend convention.

    Returns
    -------
    (model_fn, prior_mean_model)
        ``model_fn(physical_positions) -> {intensity, cx, cy, wx, wy}`` is the
        numpy callable for ``SimulationBackend``; ``prior_mean_model`` is the
        :class:`SnDObjectivePriorMean`.
    """
    if axis_names is None:
        axis_names = list(DEFAULT_AXIS_NAMES)
    else:
        axis_names = list(axis_names)

    template = SND(energy=energy, delay=delay)

    if backend_start_pos is None:
        backend_start_pos = {
            name: float(getattr(template, name).wm()) for name in axis_names
        }
    if backend_pos_range is None:
        import numpy as np

        backend_pos_range = {name: 50e-6 * 180 / np.pi for name in axis_names}

    def model_fn(positions: Dict[str, float]) -> Dict[str, float]:
        snd = copy.deepcopy(template)
        for name in axis_names:
            getattr(snd, name).mv(positions[name])
        snd.propagate_delay()

        intensity, cx, cy = _differentiable_readout(snd, detector)
        # widths are not on the objective path; pull from the numpy analysis layer
        wx = getattr(snd, f"get_{detector}_wx")()
        wy = getattr(snd, f"get_{detector}_wy")()
        return {
            "intensity": float(intensity.detach()),
            "cx": float(cx.detach()),
            "cy": float(cy.detach()),
            "wx": float(wx),
            "wy": float(wy),
        }

    prior_mean_model = SnDObjectivePriorMean(
        template,
        axis_names,
        backend_pos_range,
        backend_start_pos,
        x_target=x_target,
        y_target=y_target,
        intensity_scale=intensity_scale,
        detector=detector,
    )

    return model_fn, prior_mean_model
