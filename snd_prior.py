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

# Detector arrays that are constant across evaluations and may be shared (not
# copied) through the per-eval deepcopy memo. The coordinate (x, y, xx, yy) and
# frequency (f_x, f_y) grids are never written during mv/reset/propagate/
# calc_profile. profile is rebound to fresh zeros by reset() before any read, so
# the template's profile object is likewise never read or mutated via the copy.
# Sharing these avoids copying the IP PPM's 2048x2048 grids (~134 MB) every eval.
_SHARED_CONST_ATTRS: Tuple[str, ...] = ("x", "y", "xx", "yy", "f_x", "f_y", "profile")


def _readout_from_branch(
    branch, detector: str
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Read ``(intensity, cx, cy)`` from a propagated delay ``branch``, keeping autograd.

    ``detector`` selects the diode/screen, e.g. ``"do"`` -> ``branch.do``.
    The centroid is the lineout-weighted mean of the detector coordinate grid,
    which is differentiable, unlike the sim's numpy ``get_<det>_cx/cy``.

    When the beam is fully detuned/clipped away the detector profile is either
    the numpy zero array left by ``reset()`` or a torch tensor of NaNs.  Both are
    treated as a beam-lost readout ``(0, 0, 0)``: zero intensity already makes the
    objective much worse than any beam-on point, so this is the correct penalty.
    """
    det = getattr(branch, detector)
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
        Physical motion range and center per axis, in the same (hardware / PV)
        units as the evaluator backend so the prior and the data live in the
        same normalized space. The normalized -> physical map mirrors
        ``SnDBackend.to_physical``: ``phys = x * pos_range - pos_range / 2 +
        start_pos``. Those physical setpoints are then scaled into the
        simulator's native units by ``unit_conversion`` before being applied to
        the sim axes (see ``input_transform``).
    unit_conversion:
        Per-axis PV -> simulation factor (``sim = pv * unit_conversion``), keyed
        by ``axis_names``; from ``pv_mapping.json``. Defaults to ``1.0`` for any
        missing axis, i.e. no scaling (identity), which is the right behavior for
        a sim-only run where the data and the prior already share sim units.
    cx_conversion, cy_conversion:
        Centroid PV -> simulation factors for the read detector
        (``sim_m = pv_mm * conversion``). The sim returns centroids in meters; we
        divide by these to express the beam-position error in the same PV units
        as ``x_target`` / ``y_target`` and the measured objective. Default ``1.0``.
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
        unit_conversion: Optional[Dict[str, float]] = None,
        cx_conversion: float = 1.0,
        cy_conversion: float = 1.0,
        x_target: float = 0.0,
        y_target: float = 0.0,
        intensity_scale: float = 1e-4,
        detector: str = "do",
    ):
        super().__init__()
        self._template = snd_template
        self.axis_names = list(axis_names)
        self.cx_conversion = float(cx_conversion)
        self.cy_conversion = float(cy_conversion)

        # propagate_delay only touches the delay branch and reads b1 (which
        # propagate_beamline clones, never mutates). So per evaluation we deep-copy
        # only the (delay_branch, moved-axes) subgraph in a SINGLE deepcopy call --
        # the cc/bypass branches are dropped (~2/3 of the full-SND copy), and the
        # one call keeps deepcopy's memo linking each axis to the crystals inside
        # the copied branch. b1 is shared read-only across evaluations.
        self._delay_template = snd_template.delay_branch
        self._axis_templates = [getattr(snd_template, n) for n in self.axis_names]
        self._b1 = snd_template.b1

        # Constant detector arrays shared (not copied) through the per-eval
        # deepcopy memo, gathered once from the delay template (see
        # _SHARED_CONST_ATTRS and _objective_one).
        self._shared_consts = []
        for dev in self._delay_template.device_list:
            for attr in _SHARED_CONST_ATTRS:
                v = getattr(dev, attr, None)
                if torch.is_tensor(v) or isinstance(v, np.ndarray):
                    self._shared_consts.append(v)
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
        # per-axis PV -> simulation input scaling (identity if not supplied)
        unit_conversion = unit_conversion or {}
        self.register_buffer(
            "input_scale_vec",
            torch.tensor(
                [float(unit_conversion.get(n, 1.0)) for n in self.axis_names],
                dtype=DTYPE,
            ),
        )

    def _objective_one(self, phys_row: torch.Tensor) -> torch.Tensor:
        """Evaluate ``f`` for one candidate given physical setpoints (axis order).

        ``phys_row`` is in PV / hardware units; it is scaled into the simulator's
        native units (``input_transform``) before driving the axes, and the
        resulting centroids are scaled back to PV units (``output_transform``)
        so the beam-position error matches the measured objective and target.
        """
        # single deepcopy of the connected subgraph so the copied axes bind to the
        # crystals inside the copied delay branch (see __init__ for why). Seed the
        # memo so the constant detector grids are shared, not copied. A FRESH dict
        # per call is required: deepcopy mutates the memo, and reusing it would
        # alias this eval's mutable crystal/axis copies into the next eval.
        memo = {id(v): v for v in self._shared_consts}
        delay, axes = copy.deepcopy((self._delay_template, self._axis_templates), memo)
        sim_row = phys_row * self.input_scale_vec
        for k, ax in enumerate(axes):
            ax.mv(sim_row[k])
        # mirror SND.propagate_delay: reset detectors, then propagate b1 (shared,
        # cloned inside propagate_beamline)
        for device in delay.device_list:
            device.reset()
        delay.propagate_beamline(self._b1)

        intensity, cx, cy = _readout_from_branch(delay, self.detector)
        cx = cx / self.cx_conversion
        cy = cy / self.cy_conversion
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
    unit_conversion: Optional[Dict[str, float]] = None,
    cx_conversion: float = 1.0,
    cy_conversion: float = 1.0,
) -> Tuple[Callable[[Dict[str, float]], Dict[str, float]], SnDObjectivePriorMean]:
    """Build a sim-backed evaluator callable and a matching prior-mean module.

    The evaluator (``model_fn``) and the prior share one ``SND`` template, so the
    data the optimizer collects and the prior it starts from come from the
    *identical* simulator.

    ``backend_pos_range`` / ``backend_start_pos`` should be the same dicts the
    backend uses (in its PV / hardware units), keyed by ``axis_names``.  If
    omitted, they default to the aligned baseline (``wm()``) with the standard
    ``50e-6 rad`` angular range, matching the hardware backend convention.

    ``unit_conversion`` (per-axis PV -> sim factor) and ``cx_conversion`` /
    ``cy_conversion`` (centroid PV -> sim factor for the read detector) come from
    ``pv_mapping.json`` and let the prior consume hardware-unit setpoints and
    emit hardware-unit centroids. They default to identity (``1.0``), the correct
    choice for a sim-only run where data and prior already share sim units.

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

    # The objective recomputes the centroid differentiably from the detector
    # lineouts, so the numpy curve_fit analysis layer (PPM.beam_analysis) is dead
    # weight on the propagation hot path. Disable it on every delay-branch
    # detector; model_fn re-runs it on demand for just the one read detector below.
    for dev in template.delay_branch.device_list:
        if hasattr(dev, "analyze"):
            dev.analyze = False

    unit_conversion = unit_conversion or {}
    if backend_start_pos is None:
        # The sim's aligned baseline (wm(), in sim units) expressed in PV units,
        # so normalized 0.5 maps to the aligned crystal positions. input_transform
        # in the prior scales it straight back to sim units before any motor move.
        backend_start_pos = {
            name: float(getattr(template, name).wm()) / unit_conversion.get(name, 1.0)
            for name in axis_names
        }
    if backend_pos_range is None:
        backend_pos_range = {name: 50e-6 * 180 / np.pi for name in axis_names}

    def model_fn(positions: Dict[str, float]) -> Dict[str, float]:
        snd = copy.deepcopy(template)
        for name in axis_names:
            getattr(snd, name).mv(positions[name])
        snd.propagate_delay()

        intensity, cx, cy = _readout_from_branch(snd.delay_branch, detector)
        # widths are not on the objective path; the analysis layer is disabled on
        # the template, so re-run beam_analysis here for just the read detector
        # (numpy, detached) and stash the widths it returns.
        det = getattr(snd.delay_branch, detector)
        lx, ly = det.x_lineout, det.y_lineout
        lx = lx.detach().cpu().numpy() if torch.is_tensor(lx) else np.asarray(lx)
        ly = ly.detach().cpu().numpy() if torch.is_tensor(ly) else np.asarray(ly)
        _, _, wx, wy, _, _ = det.beam_analysis(lx, ly)
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
        unit_conversion=unit_conversion,
        cx_conversion=cx_conversion,
        cy_conversion=cy_conversion,
        x_target=x_target,
        y_target=y_target,
        intensity_scale=intensity_scale,
        detector=detector,
    )

    return model_fn, prior_mean_model
