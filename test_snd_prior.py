"""Guard tests for the differentiable SnD prior-mean module.

The critical one is ``test_prior_mean_is_differentiable``: Xopt will silently
run a non-differentiable prior mean (acquisition optimization still gets
gradients from the GP kernel), so a detached graph degrades optimization without
ever raising. This test will turn that into a hard failure.

Skipped automatically when ``lcls_beamline_differentiable`` is not installed.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lcls_beamline_differentiable")

from snd_prior import build_snd_sim


@pytest.fixture(scope="module")
def prior():
    # one SND template shared across tests in this module (construction is cheap
    # but propagation per eval is not, so keep evaluations small)
    _, prior_mean_model = build_snd_sim()
    return prior_mean_model


def test_prior_mean_is_differentiable(prior):
    """f must carry a finite, nonzero gradient back to the input knobs.

    Inputs are taken slightly off-center (0.55, not 0.5): at exactly the aligned
    baseline the sim's rotation matrix short-circuits to a constant identity
    (``theta < 1e-30``), which legitimately severs the graph at that single
    point. Any real candidate is perturbed away from it.
    """
    d = len(prior.axis_names)
    x = torch.full((1, 2, d), 0.55, dtype=torch.float64, requires_grad=True)

    f = prior(x)
    assert f.requires_grad, "prior mean output detached from input graph"

    f.sum().backward()
    g = x.grad
    assert g is not None, "no gradient propagated to the input knobs"
    assert torch.isfinite(g).all(), "prior mean produced non-finite gradients"
    assert (g != 0).any(), "prior mean gradient is identically zero"


def test_input_unit_conversion_wiring():
    """Hardware-unit setpoints must be scaled into the sim's native units.

    The sim angle axes work in radians; the EPICS motors in degrees. The prior
    is handed PV-unit ``pos_range`` and the ``pv_mapping.json`` PV->sim factors,
    so the effective sim setpoint at normalized 0.5 must land on the aligned
    baseline and ``input_scale_vec`` must carry the per-axis factors. Regression
    guard for feeding degrees to a radian axis. The two factors below are
    deliberately distinct to prove per-axis (not broadcast) wiring.
    """
    axis_names = ["t1_th1", "t1_chi1"]
    unit_conversion = {"t1_th1": 0.017453293, "t1_chi1": 1.74533e-05}
    range_val = 50e-6 * 180 / 3.141592653589793  # backend pos_range, degrees

    _, prior = build_snd_sim(
        axis_names=axis_names,
        backend_pos_range={n: range_val for n in axis_names},
        unit_conversion=unit_conversion,
    )

    scale = prior.input_scale_vec
    assert torch.allclose(
        scale, torch.tensor([0.017453293, 1.74533e-05], dtype=torch.float64)
    )
    # start_pos is the sim baseline (wm, sim units) expressed in PV units, so
    # start_pos * input_scale recovers the sim-unit aligned baseline.
    sim_start = prior.start_pos_vec * scale
    expected = torch.tensor(
        [getattr(prior._template, n).wm() for n in axis_names], dtype=torch.float64
    )
    assert torch.allclose(sim_start, expected, atol=1e-12)


def test_prior_mean_output_shape(prior):
    """Output must drop the trailing singleton: ``b x n x d`` -> ``b x n``.

    Xopt's CustomMean broadcasts against the ``b x n`` GP batch; returning
    ``b x n x 1`` crashes the posterior prediction strategy (shape mismatch).
    """
    d = len(prior.axis_names)
    x = torch.full((1, 3, d), 0.55, dtype=torch.float64)
    f = prior(x)
    assert f.shape == (1, 3), f"expected (1, 3), got {tuple(f.shape)}"
