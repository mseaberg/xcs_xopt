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


def test_prior_mean_output_shape(prior):
    """Output must drop the trailing singleton: ``b x n x d`` -> ``b x n``.

    Xopt's CustomMean broadcasts against the ``b x n`` GP batch; returning
    ``b x n x 1`` crashes the posterior prediction strategy (shape mismatch).
    """
    d = len(prior.axis_names)
    x = torch.full((1, 3, d), 0.55, dtype=torch.float64)
    f = prior(x)
    assert f.shape == (1, 3), f"expected (1, 3), got {tuple(f.shape)}"
