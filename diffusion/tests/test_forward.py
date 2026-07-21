"""Unit tests for the forward process q(x_t | x_0) and its inverse.

These check the core diffusion identity numerically: at t=0 the image is nearly
clean, and averaged over many noise draws x_t has the mean/variance the closed
form predicts. This is the 'overfit one batch' spirit applied to the math itself."""
import torch

from diffusion.config import DiffusionConfig
from diffusion.core import GaussianDiffusion


def _make():
    return GaussianDiffusion(DiffusionConfig(schedule="linear", timesteps=200))


def test_q_sample_shape_and_t0_is_nearly_clean():
    d = _make()
    x0 = torch.randn(4, 3, 16, 16)
    t0 = torch.zeros(4, dtype=torch.long)
    x_t = d.q_sample(x0, t0, torch.randn_like(x0))
    assert x_t.shape == x0.shape
    # at t=0 almost no noise is added, so x_t should sit close to x0
    assert (x_t - x0).abs().mean() < 0.15


def test_q_sample_statistics_match_closed_form():
    """For a fixed x0 and timestep, averaging many noised samples should recover
    mean = sqrt(acp)*x0 and variance = (1 - acp)."""
    d = _make()
    x0 = torch.ones(1, 1, 1, 1) * 0.5
    t = torch.full((20000,), 100, dtype=torch.long)
    x0b = x0.expand(20000, 1, 1, 1)
    x_t = d.q_sample(x0b, t, torch.randn_like(x0b))

    acp = d.alphas_cumprod[100]
    assert torch.allclose(x_t.mean(), (acp.sqrt() * 0.5), atol=0.02)
    assert torch.allclose(x_t.var(unbiased=False), (1 - acp), atol=0.02)


def test_predict_x0_inverts_q_sample():
    """Recovering x0 from (x_t, true eps) must return the original image."""
    d = _make()
    x0 = torch.randn(4, 3, 16, 16)
    t = torch.randint(0, 200, (4,))
    eps = torch.randn_like(x0)
    x_t = d.q_sample(x0, t, eps)
    x0_rec = d.predict_x0_from_eps(x_t, t, eps)
    assert torch.allclose(x0_rec, x0, atol=1e-4)
