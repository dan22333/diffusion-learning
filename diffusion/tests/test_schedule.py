"""Unit tests for the noise schedule — a wrong schedule is a *silent* bug that
FID/quality won't localize, so we pin its properties directly."""
import torch

from diffusion.core.schedule import make_beta_schedule


def test_length_and_range():
    """Betas must have length T and stay strictly inside (0, 1)."""
    for kind in ("linear", "cosine"):
        betas = make_beta_schedule(kind, 200)
        assert betas.shape == (200,)
        assert (betas > 0).all() and (betas < 1).all()


def test_alphas_cumprod_monotonic_decreasing():
    """Cumulative signal \bar{alpha}_t must decay monotonically from ~1 toward ~0:
    more noise as t grows. If it isn't decreasing, the forward process is wrong."""
    for kind in ("linear", "cosine"):
        betas = make_beta_schedule(kind, 200)
        acp = torch.cumprod(1.0 - betas, dim=0)
        assert acp[0] < 1.0 and acp[0] > 0.9      # barely any noise at t=0
        assert acp[-1] < acp[0]                    # much noisier at t=T
        assert (acp[1:] <= acp[:-1] + 1e-6).all()  # never increases


def test_unknown_kind_raises():
    try:
        make_beta_schedule("nope", 10)
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown schedule kind")
