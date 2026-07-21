"""EDM sampler (Karras et al., 2022) — STUB, implemented in Phase 0.5.

Why it's deferred: EDM isn't just a different sampler on the same weights (the way
DDIM is). It reframes the whole problem in continuous σ-space with its own
preconditioning (c_skip / c_in / c_out / c_noise around sigma_data) and trains the
net accordingly — so it needs its own training path, which is Phase 0.5 work, not
the Phase 0- correctness check.

What Phase 0.5 will implement here:
  - Karras σ-schedule: sigmas spaced by `rho` between sigma_min and sigma_max.
  - Preconditioned denoiser wrapper (the c_* terms above).
  - A 2nd-order **Heun** sampler (predictor-corrector) with optional `s_churn`
    stochasticity — SOTA quality-per-step (~10-35 steps).

Why it matters most: **this is exactly what DIAMOND uses** (`diffusion_sampler.py`),
so reproducing it on CIFAR-10 is the direct on-ramp to reading Diamond in Phase 1.
"""
from __future__ import annotations

from .base import Sampler


class EDMSampler(Sampler):
    """Placeholder for the EDM (Karras) Heun sampler — see module docstring."""

    @staticmethod
    def _not_yet():
        raise NotImplementedError(
            "EDM is a Phase 0.5 deliverable (needs σ-space preconditioning + its own "
            "training path). Phase 0- ships DDPM + DDIM. See this module's docstring."
        )

    def sample(self, model, shape, device, num_steps=32, generator=None):
        self._not_yet()
