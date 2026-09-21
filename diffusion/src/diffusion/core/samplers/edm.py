"""EDM sampler (Karras et al., 2022) — 2nd-order Heun on the probability-flow ODE.

Pairs with `core/edm.py` (the preconditioned training objective). Sampling solves
the ODE in sigma-space

    dx/dsigma = ( x - D(x, sigma) ) / sigma ,     from sigma_max down to 0,

where D is the preconditioned denoiser. Two things make EDM strong per step:

  1. The **Karras sigma schedule**: the sigma levels are spaced by a power law
     (parameter rho ~ 7) that puts more steps at low noise, where fine detail is
     decided — better quality-per-step than uniform spacing.
  2. A **2nd-order Heun** integrator: take an Euler step, then correct with the
     average of the slope at the start and end of the step. Two network calls per
     step (one at the last step), but so accurate that ~15-35 steps rival a
     1000-step DDPM chain. This is what DIAMOND uses.

`num_steps` here counts sigma intervals; total denoiser calls ~= 2*num_steps - 1.
"""
from __future__ import annotations

import torch

from .base import Sampler


class EDMSampler(Sampler):
    """Deterministic Heun sampler over the Karras sigma schedule."""

    def _sigmas(self, num_steps: int, device) -> torch.Tensor:
        """Karras schedule: num_steps sigmas from sigma_max to sigma_min, then 0."""
        d = self.diffusion
        ramp = torch.linspace(0, 1, num_steps, device=device)
        min_inv = d.sigma_min ** (1.0 / d.rho)
        max_inv = d.sigma_max ** (1.0 / d.rho)
        sigmas = (max_inv + ramp * (min_inv - max_inv)) ** d.rho
        return torch.cat([sigmas, sigmas.new_zeros(1)])  # append sigma=0 (clean)

    @torch.no_grad()
    def sample(self, model, shape, device, num_steps=32, generator=None):
        d = self.diffusion
        num_steps = num_steps or 32
        sigmas = self._sigmas(num_steps, device)

        b = shape[0]
        # start from noise scaled to the largest sigma
        x = torch.randn(shape, device=device, generator=generator) * sigmas[0]

        for i in range(num_steps):
            sigma, sigma_next = sigmas[i], sigmas[i + 1]
            sig_b = torch.full((b,), float(sigma), device=device)
            denoised = d.denoise(model, x, sig_b)
            deriv = (x - denoised) / sigma                 # dx/dsigma at sigma
            x_euler = x + (sigma_next - sigma) * deriv      # Euler predictor
            if sigma_next > 0:                              # Heun corrector (skip at last step)
                sig_next_b = torch.full((b,), float(sigma_next), device=device)
                denoised_next = d.denoise(model, x_euler, sig_next_b)
                deriv_next = (x_euler - denoised_next) / sigma_next
                x = x + (sigma_next - sigma) * 0.5 * (deriv + deriv_next)
            else:
                x = x_euler
        return x
