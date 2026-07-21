"""DDPM ancestral sampler — the foundational, slow baseline (~T steps).

Reverses the noising chain one timestep at a time. At each step the net predicts
the noise, we compute the mean of the reverse Gaussian, and (except at the last
step) add a bit of fresh noise back — that stochasticity is why DDPM is the
*stochastic* end of the spectrum. It walks all T timesteps, so it's the slowest
method and the reference the others are judged against.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .base import Sampler


class DDPMSampler(Sampler):
    """Full-chain ancestral sampling as in Ho et al. (2020)."""

    @torch.no_grad()
    def sample(self, model, shape, device, num_steps=None, generator=None):
        d = self.diffusion
        # DDPM uses the whole chain; num_steps is accepted for a uniform interface
        # but ignored (use DDIM if you want fewer steps).
        x = torch.randn(shape, device=device, generator=generator)  # start from pure noise
        for i in reversed(range(d.num_timesteps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            eps = model(x, t)

            # Mean of the reverse step, expressed via the predicted noise.
            beta = d.betas[i]
            sqrt_recip_alpha = d.sqrt_recip_alphas[i]
            sqrt_1m_acp = d.sqrt_one_minus_alphas_cumprod[i]
            mean = sqrt_recip_alpha * (x - beta / sqrt_1m_acp * eps)

            if i > 0:  # add noise everywhere except the final denoising step
                noise = torch.randn(shape, device=device, generator=generator)
                x = mean + torch.sqrt(d.posterior_variance[i]) * noise
            else:
                x = mean
        return x
