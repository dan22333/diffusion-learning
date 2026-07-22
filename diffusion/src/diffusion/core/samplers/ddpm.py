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

            # Recover the implied clean image, then CLAMP it to the valid image
            # range before forming the posterior mean. This clamp is the standard
            # stabilizer (Ho et al.'s `clip_denoised`): at the top timesteps 1/sqrt(a)
            # is huge (~31x for a cosine schedule with zero terminal SNR), so tiny eps
            # errors blow x out of [-1, 1] and the chain diverges into noise. Clamping
            # the x0 estimate each step keeps ancestral sampling on the data manifold.
            acp = d.alphas_cumprod[i]
            acp_prev = d.alphas_cumprod_prev[i]
            beta = d.betas[i]
            x0 = (x - d.sqrt_one_minus_alphas_cumprod[i] * eps) / torch.sqrt(acp)
            x0 = x0.clamp(-1.0, 1.0)

            # Posterior mean q(x_{t-1} | x_t, x0) in terms of the clamped x0 and x_t.
            coef_x0 = beta * torch.sqrt(acp_prev) / (1.0 - acp)
            coef_xt = (1.0 - acp_prev) * torch.sqrt(1.0 - beta) / (1.0 - acp)
            mean = coef_x0 * x0 + coef_xt * x

            if i > 0:  # add noise everywhere except the final denoising step
                noise = torch.randn(shape, device=device, generator=generator)
                x = mean + torch.sqrt(d.posterior_variance[i]) * noise
            else:
                x = mean
        return x
