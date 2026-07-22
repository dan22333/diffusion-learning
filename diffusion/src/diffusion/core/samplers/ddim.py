"""DDIM sampler — same trained weights, deterministic, and *skips* timesteps.

DDIM (Song et al., 2021) reinterprets the reverse process as an ODE you can solve
on a *subset* of timesteps — so the same net that needed ~1000 DDPM steps can make
a good image in 20-50. With eta=0 it's fully deterministic (fix the seed -> same
image every time; also enables latent interpolation). This is the sampler to reach
for when you want to *play with the number of steps* — just change `num_steps`.
"""
from __future__ import annotations

import torch

from .base import Sampler


class DDIMSampler(Sampler):
    """Deterministic (eta=0 by default) accelerated sampler on DDPM weights."""

    def __init__(self, diffusion, eta: float = 0.0):
        super().__init__(diffusion)
        self.eta = eta  # 0 = deterministic DDIM; >0 reintroduces stochasticity

    @torch.no_grad()
    def sample(self, model, shape, device, num_steps=50, generator=None):
        d = self.diffusion
        num_steps = num_steps or 50

        # Choose `num_steps` timesteps evenly spaced over the full [0, T) chain.
        # Fewer steps = faster but coarser: this list IS the speed/quality knob.
        step_indices = torch.linspace(0, d.num_timesteps - 1, num_steps, device=device)
        step_indices = step_indices.long().flip(0)  # go from noisy -> clean

        x = torch.randn(shape, device=device, generator=generator)
        for j, i in enumerate(step_indices):
            t = torch.full((shape[0],), int(i), device=device, dtype=torch.long)
            eps = model(x, t)

            acp_t = d.alphas_cumprod[i]
            # the previous (less noisy) timestep in our sub-schedule, or clean at the end
            i_prev = step_indices[j + 1] if j + 1 < len(step_indices) else torch.tensor(-1)
            acp_prev = d.alphas_cumprod[i_prev] if i_prev >= 0 else torch.tensor(1.0, device=device)

            # Predict the clean image, then re-noise it to the previous timestep.
            # Clamp the x0 estimate to the valid range (Ho et al.'s `clip_denoised`)
            # — keeps the trajectory on the data manifold and stops the top-timestep
            # 1/sqrt(acp) blow-up from diverging into noise.
            x0_pred = (x - torch.sqrt(1 - acp_t) * eps) / torch.sqrt(acp_t)
            x0_pred = x0_pred.clamp(-1.0, 1.0)

            # eta controls how much stochastic noise to reinject (0 -> deterministic).
            sigma = self.eta * torch.sqrt(
                (1 - acp_prev) / (1 - acp_t) * (1 - acp_t / acp_prev)
            )
            dir_xt = torch.sqrt((1 - acp_prev - sigma**2).clamp(min=0)) * eps
            x = torch.sqrt(acp_prev) * x0_pred + dir_xt
            if self.eta > 0 and i_prev >= 0:
                x = x + sigma * torch.randn(shape, device=device, generator=generator)
        return x
