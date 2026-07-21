"""The Gaussian diffusion process (DDPM math): forward noising + training loss.

This object owns all the precomputed schedule tensors and the two operations the
*trainer* needs:
  1. `q_sample`      — the forward process: jump straight to a noisy x_t (no loop).
  2. `training_loss` — sample a random timestep, noise the image, ask the net to
                       predict the noise, return the MSE.
The reverse process (sampling) lives in `samplers/`, which read the same buffers.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .schedule import make_beta_schedule


def _extract(coeffs: torch.Tensor, t: torch.Tensor, broadcast_shape) -> torch.Tensor:
    """Gather per-sample schedule coefficients c[t] and reshape for broadcasting.

    `t` is a (B,) batch of timesteps; we pick coeffs[t] and reshape to (B,1,1,1)
    so it multiplies cleanly against a (B,C,H,W) image tensor."""
    out = coeffs.to(t.device).gather(0, t)
    return out.reshape(t.shape[0], *([1] * (len(broadcast_shape) - 1)))


class GaussianDiffusion:
    """DDPM forward process + training objective over a fixed noise schedule."""

    def __init__(self, config):
        betas = make_beta_schedule(
            config.schedule, config.timesteps, config.beta_start, config.beta_end
        )
        self.num_timesteps = len(betas)

        # --- precompute everything the forward process and samplers reuse ---
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)          # \bar{alpha}_t
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.betas = betas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev
        # coefficients for q(x_t | x_0):  x_t = sqrt(acp)*x_0 + sqrt(1-acp)*noise
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        # coefficients for the DDPM reverse (posterior) step
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        self.posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

        self._tensor_names = [
            "betas", "alphas_cumprod", "alphas_cumprod_prev", "sqrt_alphas_cumprod",
            "sqrt_one_minus_alphas_cumprod", "sqrt_recip_alphas", "posterior_variance",
        ]

    def to(self, device) -> "GaussianDiffusion":
        """Move all schedule tensors onto `device` (call once at setup)."""
        for name in self._tensor_names:
            setattr(self, name, getattr(self, name).to(device))
        return self

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Forward process: produce the noisy image x_t directly from x_0.

        The magic of DDPM is this closed form — no need to iterate the chain. We
        interpolate between the clean image and pure Gaussian noise according to
        the schedule at timestep t."""
        sqrt_acp = _extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_1m_acp = _extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return sqrt_acp * x0 + sqrt_1m_acp * noise

    def predict_x0_from_eps(self, x_t, t, eps) -> torch.Tensor:
        """Invert q_sample: recover the implied clean image from x_t and predicted eps.

        Used by the DDIM sampler (and handy for logging what the net 'thinks' the
        clean image is at any step)."""
        sqrt_acp = _extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_1m_acp = _extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        return (x_t - sqrt_1m_acp * eps) / sqrt_acp

    def training_loss(self, model: nn.Module, x0: torch.Tensor) -> torch.Tensor:
        """The DDPM objective: predict the noise you added, and score it by MSE.

        Steps: pick a random timestep per image, draw fresh Gaussian noise, form
        the noisy x_t, ask the net for eps_hat, and return MSE(eps_hat, eps). This
        one line is the entire learning signal — it's identical for 8 images or 8M."""
        b = x0.shape[0]
        t = torch.randint(0, self.num_timesteps, (b,), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        eps_hat = model(x_t, t)
        return F.mse_loss(eps_hat, noise)
