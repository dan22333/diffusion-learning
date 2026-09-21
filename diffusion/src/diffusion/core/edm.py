"""EDM (Karras et al., 2022) — a preconditioned denoiser in continuous sigma-space.

The third sibling of `gaussian_diffusion.py` (DDPM) and `flow_matching.py`. Same
trainer-facing interface (`.to(device)`, `.training_loss(model, x0)`), so it's a
one-line config switch (`diffusion.objective: edm`).

The idea (see the handbook / the HTML's EDM section):
  * Parametrize noise by a continuous level sigma, not an integer timestep:
        x = x0 + sigma * eps,   eps ~ N(0, I).
  * Don't have the raw network F predict noise. Wrap it so the *denoiser* D
    (which outputs the clean image) is well-scaled at EVERY sigma:
        D(x, sigma) = c_skip(sigma)*x + c_out(sigma)*F(c_in(sigma)*x, c_noise(sigma))
    with the four c_* derived from one number, sigma_data, so the network's input
    and output are both ~unit-variance regardless of sigma. That normalization —
    not a new loss — is most of EDM's quality win.
  * Train sigma from a log-normal (concentrate on the mid-noise levels that matter)
    and weight the loss by lambda(sigma) so no sigma dominates the gradient.

`denoise()` is shared with the Heun sampler in `samplers/edm.py`.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class EDMProcess:
    """Karras preconditioning + training objective over continuous sigma."""

    def __init__(self, config):
        self.sigma_data = config.sigma_data
        self.sigma_min = config.sigma_min
        self.sigma_max = config.sigma_max
        self.rho = config.rho
        self.P_mean = config.P_mean
        self.P_std = config.P_std

    def to(self, device) -> "EDMProcess":
        return self  # no buffers to move

    # --- the four preconditioning coefficients (Karras et al., Table 1, "EDM") ---
    def _coeffs(self, sigma: torch.Tensor):
        sd2 = self.sigma_data ** 2
        s2 = sigma ** 2
        c_skip = sd2 / (s2 + sd2)
        c_out = sigma * self.sigma_data / torch.sqrt(s2 + sd2)
        c_in = 1.0 / torch.sqrt(s2 + sd2)
        c_noise = 0.25 * torch.log(sigma)   # maps sigma -> the net's time input
        return c_skip, c_out, c_in, c_noise

    def denoise(self, model: nn.Module, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """The preconditioned denoiser D(x, sigma) -> estimate of the clean image.

        `sigma` is a per-sample (B,) tensor. We reshape the coefficients to (B,1,1,1)
        to broadcast over the image, call the raw net on the scaled input, and blend."""
        b = x.shape[0]
        view = (b, *([1] * (x.dim() - 1)))
        c_skip, c_out, c_in, c_noise = self._coeffs(sigma)
        f = model(c_in.reshape(view) * x, c_noise)   # c_noise (B,) is the "time" input
        return c_skip.reshape(view) * x + c_out.reshape(view) * f

    def training_loss(self, model: nn.Module, x0: torch.Tensor) -> torch.Tensor:
        """Weighted denoising loss with log-normal sigma sampling.

        Draw ln(sigma) ~ N(P_mean, P_std^2), noise the image to that level, and
        score the denoiser's clean-image estimate. The weight lambda = (sigma^2 +
        sigma_data^2)/(sigma*sigma_data)^2 exactly cancels c_out^2, so every noise
        level contributes a balanced, unit-scale gradient."""
        b = x0.shape[0]
        view = (b, *([1] * (x0.dim() - 1)))
        # log-normal noise levels
        sigma = torch.exp(self.P_mean + self.P_std * torch.randn(b, device=x0.device))
        noise = torch.randn_like(x0)
        x = x0 + sigma.reshape(view) * noise
        d = self.denoise(model, x, sigma)
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        return (weight.reshape(view) * (d - x0) ** 2).mean()
