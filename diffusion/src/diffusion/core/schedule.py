"""Noise schedules — how much noise the forward process adds at each timestep.

The schedule is a sequence of `betas` (variance added per step). Everything else
(alphas, cumulative products) derives from it. The choice matters: the classic
DDPM `linear` schedule dumps noise too fast at high resolution; the `cosine`
schedule (Nichol & Dhariwal, 2021) spreads it more evenly and is the modern
default. Being able to swap schedules and *see* the difference is exactly the
kind of knob Phase 0.5 explores.
"""
from __future__ import annotations

import math

import torch


def make_beta_schedule(
    kind: str, timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02
) -> torch.Tensor:
    """Return the length-`timesteps` beta schedule of the requested `kind`.

    "linear": betas increase linearly from beta_start to beta_end (original DDPM).
    "cosine": derived so that the *cumulative* signal decays like a cosine curve —
              gentler at the ends, which preserves image structure longer."""
    if kind == "linear":
        return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64).float()

    if kind == "cosine":
        # Build alphas_cumprod from a cosine, then back out the betas.
        s = 0.008  # small offset so beta near t=0 isn't exactly 0
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
        acp = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        acp = acp / acp[0]
        betas = 1 - (acp[1:] / acp[:-1])
        return betas.clamp(1e-8, 0.999).float()

    raise ValueError(f"unknown schedule kind: {kind!r} (expected 'linear' or 'cosine')")
