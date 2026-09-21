"""Flow-matching sampler — integrate the learned velocity ODE from noise to data.

Pairs with `core/flow_matching.py` (the training objective). Training taught the
net a velocity field v_theta(x, t) that transports the noise distribution onto the
data distribution along near-straight paths. Generation is just solving the ODE

    dx/dt = v_theta(x, t)      from t = 0 (pure noise) to t = 1 (data).

Two integrators are offered:
  - "euler"    : first-order, one net call per step. Straight paths make this
                 surprisingly good even at ~20-50 steps.
  - "midpoint" : second-order (two net calls per step); more accurate per step,
                 so it needs fewer steps for the same quality.

Because the trained paths are near-straight, this is the on-ramp to few-step
("rectified") sampling — the reason flow matching is the modern default (SD3/Flux).
"""
from __future__ import annotations

import torch

from ..flow_matching import TIME_SCALE
from .base import Sampler


class FlowMatchingSampler(Sampler):
    """Deterministic ODE sampler for a flow-matching-trained network."""

    def __init__(self, diffusion, method: str = "euler"):
        super().__init__(diffusion)
        if method not in ("euler", "midpoint"):
            raise ValueError(f"unknown integrator {method!r} (expected 'euler'|'midpoint')")
        self.method = method

    @torch.no_grad()
    def sample(self, model, shape, device, num_steps=50, generator=None):
        num_steps = num_steps or 50
        x = torch.randn(shape, device=device, generator=generator)  # x at t=0 is pure noise
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t = i * dt
            t_batch = torch.full((shape[0],), t * TIME_SCALE, device=device)
            if self.method == "euler":
                v = model(x, t_batch)
                x = x + dt * v
            else:  # midpoint: evaluate the field at the half-step for a 2nd-order update
                v1 = model(x, t_batch)
                t_mid = torch.full((shape[0],), (t + 0.5 * dt) * TIME_SCALE, device=device)
                v2 = model(x + 0.5 * dt * v1, t_mid)
                x = x + dt * v2
        return x
