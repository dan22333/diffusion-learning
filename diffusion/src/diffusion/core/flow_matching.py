"""Flow matching (rectified flow) — the modern training objective.

This is the sibling of `gaussian_diffusion.py`. Where DDPM defines a *noising
chain* and trains the net to predict the noise eps, flow matching defines a
*straight-line path* from noise to data and trains the net to predict the
**velocity** along it. It exposes the exact same two-method interface the trainer
needs — `.to(device)` and `.training_loss(model, x0)` — so switching objectives is
a one-line config change (`diffusion.objective: flow`).

The path (rectified flow / conditional-flow-matching, the SD3 / Flux choice):

    t ~ U(0, 1),   noise ~ N(0, I)
    x_t = (1 - t) * noise + t * data          # t=0 -> pure noise, t=1 -> data
    target velocity  v = data - noise         # constant along each straight line

The net regresses v. The L2 minimizer of this per-sample target is the
conditional expectation E[data - noise | x_t] — which is exactly the marginal
velocity field that transports the noise distribution onto the data distribution.
So regressing the trivial per-sample velocity yields the correct global field.
Sampling then integrates dx/dt = v_theta(x, t) from t=0 to t=1 (see
`samplers/flow_matching.py`).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# The network's time embedding was designed for DDPM's integer timesteps (0..T).
# Flow matching's t lives in [0, 1], which would make the sinusoidal frequencies
# almost constant. We rescale t by this factor before feeding the net so both
# objectives excite the embedding over a comparable range. Training and sampling
# MUST use the same scale — hence a single shared constant.
TIME_SCALE = 1000.0


class FlowMatching:
    """Rectified-flow objective over the straight noise->data path.

    Mirrors `GaussianDiffusion`'s trainer-facing interface. Holds no schedule
    tensors (the path needs none), so `.to()` is a no-op that returns self."""

    def __init__(self, config):
        # timesteps/schedule are irrelevant to flow matching; kept only so the same
        # DiffusionConfig can describe either objective.
        self.time_scale = TIME_SCALE

    def to(self, device) -> "FlowMatching":
        return self  # no buffers to move

    def training_loss(self, model: nn.Module, x0: torch.Tensor) -> torch.Tensor:
        """Predict the velocity of the straight noise->data path; score by MSE.

        `x0` is the clean data batch (the trainer's name for it). We draw a random
        time and a random noise per image, place x_t on the line between them, and
        ask the net for the velocity data - noise."""
        b = x0.shape[0]
        data = x0
        noise = torch.randn_like(data)
        t = torch.rand(b, device=data.device)                 # continuous t ~ U(0,1)
        t_b = t.reshape(b, *([1] * (data.dim() - 1)))         # (B,1,1,1) for broadcasting
        x_t = (1.0 - t_b) * noise + t_b * data                # point on the straight path
        target = data - noise                                 # constant velocity along it
        v_pred = model(x_t, t * self.time_scale)              # net sees rescaled time
        return F.mse_loss(v_pred, target)
