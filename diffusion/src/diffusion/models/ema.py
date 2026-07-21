"""Exponential Moving Average (EMA) of the model weights.

One of the highest-impact, easiest-to-get-wrong details in diffusion training —
the plan flags it as *non-negotiable*. We keep a second, slowly-updated copy of
the weights (the "shadow") and *sample from that*, not from the raw training
weights. The shadow averages out the noise in individual SGD steps, and reliably
gives noticeably better samples / FID than the live weights.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class EMA:
    """Maintain shadow weights `theta_ema <- decay*theta_ema + (1-decay)*theta`."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        # Detached clones of every parameter/buffer, kept on the model's device.
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Nudge the shadow weights a little toward the current live weights.

        Called once per optimizer step. Float tensors are averaged; integer
        buffers (e.g. batchnorm counts) are copied straight through."""
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                self.shadow[k].copy_(v)

    def copy_to(self, model: nn.Module) -> None:
        """Load the shadow weights into `model` (do this before sampling)."""
        model.load_state_dict(self.shadow, strict=True)

    def state_dict(self) -> dict:
        """Shadow weights, for checkpointing."""
        return self.shadow

    def load_state_dict(self, state: dict) -> None:
        """Restore shadow weights from a checkpoint."""
        self.shadow = {k: v.clone() for k, v in state.items()}
