"""The common sampler interface.

Keeping one signature across DDPM/DDIM/EDM/flow-matching is what makes the Phase
0.5 head-to-head race possible: same call, swap the method and the step count,
plot quality-vs-steps on shared axes.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class Sampler(ABC):
    """Base class for a reverse-process sampler bound to a trained diffusion model."""

    def __init__(self, diffusion):
        self.diffusion = diffusion  # holds the schedule tensors

    @abstractmethod
    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: tuple,
        device: torch.device,
        num_steps: int | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Generate a batch of images.

        shape:     (B, C, H, W) of the desired output.
        num_steps: how many denoising steps to take (None -> the method's default).
                   This is *the* knob you'll sweep to see the speed/quality tradeoff.
        generator: optional RNG for reproducible / seed-fixed sampling.
        Returns images in the training range [-1, 1]."""
        raise NotImplementedError
