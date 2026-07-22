"""The Phase 0- toy batch: a few fixed, simple, *distinct* images to overfit.

Why synthetic instead of CIFAR-10 here? Phase 0- tests *code correctness*, not
learning. A tiny deterministic batch (a) needs no download, so unit tests run
offline, and (b) is made of smooth, low-frequency patterns so that a *memorizing*
model reproduces something recognizable — you can eyeball success. Generalization
to real images is Phase 0.5 (full CIFAR-10 + a GPU), not here.
"""
from __future__ import annotations

import math

import torch


def make_toy_batch(
    num_images: int = 8, image_size: int = 16, channels: int = 3, seed: int = 0
) -> torch.Tensor:
    """Return a deterministic (num_images, channels, image_size, image_size) batch
    in the standard diffusion input range [-1, 1].

    Each image is a distinct low-frequency pattern: per-image spatial frequencies
    and phase offsets drive smooth color gradients (plus a soft radial blob), so
    the images are easy to tell apart and cheap for a small net to memorize."""
    g = torch.Generator().manual_seed(seed)
    coords = torch.linspace(-1.0, 1.0, image_size)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")  # (H, W) each
    radius = torch.sqrt(xx**2 + yy**2)

    images = []
    for _ in range(num_images):
        # Random-but-fixed frequencies/phases per channel give each image its look.
        freqs = torch.rand(channels, 2, generator=g) * 2.5 + 0.5
        phases = torch.rand(channels, generator=g) * 2 * math.pi
        blob_sign = (torch.rand(1, generator=g).item() > 0.5) * 2 - 1

        chans = []
        for c in range(channels):
            wave = torch.sin(freqs[c, 0] * math.pi * xx + phases[c]) * torch.cos(
                freqs[c, 1] * math.pi * yy + phases[c]
            )
            blob = blob_sign * torch.exp(-(radius**2) * 3.0)  # soft centered blob
            chans.append(0.7 * wave + 0.3 * blob)
        images.append(torch.stack(chans, dim=0))

    batch = torch.stack(images, dim=0)
    return batch.clamp(-1.0, 1.0)
