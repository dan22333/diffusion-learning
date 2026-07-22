"""A fixed batch of real CIFAR-10 images to overfit (Phase 0-, real-image variant).

The toy batch (`toy.py`) proves the pipeline is *coded* right using smooth
synthetic gradients. This module runs the same "can it memorize one batch?" test
on *real* images, whose high-frequency detail and natural statistics are a harder,
more honest correctness check before the full Phase 0.5 CIFAR-10 learning run.

We deliberately do NOT pull the whole 162 MB CIFAR-10 tarball (torchvision's
default, and painfully slow from the canonical Toronto host). Phase 0- needs only
a handful of images, so we fetch exactly `num_images` from the HuggingFace
datasets-server — a few tiny 32x32 files — then cache the assembled batch to disk
so later runs are instant and fully offline. The full dataset arrives in Phase 0.5.
"""
from __future__ import annotations

import io
import json
import os
import urllib.request

import numpy as np
import torch
from PIL import Image

_HF_DATASET = "uoft-cs/cifar10"
_HF_CONFIG = "plain_text"
_ROWS_API = "https://datasets-server.huggingface.co/rows"


def _fetch_first_images(num_images: int, timeout: float = 30.0) -> list[Image.Image]:
    """Return the first `num_images` CIFAR-10 train images (deterministic order)
    via the HuggingFace datasets-server, decoded as PIL RGB images."""
    query = (
        f"{_ROWS_API}?dataset={_HF_DATASET}&config={_HF_CONFIG}"
        f"&split=train&offset=0&length={num_images}"
    )
    with urllib.request.urlopen(query, timeout=timeout) as resp:
        rows = json.load(resp)["rows"]
    if len(rows) < num_images:
        raise RuntimeError(f"asked for {num_images} images, server returned {len(rows)}")

    images = []
    for row in rows[:num_images]:
        src = row["row"]["img"]["src"]
        with urllib.request.urlopen(src, timeout=timeout) as img_resp:
            images.append(Image.open(io.BytesIO(img_resp.read())).convert("RGB"))
    return images


def make_cifar_batch(
    num_images: int = 8,
    image_size: int = 32,
    channels: int = 3,
    seed: int = 0,
    data_root: str = "./data",
) -> torch.Tensor:
    """Return a deterministic (num_images, channels, image_size, image_size) batch
    of real CIFAR-10 images in the standard diffusion input range [-1, 1].

    `channels` must be 3 (CIFAR-10 is RGB); it exists only to mirror
    `make_toy_batch`'s signature so the trainer can call either uniformly. Images
    are resized to `image_size` (32 is native — no resampling in that case). The
    `seed` is unused (the first `num_images` are a fixed target) but kept for
    signature parity. The assembled batch is cached under `data_root`.
    """
    if channels != 3:
        raise ValueError(f"CIFAR-10 is RGB; channels must be 3, got {channels}")

    cache = os.path.join(data_root, f"cifar_overfit_n{num_images}_s{image_size}.pt")
    if os.path.exists(cache):
        return torch.load(cache)

    pil_images = _fetch_first_images(num_images)

    tensors = []
    for img in pil_images:
        if img.size != (image_size, image_size):
            img = img.resize((image_size, image_size), Image.BICUBIC)
        arr = np.asarray(img, dtype=np.float32) / 255.0        # [0,1], (H, W, C)
        t = torch.from_numpy(arr).permute(2, 0, 1)             # -> (C, H, W)
        tensors.append(t * 2.0 - 1.0)                          # [0,1] -> [-1,1]

    batch = torch.stack(tensors, dim=0).clamp(-1.0, 1.0)

    os.makedirs(data_root, exist_ok=True)
    torch.save(batch, cache)
    return batch
