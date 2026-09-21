"""Model registry — turn a `ModelConfig` into the right backbone.

This closes the Phase-1 'known gap': the trainer used to construct `UNet(...)`
directly (twice), so adding a second backbone would have meant editing the loop.
Now the backbone is a config line — `model.name: unet` or `model.name: dit` —
and the training loop, diffusion math, and samplers are all backbone-agnostic
because both nets share the same `forward(x_t, t) -> prediction` signature.
"""
from __future__ import annotations

import torch.nn as nn

from ..config import ModelConfig
from .dit import DiT
from .unet import UNet


def build_model(cfg: ModelConfig) -> nn.Module:
    """Instantiate the backbone named by `cfg.name`, reading only its own fields."""
    if cfg.name == "unet":
        return UNet(
            in_channels=cfg.in_channels,
            base_channels=cfg.base_channels,
            channel_mults=tuple(cfg.channel_mults),
            num_res_blocks=cfg.num_res_blocks,
            time_embed_dim=cfg.time_embed_dim,
        )
    if cfg.name == "dit":
        return DiT(
            image_size=cfg.image_size,
            patch_size=cfg.patch_size,
            in_channels=cfg.in_channels,
            hidden_size=cfg.hidden_size,
            depth=cfg.depth,
            num_heads=cfg.num_heads,
            mlp_ratio=cfg.mlp_ratio,
        )
    raise ValueError(f"unknown model name {cfg.name!r} (expected 'unet' or 'dit')")
