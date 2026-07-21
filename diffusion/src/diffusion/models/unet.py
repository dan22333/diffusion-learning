"""A small but real UNet — the network that learns to denoise.

Every diffusion model needs a net that, given a *noisy* image and *which noise
level* it's at, predicts something that lets us take one step back toward a clean
image. Here that net predicts the noise epsilon (the DDPM parameterization).

Architecture = the standard diffusion UNet, shrunk:
  encoder (downsample) -> bottleneck -> decoder (upsample), with skip connections
  from encoder to decoder, and *every* block conditioned on the timestep.
Kept tiny (few channels, 1-2 resolutions) so Phase 0- overfits in seconds on CPU/MPS.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_norm(channels: int, max_groups: int = 8) -> nn.GroupNorm:
    """GroupNorm with a group count guaranteed to divide `channels`.

    GroupNorm normalizes within groups of channels (batch-size independent, which
    matters for the small batches diffusion often uses). We pick gcd(channels, 8)
    groups so the division is always valid even for tiny channel counts."""
    return nn.GroupNorm(math.gcd(channels, max_groups), channels)


class SinusoidalTimeEmbedding(nn.Module):
    """Turn an integer timestep t into a smooth vector the net can condition on.

    A raw scalar t carries almost no usable signal and doesn't generalize across
    noise levels. We expand it into sinusoids of geometrically-spaced frequencies
    (the Transformer positional-encoding trick): nearby timesteps get nearby codes,
    so the net can share what it learns across adjacent noise levels."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device) / max(half - 1, 1)
        )
        args = t[:, None].float() * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:  # pad if dim is odd so the shape is exactly `dim`
            emb = F.pad(emb, (0, 1))
        return emb


class ResidualBlock(nn.Module):
    """The UNet workhorse: two conv layers + a skip, with the timestep injected.

    norm -> SiLU -> conv, add the projected time embedding, then norm -> SiLU ->
    conv, and finally add the input (a residual/skip connection so gradients flow).
    Adding the time embedding to the feature map is *how the block is told the
    current noise level*."""

    def __init__(self, in_ch: int, out_ch: int, t_dim: int):
        super().__init__()
        self.norm1 = _group_norm(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(t_dim, out_ch)      # time embedding -> per-channel bias
        self.norm2 = _group_norm(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        # 1x1 conv on the skip path when channel counts differ, else identity
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(t_emb))[:, :, None, None]  # broadcast over H,W
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class Downsample(nn.Module):
    """Halve spatial resolution with a strided conv (encoder step down)."""

    def __init__(self, ch: int):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample(nn.Module):
    """Double spatial resolution (nearest-neighbor) then smooth with a conv."""

    def __init__(self, ch: int):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(F.interpolate(x, scale_factor=2, mode="nearest"))


class UNet(nn.Module):
    """Predict the noise epsilon in a noisy image, conditioned on the timestep.

    Forward:  x_t (B,C,H,W) noisy image, t (B,) timestep  ->  eps_hat (B,C,H,W).
    The encoder saves each feature map; the decoder concatenates the matching
    encoder map (skip connection) so fine detail lost during downsampling is
    restored. This is the exact structure real diffusion models use, just small."""

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        channel_mults=(1, 2),
        num_res_blocks: int = 1,
        time_embed_dim: int = 128,
    ):
        super().__init__()
        # Timestep -> embedding -> small MLP (shared by every residual block).
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(base_channels),
            nn.Linear(base_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        self.in_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # ----- encoder: build blocks and remember channel counts for the skips -----
        self.downs = nn.ModuleList()
        skip_channels = [base_channels]
        cur = base_channels
        for i, mult in enumerate(channel_mults):
            out = base_channels * mult
            for _ in range(num_res_blocks):
                self.downs.append(ResidualBlock(cur, out, time_embed_dim))
                cur = out
                skip_channels.append(cur)
            if i != len(channel_mults) - 1:  # no downsample after the last level
                self.downs.append(Downsample(cur))
                skip_channels.append(cur)

        # ----- bottleneck -----
        self.mid1 = ResidualBlock(cur, cur, time_embed_dim)
        self.mid2 = ResidualBlock(cur, cur, time_embed_dim)

        # ----- decoder: mirror the encoder, consuming one skip per residual block -----
        self.ups = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_mults))):
            out = base_channels * mult
            for _ in range(num_res_blocks + 1):  # +1 to also consume the downsample skip
                self.ups.append(ResidualBlock(cur + skip_channels.pop(), out, time_embed_dim))
                cur = out
            if i != 0:
                self.ups.append(Upsample(cur))

        self.out_norm = _group_norm(cur)
        self.out_conv = nn.Conv2d(cur, in_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t)
        h = self.in_conv(x)
        skips = [h]
        for module in self.downs:
            h = module(h, t_emb) if isinstance(module, ResidualBlock) else module(h)
            skips.append(h)
        h = self.mid2(self.mid1(h, t_emb), t_emb)
        for module in self.ups:
            if isinstance(module, ResidualBlock):
                h = module(torch.cat([h, skips.pop()], dim=1), t_emb)
            else:
                h = module(h)
        return self.out_conv(F.silu(self.out_norm(h)))
