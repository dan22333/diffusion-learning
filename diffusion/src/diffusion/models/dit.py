"""A DiT — Diffusion Transformer (Peebles & Xie, 2023) — the modern backbone.

Same job as the UNet (`unet.py`): given a noisy image x_t and its noise level t,
predict the training target (noise eps for the DDPM objective, or the velocity v
for flow matching — the DiT doesn't care which, it just regresses whatever the
process asks for). The *only* difference is the architecture:

  UNet:  conv encoder -> bottleneck -> conv decoder, conditioned by adding the
         time embedding to feature maps. Locality baked in by convolution.
  DiT:   cut the image into patches -> a plain Transformer over the patch tokens
         -> stitch the patches back into an image. No convolutional inductive
         bias; geometry comes from a positional embedding, and conditioning comes
         from **adaLN-Zero**: the time (and, later, class/text) vector produces
         per-block scale/shift/gate parameters that modulate every LayerNorm.

Because the forward signature is identical to the UNet's — `forward(x_t, t) ->
same-shape tensor` — a DiT is a drop-in swap for the UNet we already trained:
flip `model.name: dit` in the config and nothing else in the training loop,
diffusion math, or samplers changes.

Kept small by default (few blocks, narrow width) so it overfits a toy batch on
CPU/MPS, exactly like the tiny UNet.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet import SinusoidalTimeEmbedding


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """adaLN modulation: scale-and-shift a (B,N,D) sequence with (B,D) vectors.

    `x * (1 + scale) + shift`, broadcasting scale/shift across the token axis. The
    `1 +` means "start from the identity transform" — with scale/shift initialized
    to zero (see adaLN-Zero below) the block begins as a no-op, which is what makes
    DiT training stable from step 0."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class Attention(nn.Module):
    """Standard multi-head self-attention over the patch tokens.

    This is *the* thing a convolution can't do: every patch can look at every other
    patch in one layer (global receptive field), which is why transformers scale so
    well. Uses PyTorch's fused scaled-dot-product-attention kernel."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        assert dim % num_heads == 0, "hidden_size must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)   # produce query, key, value in one matmul
        self.proj = nn.Linear(dim, dim)      # mix the heads back together

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        # (B,N,3D) -> 3 x (B, heads, N, head_dim)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        out = F.scaled_dot_product_attention(q, k, v)  # (B, heads, N, head_dim)
        out = out.transpose(1, 2).reshape(B, N, D)     # recombine heads
        return self.proj(out)


class DiTBlock(nn.Module):
    """One Transformer block, conditioned by adaLN-Zero.

    Structure per block: LayerNorm -> attention -> residual, then LayerNorm -> MLP
    -> residual. What makes it a *DiT* block: the LayerNorms have NO learned
    affine; instead the conditioning vector `c` is projected into six vectors —
    (shift, scale, gate) for the attention path and the same three for the MLP
    path — and those modulate the block. The gates multiply each residual branch;
    initializing the projection to zero makes every block start as the identity
    ('adaLN-Zero'), so the untrained DiT is a clean residual stream."""

    def __init__(self, hidden: int, num_heads: int, mlp_ratio: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden, num_heads)
        self.norm2 = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, mlp_hidden), nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, hidden),
        )
        # conditioning -> 6 modulation vectors (shift/scale/gate for attn and mlp)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 6 * hidden))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.ada(c).chunk(6, dim=1)
        x = x + gate_a.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_a, scale_a))
        x = x + gate_m.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_m, scale_m))
        return x


class FinalLayer(nn.Module):
    """Project each token back to raw pixels, with a last adaLN modulation."""

    def __init__(self, hidden: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden, patch_size * patch_size * out_channels)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 2 * hidden))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.ada(c).chunk(2, dim=1)
        return self.linear(modulate(self.norm(x), shift, scale))


class DiT(nn.Module):
    """Diffusion Transformer. Drop-in replacement for `UNet` — same forward API.

    forward(x_t (B,C,H,W), t (B,)) -> prediction (B,C,H,W), where the prediction is
    eps (DDPM objective) or velocity (flow objective). H and W must be divisible by
    `patch_size`."""

    def __init__(
        self,
        image_size: int = 16,
        patch_size: int = 4,
        in_channels: int = 3,
        hidden_size: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.grid = image_size // patch_size          # tokens per side
        self.num_patches = self.grid * self.grid

        # Patchify: a stride-`patch` conv turns each patch into one token vector.
        self.x_embed = nn.Conv2d(in_channels, hidden_size, patch_size, stride=patch_size)
        # Learned positional embedding — the transformer has no built-in geometry.
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size))
        # Timestep -> conditioning vector (same sinusoidal trick as the UNet).
        self.t_embed = nn.Sequential(
            SinusoidalTimeEmbedding(hidden_size),
            nn.Linear(hidden_size, hidden_size), nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.final = FinalLayer(hidden_size, patch_size, in_channels)
        self._init_weights()

    def _init_weights(self) -> None:
        """Standard DiT init: small pos-embed, and ZERO the adaLN/final projections
        so every block and the output start as the identity — the 'adaLN-Zero' that
        keeps training stable."""
        nn.init.normal_(self.pos_embed, std=0.02)
        for block in self.blocks:
            nn.init.zeros_(block.ada[-1].weight)
            nn.init.zeros_(block.ada[-1].bias)
        nn.init.zeros_(self.final.ada[-1].weight)
        nn.init.zeros_(self.final.ada[-1].bias)
        nn.init.zeros_(self.final.linear.weight)
        nn.init.zeros_(self.final.linear.bias)

    def _unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """(B, N, patch*patch*C) tokens -> (B, C, H, W) image."""
        B = x.shape[0]
        p, c, g = self.patch_size, self.in_channels, self.grid
        x = x.reshape(B, g, g, p, p, c)
        x = torch.einsum("bhwpqc->bchpwq", x)
        return x.reshape(B, c, g * p, g * p)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = self.x_embed(x).flatten(2).transpose(1, 2)   # (B, N, hidden)
        x = x + self.pos_embed
        c = self.t_embed(t)                              # (B, hidden) conditioning
        for block in self.blocks:
            x = block(x, c)
        x = self.final(x, c)                             # (B, N, patch*patch*C)
        return self._unpatchify(x)
