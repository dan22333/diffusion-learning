"""Tiny visualization helpers: save a grid of samples and the loss curve.

These are the only "results" Phase 0- produces — a picture of what the model
generates (should look like the memorized toy images) and a loss curve that
collapses toward ~0 (the correctness proof).
"""
from __future__ import annotations

import torch
from torchvision.utils import make_grid, save_image


def save_image_grid(images: torch.Tensor, path: str, nrow: int = 4) -> None:
    """Save a (B,C,H,W) batch in [-1,1] as one PNG grid, rescaled to [0,1]."""
    grid = make_grid(images.clamp(-1, 1).add(1).div(2).cpu(), nrow=nrow)
    save_image(grid, path)


def save_loss_curve(losses: list[float], path: str) -> None:
    """Save the training loss curve as a PNG (log-y so the collapse is visible).

    Imports matplotlib lazily so the core package has no hard plotting dependency."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(losses)
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("MSE loss (log scale)")
    ax.set_title("Overfit-one-batch: loss should collapse toward ~0")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
