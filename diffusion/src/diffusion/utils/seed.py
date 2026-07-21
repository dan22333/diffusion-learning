"""Reproducibility helpers.

A frontier-lab run is reproducible: same seed + same config -> same numbers. We
seed every RNG the pipeline touches, and we can snapshot/restore RNG state so a
resumed checkpoint continues the *exact* random stream it left off on.
"""
from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and Torch (CPU + CUDA) so a run is repeatable."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_rng_state() -> dict:
    """Capture the current RNG state of every generator, for checkpointing.

    Saving this alongside model/optimizer state is what lets a resumed run pick up
    the identical random stream (timesteps, noise) rather than diverging."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict) -> None:
    """Restore RNG state captured by `get_rng_state` (used on resume)."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
