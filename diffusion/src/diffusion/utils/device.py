"""Device selection.

Phase 0- runs on whatever the laptop has. On Apple Silicon that's MPS (Metal);
elsewhere CPU. The *same code* picks CUDA automatically once we move to the GPU
VM in Phase 0 — nothing in the model or training loop changes.
"""
from __future__ import annotations

import torch


def get_device(preference: str = "auto") -> torch.device:
    """Resolve a device string to a torch.device.

    "auto" prefers CUDA (the GPU VM, Phase 0+), then MPS (Apple Silicon, Phase 0-),
    then CPU. An explicit string ("cpu"/"mps"/"cuda") is honored as-is."""
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
