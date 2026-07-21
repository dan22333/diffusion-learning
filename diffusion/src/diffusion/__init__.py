"""Diffusion learning lab.

A small, production-shaped diffusion codebase. Phase 0- proves the pipeline is
*correct* (overfit one batch to ~0 loss on CPU/MPS). The same package scales
unchanged to the full CIFAR-10 run in Phase 0.5 — only config + hardware change.

Package map (mirrors LEARNING_PLAN.md's layout):
    models/   — the neural net (UNet) + EMA of its weights
    core/     — the diffusion *math*: noise schedule, forward/reverse, samplers
    data/     — datasets / the fixed toy batch we overfit
    train/    — the training loop, checkpointing, EMA wiring
    utils/    — seeding, device selection
"""

__version__ = "0.0.1"
