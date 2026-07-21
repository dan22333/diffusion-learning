"""Evaluation / visualization. Phase 0- only needs sample grids + the loss plot;
the full generative-quality metrics (FID, IS, precision/recall, ...) arrive in
Phase 0.75."""
from .grid import save_image_grid, save_loss_curve

__all__ = ["save_image_grid", "save_loss_curve"]
