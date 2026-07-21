"""The diffusion *math* (maps to the plan's `diffusion/` box).

- `schedule.py`         — the noise schedule (how much noise at each timestep)
- `gaussian_diffusion.py` — forward process q(x_t|x_0) + the training loss (DDPM)
- `samplers/`           — the reverse process: DDPM, DDIM (+ EDM/flow-matching stubs)
"""
from .gaussian_diffusion import GaussianDiffusion
from .schedule import make_beta_schedule

__all__ = ["GaussianDiffusion", "make_beta_schedule"]
