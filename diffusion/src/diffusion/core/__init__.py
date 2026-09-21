"""The diffusion *math* (maps to the plan's `diffusion/` box).

- `schedule.py`           — the noise schedule (how much noise at each timestep)
- `gaussian_diffusion.py` — DDPM: forward q(x_t|x_0) + noise-prediction loss
- `flow_matching.py`      — flow matching: straight noise->data path + velocity loss
- `edm.py`                — EDM (Karras): preconditioned denoiser in sigma-space
- `samplers/`             — the reverse process: DDPM, DDIM, flow-matching, EDM

`build_process` turns a `DiffusionConfig` into the objective it names, so the
trainer stays objective-agnostic (`diffusion.objective: ddpm | flow | edm`).
"""
from ..config import DiffusionConfig
from .edm import EDMProcess
from .flow_matching import FlowMatching
from .gaussian_diffusion import GaussianDiffusion
from .schedule import make_beta_schedule

# objective -> the sampler names that are valid for a model trained that way.
# (ddpm-trained weights work with both the DDPM and DDIM samplers.)
SAMPLERS_FOR_OBJECTIVE = {
    "ddpm": ("ddpm", "ddim"),
    "flow": ("flow_matching",),
    "edm": ("edm",),
}


def build_process(cfg: DiffusionConfig):
    """Instantiate the training/forward process named by `cfg.objective`."""
    if cfg.objective == "ddpm":
        return GaussianDiffusion(cfg)
    if cfg.objective == "flow":
        return FlowMatching(cfg)
    if cfg.objective == "edm":
        return EDMProcess(cfg)
    raise ValueError(f"unknown objective {cfg.objective!r} (expected 'ddpm', 'flow', or 'edm')")


__all__ = [
    "GaussianDiffusion", "FlowMatching", "EDMProcess",
    "build_process", "SAMPLERS_FOR_OBJECTIVE", "make_beta_schedule",
]
