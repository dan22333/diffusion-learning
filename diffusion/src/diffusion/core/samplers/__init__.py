"""Samplers = the *reverse* process (noise -> image). This is where "number of
steps" lives, and where the four formulations in the plan differ.

Implemented now (Phase 0-), because both run on a plain DDPM-trained net:
  - `DDPMSampler` — the foundational baseline, ~T stochastic steps.
  - `DDIMSampler` — same weights, deterministic, *skip* timesteps -> few steps.

Stubs for Phase 0.5 (different training objective, so deferred on purpose):
  - `EDMSampler`          — Karras σ-space + Heun; what DIAMOND uses.
  - `FlowMatchingSampler` — velocity field along near-straight paths.

Every sampler exposes the same `.sample(model, shape, ...) -> images` call, so you
can swap methods and step counts and plot them on shared axes (the Phase 0.5 race).
"""
from .base import Sampler
from .ddim import DDIMSampler
from .ddpm import DDPMSampler
from .edm import EDMSampler
from .flow_matching import FlowMatchingSampler

SAMPLERS = {
    "ddpm": DDPMSampler,
    "ddim": DDIMSampler,
    "edm": EDMSampler,                     # Phase 0.5
    "flow_matching": FlowMatchingSampler,  # Phase 0.5
}

__all__ = [
    "Sampler", "DDPMSampler", "DDIMSampler", "EDMSampler", "FlowMatchingSampler", "SAMPLERS",
]
