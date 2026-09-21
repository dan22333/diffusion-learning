"""Samplers = the *reverse* process (noise -> image). This is where "number of
steps" lives, and where the four formulations in the plan differ.

Run on a plain DDPM-trained net (objective=ddpm):
  - `DDPMSampler` — the foundational baseline, ~T stochastic steps.
  - `DDIMSampler` — same weights, deterministic, *skip* timesteps -> few steps.

Run on a flow-matching-trained net (objective=flow):
  - `FlowMatchingSampler` — integrate the velocity ODE from noise to data.

Run on an EDM-trained net (objective=edm):
  - `EDMSampler`          — Karras σ-space + 2nd-order Heun; what DIAMOND uses.

Every sampler exposes the same `.sample(model, shape, ...) -> images` call, so you
can swap methods and step counts and plot them on shared axes (the Phase 3 race).
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
