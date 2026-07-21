"""Flow-matching sampler — STUB, implemented in Phase 0.5.

Why it's deferred: flow matching changes the *training objective*. Instead of
predicting noise, the net learns a **velocity field** that transports samples
along near-straight paths from noise to data; you then integrate that ODE. Same
reason as EDM — it needs its own training path, so it's Phase 0.5, not the Phase
0- correctness check.

What Phase 0.5 will implement here:
  - Rectified-flow / conditional-flow-matching training target (velocity along the
    straight line x_t = (1-t)*noise + t*data).
  - An ODE integrator (Euler / midpoint) over ~10-30 steps; the straighter paths
    are what later enable 1-4 step "rectified" sampling.

Why it matters: this is the modern default — what SD3 / Flux use — so it's the
fourth contender in the Phase 0.5 four-way race.
"""
from __future__ import annotations

from .base import Sampler


class FlowMatchingSampler(Sampler):
    """Placeholder for the flow-matching ODE sampler — see module docstring."""

    @staticmethod
    def _not_yet():
        raise NotImplementedError(
            "Flow matching is a Phase 0.5 deliverable (velocity-field objective + ODE "
            "integrator, its own training path). Phase 0- ships DDPM + DDIM."
        )

    def sample(self, model, shape, device, num_steps=20, generator=None):
        self._not_yet()
