# Research journal

Chronological index of substantial sessions. Newest entries at the bottom.
Each entry: intent, outcome, and a relative link to the study, experiment, or
session record.

## 2026-07-21 — Enabled research notebook

Intent: Stand up a self-contained lab notebook to track learnings across the
diffusion learning plan (Phase 0− → Phase 4).
Outcome: Notebook scaffolded and activated. No experiments run yet; first target
will be Phase 0− (local overfit-one-batch correctness).
Record: [LEARNING_PLAN.md](../LEARNING_PLAN.md).

## 2026-07-21 — Phase 0- built and passing

Intent: Build the production diffusion package and prove the pipeline is correct
by overfitting one batch locally.
Outcome: Raw-PyTorch package under `diffusion/`; DDPM+DDIM samplers (EDM/FM stubs
for 0.5); 8/8 tests pass; overfit run collapsed loss ~22× on MPS. Correctness
gate passed — ready for Phase 0 (GPU VM).
Record: [Phase 0- overfit](../experiments/phase0-overfit/README.md).
