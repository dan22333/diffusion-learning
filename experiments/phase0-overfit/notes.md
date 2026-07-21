# Notes: phase0-overfit

## Running log

### 2026-07-21

- Observation: schedule + forward-process unit tests passed first try; forward
  q(x_t|x_0) mean/variance matched the closed form to <0.02.
- Observation: per-step loss is noisy (random t each step) — final step alone is a
  bad "did it converge?" signal. Switched test to a 50-draw averaged eval loss.
- Observation: overfit run collapsed 1.08 → ~0.05 (min ~0.017) in 1500 steps on MPS.
- Observation: DDIM(50) samples from EMA land in the learned toy-pattern family but
  are not pixel-sharp copies of the 8 targets (min loss ≈ 0.017 ≠ 0).
- Need to check: whether more steps / longer schedule sharpens memorization (a good
  first "play with the knobs" exercise).

## Things to extract later

- [x] Learning: overfit loss is a noisy single-sample estimate → extracted: [learning](../../lab-notebook/notes/learnings/2026-07-21-overfit-loss-is-noisy.md)

## Assistant session memory deltas

### 2026-07-21 — phase 0- build (short)

- Outcome: raw-PyTorch diffusion package built; 8/8 tests pass; overfit gate passed (~22× loss collapse on MPS).
- Durable change: 1 learning note (noisy overfit loss).
- Follow-up: Phase 0 (move to GPU VM); play with DDIM `--steps`.
