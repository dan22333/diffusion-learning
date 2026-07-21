---
kind: experiment
status: done
created: 2026-07-21
updated: 2026-07-21
---

# Phase 0- — overfit one batch (local correctness)

## Parent study

[Diffusion full-stack](../../lab-notebook/studies/diffusion-full-stack.md) — stage 0-.

## Question

Is the diffusion pipeline (schedule → forward → UNet → DDPM loss → optimizer →
EMA → sampler) *correct*? Can it drive the loss toward ~0 on a single fixed batch
(memorize), before we spend anything on cloud/GPU?

## Why this matters

"Overfit one batch" is the fastest bug-catcher in ML. If the code can't memorize
8 tiny images, no dataset or GPU will save it. Correctness gate before Phase 0.5.

## Setup

- Code: `diffusion/` package (raw PyTorch), commit at run time
- Config: `diffusion/configs/overfit_ddpm.yaml`
- Model: small UNet, base=32, mults=(1,2), 1 res block, ~635k params
- Diffusion: cosine schedule, T=200, eps-prediction DDPM loss
- Data: fixed synthetic toy batch, 8 × 16×16×3
- Train: 1500 steps, AdamW lr=2e-4, warmup=100, grad-clip=1.0, EMA=0.999
- Device: MPS (Apple Silicon)

## Results

- Loss: 1.08 (step 1) → 0.049 (final), min ~0.017 — **~22× collapse**.
- 8/8 unit tests pass (schedule math, forward q(x_t|x_0) statistics, overfit).
- Artifacts: `diffusion/runs/overfit/{loss_curve,samples,target}.png`.
- Samples (DDIM, 50 steps, EMA weights) land in the learned toy-pattern family.

## Interpretation

Pipeline is correct — it learns. Per-step loss is noisy (one random timestep per
step = high-variance MC estimate), so correctness is judged on the collapse /
low-variance eval, not any single step. Samples are not pixel-sharp copies of the
8 targets (min loss ≈ 0.017 ≠ 0); sharper memorization needs more steps.

## Caveats

- This is memorization, NOT generalization. Novel plausible images need the full
  50k + a real GPU (Phase 0.5).
- Toy synthetic data (offline, deterministic) — CIFAR-10 is Phase 0.5.

## Follow-ups

- [ ] Phase 0: move this exact repo to a GCE VM, learn the GPU systems layer.
- [ ] Play with `scripts/sample.py --steps` (DDIM) to feel steps-vs-quality.
- [ ] Phase 0.5: implement EDM + flow-matching (stubs present), swap toy→CIFAR-10.

## Closeout log

### 2026-07-21

Done. Correctness gate passed (22× loss collapse, tests green). Ready for Phase 0.

## Durable notes

- Learning: [Overfit-one-batch loss is a noisy single-sample estimate](../../lab-notebook/notes/learnings/2026-07-21-overfit-loss-is-noisy.md)
