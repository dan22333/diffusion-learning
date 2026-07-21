---
kind: study
status: running
created: 2026-07-21
updated: 2026-07-21
---

# Diffusion full-stack: training → diffusion → distributed → deploy

## Objective

Learn the full stack end-to-end — training, diffusion, distributed training, and
the frameworks/orchestration around it — to an employable level, using one anchor
project (the DIAMOND diffusion world model) attacked in phases. This study card is
the map; each phase's concrete work hangs off it as a child experiment or session.

See [LEARNING_PLAN.md](../../LEARNING_PLAN.md) for the full rationale.

## Research questions

- Can the diffusion pipeline overfit one batch (correctness) before it can
  generalize (learning)? Where exactly does that boundary sit?
- How do DDPM / DDIM / EDM / flow-matching trade steps for quality on CIFAR-10,
  and do they match published FID targets (EDM \~1.9, DDPM \~3.2)?
- Where do generative-quality metrics (FID, IS, sFID/KID, precision/recall,
  CLIP, LPIPS) disagree, and what does each actually capture?
- How much can distillation + quantization cut sampling cost before quality
  visibly breaks?
- What decides the parallelism (model size) vs the orchestrator (org context)?

## Design commitments

Shared constraints every child experiment inherits:

- **Production-grade code from day one** — one installable package, typed configs
  (no hardcoded hyperparameters), seed control, W&B tracking, checkpoint+resume,
  unit tests on the diffusion math. Toy batch and full 50k run share one codebase;
  only config + hardware change.
- **Overfit one batch first** — correctness gate before any full-data run.
- **The "right checks" apply on the full 50k run, not the toy batch** — LR range
  test + linear scaling rule + warmup, batch-size-to-memory, AdamW/grad-clip,
  EMA (non-negotiable for diffusion), mixed precision, NaN/divergence guards.
- **Validate against published numbers** — a wildly-off FID means a buggy impl.
- **CIFAR-10 as the toy benchmark**; A100-scale (ImageNet-64 / latent) only from
  Phase 1 onward.

## Experiment registry

| Stage | Status | Experiment | Purpose |
|---|---|---|---|
| 0− | planned | _Not created_ | Local (Mac): build repo, unit-test math, overfit one batch to ~0 (correctness) |
| 0 | planned | _Not created_ | Move same repo to cheap L4/T4 GCE VM; learn GPU systems (util, profiler, dataloader stalls) |
| 0.5 | planned | _Not created_ | Full CIFAR-10 (50k): build + race DDPM/DDIM/EDM/flow-matching; steps-vs-quality |
| 0.75 | planned | _Not created_ | Generative-quality metrics suite on the four models; where metrics disagree |
| 1 | planned | _Not created_ | DIAMOND end-to-end on A100: train → sample → play |
| 2 | planned | _Not created_ | Distillation: progressive → consistency → DMD on own checkpoint |
| 2.5 | planned | _Not created_ | Hyperparameter sweeps as a first-class skill (W&B/Optuna/Ray Tune) |
| 3 | planned | _Not created_ | Quantization (int8/fp8 → low-bit, watch it break) + serve in a container |
| 4 | planned | _Not created_ | Distributed training: 4a single-node multi-GPU (DDP→FSDP/DeepSpeed), 4b multi-node |

Every created child experiment links back under `## Parent study`; replace the
placeholder with a relative link and keep the registry current.

## Open decisions

- Which VM tier for Phase 0 (L4 vs T4) — decide at spin-up on price/availability.
- Sweep tool for Phase 2.5 (W&B Sweeps as current default).
- Orchestrator for Phase 4b (Vertex vs Ray-on-GCE).

## Decision log

### 2026-07-21

- Keep the plan's phase order (toy-first is already built in via CIFAR-10 on a
  cheap VM); do NOT reorder to a local-only start, because Phase 0's core goal —
  GPU systems literacy (`nvidia-smi`, profiler) — requires NVIDIA hardware the
  Mac lacks.
- Added Phase 0− (local Mac): prove code correctness by overfitting one batch on
  CPU/MPS before spending on cloud. Memorization ≠ generalization; generalization
  needs the full 50k + a GPU.
- Committed to production-grade ("frontier-lab") code discipline as a cross-cutting
  principle, with tuning checks scoped to the full-data run (Phase 0.5+).

## Assistant session memory deltas

### 2026-07-21 — enable + plan revision (short)

- Outcome: Enabled the notebook; revised LEARNING_PLAN.md with Phase 0− and a
  production-code discipline section; created this study card.
- Durable change: none yet (no experiments run).
- Follow-up: start Phase 0− (build repo + overfit one batch locally).

## Durable notes
