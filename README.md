# diffusion-learning

Learning diffusion models end-to-end — from the DDPM math to interactive world models — by **building each piece from scratch** rather than calling into a framework. Structured as a 17-phase plan with two anchor projects: **DIAMOND** (to understand) and **MIRA** (to be current).

The goal is a specific, uncommon combination: **model understanding + systems literacy**. Most people who know diffusion can't read a profiler trace; most people who can read a profiler trace can't explain EDM preconditioning.

---

## Start here

| File | What it is |
|---|---|
| **[`LEARNING_PLAN.md`](LEARNING_PLAN.md)** | **The roadmap.** 17 phases in 9 parts, with datasets, hardware, and per-phase cost estimates. Read this first |
| **[`diffusion-handbook.pdf`](diffusion-handbook.pdf)** | **The reference.** 84 pages, Parts I–X — the probabilistic core, training/sampling/distillation recipes, inference optimisation, real-time video, world models, evaluation, and a corrections log |
| [`diffusion/`](diffusion/) | **The code.** A production-shaped diffusion package built from scratch. See its own [README](diffusion/README.md) |
| [`diamond/`](diamond/) | Vendored [DIAMOND](https://github.com/eloialonso/diamond) — read-only reference for Phase 8 |

The plan is *what to do next*; the handbook is *what we've concluded*. They cross-reference each other.

---

## Status

**Phase 1 complete.** The package exists, the diffusion math is unit-tested, and it overfits 8 real CIFAR-10 images to ~0 loss on Apple Silicon (MPS) — proving the pipeline is correct before spending anything on cloud.

Notable result from getting there: sampling was silently broken while the loss curve looked perfect. Clamping the predicted x₀ to [−1, 1] at every reverse step moved sample MSE from **0.61 → 0.043**. Two lessons that shaped the plan:

- **Loss collapse does not validate generation** — you must sample from pure noise to check.
- **Per-step diffusion loss is too noisy to assert on** — average over many timestep draws.

**Currently implemented:** DDPM (trained) + DDIM (same weights, no retraining). EDM and flow matching are deliberate, documented stubs — they need their own training objectives, which is Phase 3 work.

**Next: Phase 2** — move the same repo, unchanged, to a cheap L4/T4 GCE VM; confirm the overfit still collapses; then learn to tell compute-bound from memory-bandwidth-bound from dataloader-bound in a profiler trace.

---

## The plan at a glance

| Part | Phases |
|---|---|
| **A** Foundations | **1** build the repo, prove correctness *(done)* |
| **B** Fundamentals + GPU | **2** GPU + profiler · **3** DDPM/DDIM/EDM/rectified-flow race · **4** metrics suite |
| **C** Modern architecture | **5** what space diffusion runs in (VAE / VQ / representation autoencoder) · **6** build a DiT |
| **D** Ecosystem | **7** HuggingFace, CFG, ControlNet, LoRA, open weights |
| **E** World models | **8** DIAMOND train→play, paired with IRIS · **9** the forcing family (drift) |
| **F** Scale | **10** DDP → FSDP, 1→8 GPU scaling study, one Vertex job |
| **G** Current stack | **11** MIRA end-to-end, single-player first |
| **H** Make it fast | **12** distillation · **13** sweeps · **14** kernels · **15** quantise + serve |
| **I** Frontier | **16** memory & multiplayer · **17** latent actions (Genie, LAPO) |

Roughly **$350–750** of cloud compute through Phase 10. Phase 11 is the cost cliff.

---

## Quick start

```bash
cd diffusion
python3.10 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]" matplotlib

pytest -q                                                   # schedule math, forward process, overfit
python scripts/train.py --config configs/overfit_cifar.yaml  # 8 real CIFAR images, minutes on MPS
python scripts/sample.py --ckpt runs/overfit_cifar/latest.pt --sampler ddim --steps 20
```

---

## Also in here

**Course materials** — `lecture_*.pdf`, `Lecture_1.pdf`, `midterm*.pdf`, `exam_final.pdf`, `final-solutions.pdf`, `diffusion-derivation.html`.

**Reference papers** — `v2v.pdf`, `orthogonalAdaptation.pdf`, `flightDiffusion.pdf`, `imitating image-image.pdf`.

---

## Conventions

**Production-grade code from run 1.** No notebooks as source, no hardcoded hyperparameters, no "clean it up later." The toy batch and the 50k run share one codebase — only the config and the hardware change. Typed configs, seeded runs, resumable checkpoints (model + optimiser + EMA + step + RNG state), and unit tests on the diffusion math, because a wrong noise schedule is a *silent* bug that FID will not localise for you.

That standard is not aspirational: MIRA — a 5B-parameter frontier world model — ships the same shape (`configs/ src/ tests/ scripts/`, Hydra, `torchrun`, `wandb`, `ruff`, `pytest`).

> ⚠️ **Phase numbering changed on 2026-07-28**, from a fractional scheme to integers 1–17. Mapping: `0−→1, 0→2, 0.5→3, 0.75→4, 0.9→5, 0.95→6, 1→8, 1.5→9, 2→12, 2.5→13, 3→15, 4→10`. [`diffusion/README.md`](diffusion/README.md) still uses the old numbers in places — read "Phase 0−" as Phase 1 and "Phase 0.5" as Phase 3.
