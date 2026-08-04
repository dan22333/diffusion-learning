# diffusion — Phase 1 (local correctness)

A small, production-shaped diffusion codebase. **Phase 1** proves the pipeline is
*correct* by overfitting a single fixed batch to ~0 loss on CPU/MPS — no cloud, no
GPU. The same package moves unchanged to a GPU in **Phase 2** and to the full
CIFAR-10 four-way race in **Phase 3**; only the config and the hardware change.

> Memorize (here) ≠ generalize (Phase 3). Overfitting 8 images proves the code
> can learn *something*; generating novel plausible images needs the full 50k + a
> real GPU.

See [`../LEARNING_PLAN.md`](../LEARNING_PLAN.md) for the full 17-phase plan and
[`../diffusion-handbook.pdf`](../diffusion-handbook.pdf) for the reference material.

## Layout

```
src/diffusion/
  config.py          typed configs — no hardcoded hyperparameters
  models/            unet.py (denoiser), ema.py (sampled-from weights)
  core/              the diffusion MATH:
    schedule.py        noise schedule (linear | cosine)
    gaussian_diffusion.py  forward q(x_t|x_0) + DDPM training loss
    samplers/          reverse process — DDPM, DDIM (now) + EDM, flow-matching (Phase 3 stubs)
  data/
    toy.py             fixed synthetic batch (smooth gradients — the first sanity check)
    cifar.py           8 real CIFAR-10 images (high-frequency detail, natural statistics)
  train/trainer.py   the loop: warmup, grad-clip, EMA, NaN guard, resumable ckpts
  eval/grid.py       sample-grid + loss-curve PNGs
  utils/             device.py (auto -> mps/cuda/cpu), seed.py
configs/
  overfit_ddpm.yaml    toy batch — fastest correctness gate
  overfit_cifar.yaml   8 real CIFAR-10 images — the Phase 1 deliverable
scripts/train.py     overfit one batch
scripts/sample.py    play with --sampler and --steps
tests/               schedule math, forward process, overfit-one-batch
```

## Setup

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]" matplotlib
```

## Run

```bash
pytest -q                                                     # math + overfit tests

# toy batch first (fast), then real images
python scripts/train.py --config configs/overfit_ddpm.yaml    # -> runs/overfit/*.png
python scripts/train.py --config configs/overfit_cifar.yaml   # -> runs/overfit_cifar/*.png

python scripts/sample.py --ckpt runs/overfit_cifar/latest.pt --sampler ddim --steps 20
```

## Samplers: now vs Phase 3

| Sampler | Status | Note |
|---|---|---|
| DDPM  | ✅ implemented | full-chain stochastic baseline (~T steps) |
| DDIM  | ✅ implemented | **same weights**, deterministic, **sweep `--steps`** |
| EDM   | 🔜 Phase 3 stub | Karras σ-space + Heun; what DIAMOND uses |
| Flow-matching | 🔜 Phase 3 stub | velocity field; do it as **rectified flow** (SD3/Flux) |

DDPM + DDIM already let you feel the core tradeoff (change `--steps`) — and the
reason is worth internalising: **DDIM is a *sampler*, not a model.** It reuses the
DDPM checkpoint with no retraining. EDM and flow matching each reframe the training
objective, so they need their own training runs, which is why Phase 3 trains
**three** models rather than four.

## Two results worth knowing before you touch this code

**1. Loss collapse does not validate generation.** The overfit loss went to ~0 while
sampling was still broken. Clamping the predicted x₀ to [−1, 1] at every reverse
step moved sample MSE from **0.61 → 0.043**. `tests/test_overfit.py` therefore
samples from *pure noise* rather than only asserting on the loss.

**2. Per-step diffusion loss is too noisy to assert on directly** — average over
many timestep draws and read curves as trends.

And one diagnosis recorded in `configs/overfit_cifar.yaml`: it uses
`timesteps: 200`, not 1000, because with only 8 images each noise level needs
enough training visits. At 1000 the reverse chain recovered images perfectly from
t₀ ≤ 600 but **diverged from pure noise**, because t ≈ 900–999 were undertrained.
Phase 3 uses the full 1000.

## Known gap

`train/trainer.py:44` and `:136` construct `UNet(...)` **directly**, and `:136`
re-lists every model field by hand to build the EMA evaluation copy. This needs a
`build_model(cfg.model)` registry plus `ModelConfig.name`, so that adding the DiT in
**Phase 6** is a config line rather than a refactor — and it removes the
duplication as a side effect. Listed as a Phase 1 deliverable in the plan.
