# diffusion — Phase 0- (local correctness)

A small, production-shaped diffusion codebase. **Phase 0-** proves the pipeline is
*correct* by overfitting a single fixed batch to ~0 loss on CPU/MPS — no cloud, no
GPU. The same package scales unchanged to the full CIFAR-10 run in Phase 0.5; only
the config and the hardware change.

> Memorize (here) ≠ generalize (Phase 0.5). Overfitting 8 images proves the code
> can learn *something*; generating novel plausible images needs the full 50k + a
> real GPU.

## Layout

```
src/diffusion/
  config.py          typed configs — no hardcoded hyperparameters
  models/            unet.py (denoiser), ema.py (sampled-from weights)
  core/              the diffusion MATH:
    schedule.py        noise schedule (linear | cosine)
    gaussian_diffusion.py  forward q(x_t|x_0) + DDPM training loss
    samplers/          reverse process — DDPM, DDIM (now) + EDM, flow-matching (0.5 stubs)
  data/toy.py        the fixed synthetic batch we overfit
  train/trainer.py   the loop: warmup, grad-clip, EMA, NaN guard, resumable ckpts
  eval/grid.py       sample-grid + loss-curve PNGs
configs/overfit_ddpm.yaml
scripts/train.py     overfit one batch (the Phase 0- deliverable)
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
python scripts/train.py --config configs/overfit_ddpm.yaml   # -> runs/overfit/*.png
pytest -q                                                    # math + overfit tests
python scripts/sample.py --ckpt runs/overfit/latest.pt --sampler ddim --steps 20
```

## Samplers now vs Phase 0.5

| Sampler | Status | Note |
|---|---|---|
| DDPM  | ✅ implemented | full-chain stochastic baseline (~T steps) |
| DDIM  | ✅ implemented | same weights, deterministic, **sweep `--steps`** |
| EDM   | 🔜 Phase 0.5 stub | Karras σ-space + Heun; what DIAMOND uses |
| Flow-matching | 🔜 Phase 0.5 stub | velocity field; SD3/Flux default |

DDPM + DDIM already let you feel the core tradeoff (change `--steps`); EDM and
flow-matching need their own training objective, so they land in Phase 0.5.
