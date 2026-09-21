#!/usr/bin/env python
"""Sample from a trained checkpoint — and play with sampler + number of steps.

    # DDIM in 20 steps:
    python scripts/sample.py --ckpt runs/overfit/latest.pt --sampler ddim --steps 20
    # DDPM full chain (slow, stochastic):
    python scripts/sample.py --ckpt runs/overfit/latest.pt --sampler ddpm

This is the knob-turning tool the plan wants: swap `--sampler` and `--steps` and
watch how quality and wall-clock change. (EDM / flow_matching are Phase 0.5 stubs
and will raise a clear NotImplementedError for now.)
"""
from __future__ import annotations

import argparse
import time

import torch

from diffusion.config import Config
from diffusion.core import SAMPLERS_FOR_OBJECTIVE, build_process
from diffusion.core.samplers import SAMPLERS
from diffusion.eval import save_image_grid
from diffusion.models import build_model
from diffusion.utils import get_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--sampler", default="ddim", choices=list(SAMPLERS))
    parser.add_argument("--steps", type=int, default=50, help="number of denoising steps")
    parser.add_argument("--num-images", type=int, default=8)
    parser.add_argument("--out", default="sample_grid.png")
    args = parser.parse_args()

    # weights_only=False: our checkpoints store the Config (with numpy scalars),
    # which PyTorch 2.6's safe unpickler rejects. These are our own trusted files.
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = Config.from_dict(ckpt["config"])
    device = get_device(cfg.train.device)

    # Rebuild the net (unet|dit, per the saved config) and load the EMA weights.
    model = build_model(cfg.model).to(device)
    model.load_state_dict(ckpt["ema"])
    model.eval()

    # The sampler must match the objective the model was trained with.
    obj = cfg.diffusion.objective
    allowed = SAMPLERS_FOR_OBJECTIVE[obj]
    if args.sampler not in allowed:
        raise SystemExit(
            f"checkpoint objective={obj!r} supports samplers {allowed}; got --sampler {args.sampler!r}")

    diffusion = build_process(cfg.diffusion).to(device)
    sampler = SAMPLERS[args.sampler](diffusion)
    shape = (args.num_images, cfg.model.in_channels, cfg.model.image_size, cfg.model.image_size)

    t0 = time.time()
    images = sampler.sample(model, shape, device, num_steps=args.steps)
    dt = time.time() - t0

    save_image_grid(images, args.out, nrow=4)
    print(f"{args.sampler} | {args.steps} steps | {dt:.2f}s "
          f"({dt / args.num_images * 1000:.0f} ms/image) -> {args.out}")


if __name__ == "__main__":
    main()
