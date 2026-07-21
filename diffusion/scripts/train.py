#!/usr/bin/env python
"""Phase 0- entrypoint: overfit one batch and prove the pipeline is correct.

    python scripts/train.py --config configs/overfit_ddpm.yaml

Trains on a single fixed toy batch, saves the loss curve + a sample grid, and
prints whether the loss collapsed (the correctness gate). Success = loss drops by
orders of magnitude and the samples resemble the memorized toy images.
"""
from __future__ import annotations

import argparse
import os

from diffusion.config import Config
from diffusion.core.samplers import DDIMSampler
from diffusion.eval import save_image_grid, save_loss_curve
from diffusion.train import Trainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to a YAML config")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    trainer = Trainer(cfg)

    print(f"device={trainer.device} | params={sum(p.numel() for p in trainer.model.parameters()):,}")
    losses = trainer.train()

    out = cfg.train.out_dir
    save_loss_curve(losses, os.path.join(out, "loss_curve.png"))

    # Sample with DDIM (deterministic, fast) from the EMA weights.
    sampler = DDIMSampler(trainer.diffusion)
    samples = trainer.sample(sampler, num_images=cfg.data.num_images, num_steps=50)
    save_image_grid(samples, os.path.join(out, "samples.png"), nrow=4)
    save_image_grid(trainer.batch, os.path.join(out, "target.png"), nrow=4)

    print(f"\nfirst-step loss: {losses[0]:.5f}  ->  final loss: {losses[-1]:.5f}")
    print(f"loss dropped {losses[0] / max(losses[-1], 1e-9):.0f}x")
    print(f"artifacts written to {out}/ (loss_curve.png, samples.png, target.png)")


if __name__ == "__main__":
    main()
