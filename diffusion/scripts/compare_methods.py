#!/usr/bin/env python
"""The four-way race: DDPM vs DDIM vs flow matching vs EDM, on one dataset.

This is the Phase-3 experiment in miniature. It trains the THREE training
objectives (DDPM, flow, EDM) on the same fixed batch, then samples each with its
sampler(s) across a sweep of step counts, and reports:

  * training: final (low-variance) loss + wall-clock per objective
  * results:  reconstruction MSE of the samples vs the memorized batch
              (lower = the sampler recovered the data better)
  * cost:     the step count AND the number of network evals (NFE) — the honest
              compute axis, since EDM's Heun and flow's midpoint use 2 evals/step

Note the key point the plan makes: DDPM and DDIM share ONE trained model (DDIM is
just a faster sampler on the same weights), so "4 methods" = "3 training runs".

    python scripts/compare_methods.py                       # cifar10, 8 images (default)
    python scripts/compare_methods.py --dataset toy --steps 1500   # fast synthetic run

Outputs go to runs/compare/: per-method sample grids, target.png, summary.json,
and comparison plots (quality-vs-steps, quality-vs-NFE, train loss curves).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch

from diffusion.config import (
    Config,
    DataConfig,
    DiffusionConfig,
    ModelConfig,
    TrainConfig,
)
from diffusion.core.samplers import (
    DDIMSampler,
    DDPMSampler,
    EDMSampler,
    FlowMatchingSampler,
)
from diffusion.eval import save_image_grid
from diffusion.train import Trainer


# ---- what each trained model is sampled with, and the step sweeps we try ----
# (name, factory(diffusion) -> sampler, [step counts], nfe_per_step_fn)
def _plans(objective):
    if objective == "ddpm":
        return [
            ("ddpm", lambda d: DDPMSampler(d), [None], lambda s, T: T),          # full chain
            ("ddim", lambda d: DDIMSampler(d), [5, 10, 20, 50, 100], lambda s, T: s),
        ]
    if objective == "flow":
        return [
            ("flow-euler", lambda d: FlowMatchingSampler(d, "euler"), [2, 5, 10, 20, 50], lambda s, T: s),
            ("flow-midpoint", lambda d: FlowMatchingSampler(d, "midpoint"), [2, 5, 10, 20], lambda s, T: 2 * s),
        ]
    if objective == "edm":
        return [
            ("edm-heun", lambda d: EDMSampler(d), [4, 8, 16, 32, 64], lambda s, T: 2 * s - 1),
        ]
    return []


def _recon_mse(samples: torch.Tensor, targets: torch.Tensor) -> float:
    """Mean over samples of the min MSE to any target image (in [-1,1])."""
    s = samples.clamp(-1, 1).to(targets.device)
    d = ((s[:, None] - targets[None]) ** 2).mean(dim=(2, 3, 4))   # (B, N)
    return d.min(dim=1).values.mean().item()


def _base_config(dataset: str, train_steps: int) -> dict:
    """Shared model/data/train settings; objective is filled per run."""
    if dataset == "cifar10":
        model = dict(image_size=32, in_channels=3, base_channels=64,
                     channel_mults=[1, 2, 2], num_res_blocks=2, time_embed_dim=256)
        data = dict(dataset="cifar10", num_images=8, data_root="./data")
    else:
        model = dict(image_size=16, in_channels=3, base_channels=32,
                     channel_mults=[1, 2], num_res_blocks=1, time_embed_dim=128)
        data = dict(dataset="toy", num_images=8)
    train = dict(steps=train_steps, batch_size=8, lr=2e-4, ema_decay=0.999,
                 warmup_steps=100, grad_clip=1.0, log_every=max(1, train_steps // 10),
                 ckpt_every=10 ** 9, seed=0, device="auto", overfit_one_batch=True)
    return {"model": model, "data": data, "train": train}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10", choices=["cifar10", "toy"])
    ap.add_argument("--steps", type=int, default=5000, help="training steps per objective")
    ap.add_argument("--timesteps", type=int, default=200, help="DDPM chain length T")
    ap.add_argument("--out", default="runs/compare")
    ap.add_argument("--sample-seed", type=int, default=1234)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    base = _base_config(args.dataset, args.steps)
    summary = {"dataset": args.dataset, "train_steps": args.steps,
               "timesteps": args.timesteps, "runs": {}}
    loss_curves = {}
    trained_batch = None

    for objective in ("ddpm", "flow", "edm"):
        print(f"\n{'='*60}\nTRAIN objective={objective}\n{'='*60}")
        diff = dict(objective=objective, schedule="cosine", timesteps=args.timesteps,
                    sigma_data=0.5, sigma_min=0.002, sigma_max=80.0, rho=7.0,
                    P_mean=-1.2, P_std=1.2)
        cfg = Config.from_dict({**base, "diffusion": diff,
                                "train": {**base["train"], "out_dir": f"{args.out}/{objective}"}})
        t0 = time.time()
        trainer = Trainer(cfg)
        losses = trainer.train()
        train_time = time.time() - t0
        loss_curves[objective] = losses

        # low-variance final loss (average many timestep/noise draws)
        torch.manual_seed(999)
        final_loss = sum(trainer.diffusion.training_loss(trainer.model, trainer.batch).item()
                         for _ in range(50)) / 50
        n_params = sum(p.numel() for p in trainer.model.parameters())
        print(f"  final loss {final_loss:.4f} | {train_time:.1f}s | {n_params:,} params")

        if trained_batch is None:
            trained_batch = trainer.batch.detach().clone()
            save_image_grid(trained_batch, f"{args.out}/target.png", nrow=4)

        # ---- sampler sweeps ----
        sampler_rows = []
        for sname, factory, step_list, nfe_fn in _plans(objective):
            sampler = factory(trainer.diffusion)
            for steps in step_list:
                torch.manual_seed(args.sample_seed)   # same start noise across sweeps
                ts = time.time()
                imgs = trainer.sample(sampler, num_images=8, num_steps=steps)
                dt = time.time() - ts
                eff_steps = args.timesteps if steps is None else steps
                nfe = nfe_fn(eff_steps, args.timesteps)
                mse = _recon_mse(imgs, trained_batch)
                sampler_rows.append({"sampler": sname, "steps": eff_steps, "nfe": nfe,
                                     "recon_mse": mse, "sample_time_s": dt})
                print(f"  {sname:14s} steps={eff_steps:>4} nfe={nfe:>4} "
                      f"recon_mse={mse:.4f} ({dt:.2f}s)")
                # save the best (lowest-mse) grid per sampler name
                tag = f"{objective}_{sname}_{eff_steps}"
                save_image_grid(imgs, f"{args.out}/samples_{tag}.png", nrow=4)

        summary["runs"][objective] = {
            "final_loss": final_loss, "train_time_s": train_time,
            "n_params": n_params, "samplers": sampler_rows,
        }

    with open(f"{args.out}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    _plot(summary, loss_curves, args.out)
    _print_table(summary)
    print(f"\nartifacts in {args.out}/ (summary.json, comparison.png, grids)")


def _print_table(summary: dict) -> None:
    print(f"\n{'='*60}\nBEST RESULT PER SAMPLER (lowest recon MSE)\n{'='*60}")
    print(f"{'sampler':16s}{'steps':>7}{'nfe':>6}{'recon_mse':>12}")
    for obj, r in summary["runs"].items():
        best = {}
        for row in r["samplers"]:
            k = row["sampler"]
            if k not in best or row["recon_mse"] < best[k]["recon_mse"]:
                best[k] = row
        for k, row in best.items():
            print(f"{k:16s}{row['steps']:>7}{row['nfe']:>6}{row['recon_mse']:>12.4f}")


def _plot(summary: dict, loss_curves: dict, out: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"(matplotlib unavailable, skipping plots: {e})")
        return
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    # (1) quality vs steps, (2) quality vs NFE
    for obj, r in summary["runs"].items():
        by = {}
        for row in r["samplers"]:
            by.setdefault(row["sampler"], []).append(row)
        for sname, rows in by.items():
            rows = sorted(rows, key=lambda x: x["steps"])
            ax[0].plot([x["steps"] for x in rows], [x["recon_mse"] for x in rows], "o-", label=sname)
            ax[1].plot([x["nfe"] for x in rows], [x["recon_mse"] for x in rows], "o-", label=sname)
    for a, xl, title in ((ax[0], "sampler steps", "Quality vs steps"),
                         (ax[1], "network evals (NFE)", "Quality vs compute")):
        a.set_xlabel(xl); a.set_ylabel("recon MSE (lower=better)"); a.set_title(title)
        a.set_xscale("log"); a.grid(True, alpha=0.3); a.legend(fontsize=8)
    # (3) training loss curves (normalized to start=1, log-y)
    for obj, losses in loss_curves.items():
        n = max(losses[0], 1e-9)
        ax[2].plot([x / n for x in losses], label=obj)
    ax[2].set_yscale("log"); ax[2].set_xlabel("train step")
    ax[2].set_ylabel("loss / loss[0]"); ax[2].set_title("Training (normalized)")
    ax[2].grid(True, alpha=0.3); ax[2].legend()
    fig.tight_layout(); fig.savefig(f"{out}/comparison.png", dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
