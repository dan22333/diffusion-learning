"""The training loop — assembles net + diffusion + optimizer + EMA + checkpoints.

This is the piece that carries the "frontier-lab discipline" from the plan:
resolved-config + git-SHA logging, seed control, LR warmup, gradient clipping,
EMA updates, NaN guards, and resumable checkpoints (model + optimizer + EMA +
step + RNG). Phase 0- runs it in `overfit_one_batch` mode: one fixed batch, every
step, driving the loss toward ~0 to prove the pipeline is correct.
"""
from __future__ import annotations

import json
import os
import subprocess

import torch

from ..config import Config
from ..core import GaussianDiffusion
from ..data import make_toy_batch
from ..models import EMA, UNet
from ..utils import get_device, get_rng_state, seed_everything, set_rng_state


def _git_sha() -> str:
    """Best-effort current commit, logged with each run for reproducibility."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


class Trainer:
    """Owns all training state and runs the optimization loop."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        seed_everything(cfg.train.seed)
        self.device = get_device(cfg.train.device)

        # --- model, diffusion process, EMA ---
        m = cfg.model
        self.model = UNet(
            in_channels=m.in_channels,
            base_channels=m.base_channels,
            channel_mults=tuple(m.channel_mults),
            num_res_blocks=m.num_res_blocks,
            time_embed_dim=m.time_embed_dim,
        ).to(self.device)
        self.diffusion = GaussianDiffusion(cfg.diffusion).to(self.device)
        self.ema = EMA(self.model, decay=cfg.train.ema_decay)

        # --- optimizer (AdamW is the diffusion default) ---
        self.opt = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
        )

        # --- the fixed batch we overfit (Phase 0-) ---
        self.batch = make_toy_batch(
            num_images=cfg.data.num_images,
            image_size=m.image_size,
            channels=m.in_channels,
            seed=cfg.train.seed,
        ).to(self.device)

        self.step = 0
        os.makedirs(cfg.train.out_dir, exist_ok=True)
        self._dump_config()

    def _dump_config(self) -> None:
        """Write the fully-resolved config + git SHA next to the run (audit trail)."""
        payload = {"git_sha": _git_sha(), "config": self.cfg.to_dict()}
        with open(os.path.join(self.cfg.train.out_dir, "config.json"), "w") as f:
            json.dump(payload, f, indent=2)

    def _lr_at(self, step: int) -> float:
        """Linear LR warmup for `warmup_steps`, then hold flat.

        Warmup avoids the early-training instability a cold Adam + large LR can
        cause; it's the simplest member of the 'right checks' LR toolkit (cosine
        decay etc. come with the full-data runs)."""
        w = self.cfg.train.warmup_steps
        if w > 0 and step < w:
            return self.cfg.train.lr * (step + 1) / w
        return self.cfg.train.lr

    def train_step(self) -> float:
        """One optimization step: loss -> backward -> clip -> step -> EMA. Returns loss."""
        for group in self.opt.param_groups:  # apply the warmup LR
            group["lr"] = self._lr_at(self.step)

        loss = self.diffusion.training_loss(self.model, self.batch)
        if not torch.isfinite(loss):  # NaN/inf guard — a frontier run fails loud, not silent
            raise FloatingPointError(f"non-finite loss at step {self.step}: {loss.item()}")

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        if self.cfg.train.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip)
        self.opt.step()
        self.ema.update(self.model)
        self.step += 1
        return loss.item()

    def train(self) -> list[float]:
        """Run the full loop; return the loss history (for the deliverable plot)."""
        losses = []
        for _ in range(self.cfg.train.steps):
            loss = self.train_step()
            losses.append(loss)
            if self.step % self.cfg.train.log_every == 0 or self.step == 1:
                print(f"step {self.step:5d} | loss {loss:.5f} | lr {self._lr_at(self.step):.2e}")
            if self.step % self.cfg.train.ckpt_every == 0:
                self.save_checkpoint(os.path.join(self.cfg.train.out_dir, "latest.pt"))
        self.save_checkpoint(os.path.join(self.cfg.train.out_dir, "latest.pt"))
        return losses

    @torch.no_grad()
    def sample(self, sampler, num_images: int = 8, num_steps=None) -> torch.Tensor:
        """Sample from the **EMA** weights (never the raw training weights)."""
        eval_model = UNet(
            in_channels=self.cfg.model.in_channels,
            base_channels=self.cfg.model.base_channels,
            channel_mults=tuple(self.cfg.model.channel_mults),
            num_res_blocks=self.cfg.model.num_res_blocks,
            time_embed_dim=self.cfg.model.time_embed_dim,
        ).to(self.device)
        self.ema.copy_to(eval_model)
        eval_model.eval()
        shape = (num_images, self.cfg.model.in_channels, self.cfg.model.image_size,
                 self.cfg.model.image_size)
        return sampler.sample(eval_model, shape, self.device, num_steps=num_steps)

    def save_checkpoint(self, path: str) -> None:
        """Save everything needed to resume bit-for-bit: weights, opt, EMA, step, RNG."""
        torch.save(
            {
                "step": self.step,
                "model": self.model.state_dict(),
                "opt": self.opt.state_dict(),
                "ema": self.ema.state_dict(),
                "rng": get_rng_state(),
                "config": self.cfg.to_dict(),
            },
            path,
        )

    def load_checkpoint(self, path: str) -> None:
        """Restore a run saved by `save_checkpoint` (resume where it left off)."""
        ckpt = torch.load(path, map_location=self.device)
        self.step = ckpt["step"]
        self.model.load_state_dict(ckpt["model"])
        self.opt.load_state_dict(ckpt["opt"])
        self.ema.load_state_dict(ckpt["ema"])
        set_rng_state(ckpt["rng"])
