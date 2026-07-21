"""The headline Phase 0- test: the pipeline can overfit one batch.

This is the 'fastest bug-catcher in ML' encoded as a test. A tiny model on a tiny
batch for a few hundred steps must drive the loss down by a large factor. If it
can't memorize a handful of images, the pipeline is broken — no GPU or dataset
would fix it. Kept small so it runs in a few seconds on CPU."""
import torch

from diffusion.config import (
    Config,
    DataConfig,
    DiffusionConfig,
    ModelConfig,
    TrainConfig,
)
from diffusion.core.samplers import DDIMSampler
from diffusion.train import Trainer


def _tiny_cfg():
    return Config(
        model=ModelConfig(image_size=8, base_channels=16, channel_mults=(1, 2),
                          num_res_blocks=1, time_embed_dim=64),
        diffusion=DiffusionConfig(schedule="linear", timesteps=50),
        data=DataConfig(num_images=4),
        train=TrainConfig(steps=800, batch_size=4, lr=5e-4, warmup_steps=20,
                          log_every=1000, ckpt_every=10000, device="cpu",
                          out_dir="/tmp/diffusion_test_overfit"),
    )


@torch.no_grad()
def _eval_loss(trainer, reps: int = 50) -> float:
    """Low-variance loss: average the objective over many random timestep/noise
    draws. A single training step is a noisy 1-sample estimate, so we can't assert
    on it directly — this averages that noise out."""
    torch.manual_seed(123)
    return sum(trainer.diffusion.training_loss(trainer.model, trainer.batch).item()
               for _ in range(reps)) / reps


def test_overfits_one_batch():
    torch.manual_seed(0)
    trainer = Trainer(_tiny_cfg())
    before = _eval_loss(trainer)          # ~1.0 at init (predicting unit-variance noise)
    trainer.train()
    after = _eval_loss(trainer)
    # The pipeline must be able to memorize: eval loss collapses far below init.
    assert after < 0.3 * before, f"loss did not collapse: {before:.3f} -> {after:.3f}"
    assert after < 0.2, f"final eval loss too high to call it memorized: {after:.3f}"


def test_sampling_runs_and_produces_finite_images():
    """Sampling from the overfit model must run and produce finite, correctly-shaped
    images. (We don't assert a pixel range — an under-trained toy net can overshoot
    [-1,1]; the grid saver clamps for display. Quality is a Phase 0.5 concern.)"""
    trainer = Trainer(_tiny_cfg())
    trainer.train()
    imgs = trainer.sample(DDIMSampler(trainer.diffusion), num_images=4, num_steps=20)
    assert imgs.shape == (4, 3, 8, 8)
    assert torch.isfinite(imgs).all()
