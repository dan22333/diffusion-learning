"""Flow matching: the velocity objective trains, overfits, and its ODE sampler runs.

The flow loss magnitude is not comparable to the DDPM eps-MSE (its target is
data - noise), so we assert on the *collapse ratio*, not an absolute value. Both
backbones are exercised to confirm objective and backbone are orthogonal."""
import torch

from diffusion.config import (
    Config,
    DataConfig,
    DiffusionConfig,
    ModelConfig,
    TrainConfig,
)
from diffusion.core import FlowMatching
from diffusion.core.samplers import FlowMatchingSampler
from diffusion.train import Trainer


def _flow_cfg(model_name="unet"):
    model = (ModelConfig(name="unet", image_size=8, base_channels=16,
                         channel_mults=(1, 2), num_res_blocks=1, time_embed_dim=64)
             if model_name == "unet" else
             ModelConfig(name="dit", image_size=8, patch_size=4, hidden_size=64,
                         depth=2, num_heads=4, mlp_ratio=2.0))
    return Config(
        model=model,
        diffusion=DiffusionConfig(objective="flow"),
        data=DataConfig(num_images=4),
        train=TrainConfig(steps=800, batch_size=4, lr=5e-4, warmup_steps=20,
                          log_every=1000, ckpt_every=10000, device="cpu",
                          out_dir="/tmp/diffusion_test_flow"),
    )


def test_flow_process_is_selected_by_config():
    trainer = Trainer(_flow_cfg())
    assert isinstance(trainer.diffusion, FlowMatching)


def _eval_loss(trainer, reps: int = 50) -> float:
    torch.manual_seed(123)
    return sum(trainer.diffusion.training_loss(trainer.model, trainer.batch).item()
               for _ in range(reps)) / reps


def test_flow_overfits_one_batch():
    torch.manual_seed(0)
    trainer = Trainer(_flow_cfg())
    before = _eval_loss(trainer)
    trainer.train()
    after = _eval_loss(trainer)
    assert after < 0.3 * before, f"flow loss did not collapse: {before:.3f} -> {after:.3f}"


def test_flow_sampler_euler_and_midpoint_run_finite():
    trainer = Trainer(_flow_cfg())
    trainer.train()
    for method in ("euler", "midpoint"):
        sampler = FlowMatchingSampler(trainer.diffusion, method=method)
        imgs = trainer.sample(sampler, num_images=4, num_steps=20)
        assert imgs.shape == (4, 3, 8, 8)
        assert torch.isfinite(imgs).all(), f"{method} produced non-finite images"


def test_flow_works_with_dit_backbone():
    """Objective and backbone are orthogonal: flow + DiT trains too."""
    torch.manual_seed(0)
    trainer = Trainer(_flow_cfg("dit"))
    before = _eval_loss(trainer)
    trainer.train()
    assert _eval_loss(trainer) < 0.5 * before
