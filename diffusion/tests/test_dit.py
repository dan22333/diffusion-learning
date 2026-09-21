"""The DiT backbone is a genuine drop-in for the UNet.

Two guarantees: (1) it exposes the same forward contract — (x_t, t) in, same-shape
prediction out; (2) with the same training loop it can overfit one batch, i.e. the
config swap `model.name: dit` needs no other change. Kept tiny so it runs on CPU."""
import torch

from diffusion.config import (
    Config,
    DataConfig,
    DiffusionConfig,
    ModelConfig,
    TrainConfig,
)
from diffusion.core.samplers import DDIMSampler
from diffusion.models import DiT, UNet, build_model
from diffusion.train import Trainer


def _dit_cfg():
    return Config(
        model=ModelConfig(name="dit", image_size=8, patch_size=4, in_channels=3,
                          hidden_size=64, depth=2, num_heads=4, mlp_ratio=2.0),
        diffusion=DiffusionConfig(objective="ddpm", schedule="linear", timesteps=50),
        data=DataConfig(num_images=4),
        train=TrainConfig(steps=800, batch_size=4, lr=5e-4, warmup_steps=20,
                          log_every=1000, ckpt_every=10000, device="cpu",
                          out_dir="/tmp/diffusion_test_dit"),
    )


def test_dit_matches_unet_forward_contract():
    """Same (x_t, t) -> same-shape output as the UNet: a real drop-in."""
    x = torch.randn(2, 3, 8, 8)
    t = torch.randint(0, 50, (2,))
    dit = build_model(_dit_cfg().model)
    unet = UNet(in_channels=3, base_channels=16, channel_mults=(1, 2), num_res_blocks=1)
    assert isinstance(dit, DiT)
    assert dit(x, t).shape == unet(x, t).shape == x.shape
    assert torch.isfinite(dit(x, t)).all()


def _eval_loss(trainer, reps: int = 50) -> float:
    torch.manual_seed(123)
    return sum(trainer.diffusion.training_loss(trainer.model, trainer.batch).item()
               for _ in range(reps)) / reps


def test_dit_overfits_one_batch():
    torch.manual_seed(0)
    trainer = Trainer(_dit_cfg())
    before = _eval_loss(trainer)
    trainer.train()
    after = _eval_loss(trainer)
    assert after < 0.3 * before, f"DiT loss did not collapse: {before:.3f} -> {after:.3f}"


def test_dit_sampling_runs_finite():
    trainer = Trainer(_dit_cfg())
    trainer.train()
    imgs = trainer.sample(DDIMSampler(trainer.diffusion), num_images=4, num_steps=20)
    assert imgs.shape == (4, 3, 8, 8)
    assert torch.isfinite(imgs).all()
