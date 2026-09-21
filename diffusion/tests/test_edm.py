"""EDM (Karras): preconditioning is correct, it overfits, and Heun sampling runs.

Checks the two identities that make EDM work — the denoiser reduces to the input at
sigma->0 (nothing to denoise) and to a pure network prediction at large sigma — then
that the objective overfits one batch and the Heun sampler produces finite images.
Kept tiny so it runs on CPU."""
import torch

from diffusion.config import (
    Config,
    DataConfig,
    DiffusionConfig,
    ModelConfig,
    TrainConfig,
)
from diffusion.core import EDMProcess
from diffusion.core.samplers import EDMSampler
from diffusion.train import Trainer


def _edm_cfg(model_name="unet"):
    model = (ModelConfig(name="unet", image_size=8, base_channels=16,
                         channel_mults=(1, 2), num_res_blocks=1, time_embed_dim=64)
             if model_name == "unet" else
             ModelConfig(name="dit", image_size=8, patch_size=4, hidden_size=64,
                         depth=2, num_heads=4, mlp_ratio=2.0))
    return Config(
        model=model,
        diffusion=DiffusionConfig(objective="edm"),
        data=DataConfig(num_images=4),
        train=TrainConfig(steps=800, batch_size=4, lr=5e-4, warmup_steps=20,
                          log_every=1000, ckpt_every=10000, device="cpu",
                          out_dir="/tmp/diffusion_test_edm"),
    )


def test_edm_preconditioning_limits():
    """At sigma->0, D(x)≈x (c_skip->1, c_out->0). At large sigma, c_skip->0."""
    proc = EDMProcess(DiffusionConfig(objective="edm"))
    small = torch.tensor([1e-4])
    big = torch.tensor([80.0])
    cs_small, co_small, _, _ = proc._coeffs(small)
    cs_big, _, _, _ = proc._coeffs(big)
    assert cs_small.item() > 0.99 and co_small.item() < 1e-3
    assert cs_big.item() < 1e-3


def test_edm_process_is_selected_by_config():
    trainer = Trainer(_edm_cfg())
    assert isinstance(trainer.diffusion, EDMProcess)


def _eval_loss(trainer, reps: int = 50) -> float:
    torch.manual_seed(123)
    return sum(trainer.diffusion.training_loss(trainer.model, trainer.batch).item()
               for _ in range(reps)) / reps


def test_edm_overfits_one_batch():
    torch.manual_seed(0)
    trainer = Trainer(_edm_cfg())
    before = _eval_loss(trainer)
    trainer.train()
    after = _eval_loss(trainer)
    assert after < 0.3 * before, f"EDM loss did not collapse: {before:.3f} -> {after:.3f}"


def test_edm_heun_sampler_runs_finite():
    trainer = Trainer(_edm_cfg())
    trainer.train()
    imgs = trainer.sample(EDMSampler(trainer.diffusion), num_images=4, num_steps=16)
    assert imgs.shape == (4, 3, 8, 8)
    assert torch.isfinite(imgs).all()


def test_edm_works_with_dit_backbone():
    torch.manual_seed(0)
    trainer = Trainer(_edm_cfg("dit"))
    before = _eval_loss(trainer)
    trainer.train()
    assert _eval_loss(trainer) < 0.5 * before
