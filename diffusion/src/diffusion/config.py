"""Typed configuration objects.

Frontier-lab discipline rule #1: *no hardcoded hyperparameters scattered through
the code.* Everything that defines a run lives in one typed, serializable object
that we log alongside the results. YAML files under `configs/` populate these
dataclasses; the dataclass defaults double as documentation of every knob.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import yaml


@dataclass
class ModelConfig:
    """Shape/size of the UNet. Kept tiny by default so Phase 0- runs on a laptop."""

    image_size: int = 16          # H = W of the (square) images
    in_channels: int = 3          # RGB
    base_channels: int = 32       # width of the first UNet level
    channel_mults: tuple = (1, 2) # channel multiplier per resolution level
    num_res_blocks: int = 1       # residual blocks per level
    time_embed_dim: int = 128     # width of the timestep embedding MLP


@dataclass
class DiffusionConfig:
    """The noising process. `timesteps` is T, the length of the DDPM chain."""

    schedule: str = "cosine"      # "linear" | "cosine"
    timesteps: int = 1000
    beta_start: float = 1e-4      # only used by the linear schedule
    beta_end: float = 0.02        # only used by the linear schedule


@dataclass
class DataConfig:
    """What we train on. In Phase 0- this is a fixed synthetic batch we memorize."""

    dataset: str = "toy"          # "toy" (Phase 0-) | "cifar10" (Phase 0.5)
    num_images: int = 8           # size of the single batch we overfit
    data_root: str = "./data"     # where cifar10 would download (Phase 0.5)


@dataclass
class TrainConfig:
    """The optimization loop + the 'right checks' knobs (see LEARNING_PLAN.md)."""

    steps: int = 1500
    batch_size: int = 8
    lr: float = 2e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0        # clip grad-norm; guards against loss spikes
    ema_decay: float = 0.999      # EMA of weights — sampled-from copy
    warmup_steps: int = 100       # linear LR warmup, then hold
    log_every: int = 100
    ckpt_every: int = 500
    seed: int = 0
    device: str = "auto"          # "auto" -> cuda > mps > cpu
    amp: bool = False             # mixed precision (off by default on CPU/MPS)
    out_dir: str = "runs/overfit"
    overfit_one_batch: bool = True  # Phase 0-: reuse ONE batch every step


@dataclass
class Config:
    """Top-level config = the four sub-configs. This is the object we log per run."""

    model: ModelConfig = field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @staticmethod
    def from_dict(raw: dict) -> "Config":
        """Build a Config from a nested dict, filling any omitted key with its default."""
        raw = raw or {}
        return Config(
            model=ModelConfig(**raw.get("model", {})),
            diffusion=DiffusionConfig(**raw.get("diffusion", {})),
            data=DataConfig(**raw.get("data", {})),
            train=TrainConfig(**raw.get("train", {})),
        )

    @staticmethod
    def from_yaml(path: str) -> "Config":
        """Load a config from a YAML file, falling back to the dataclass defaults
        for any section or key the file omits."""
        with open(path) as f:
            return Config.from_dict(yaml.safe_load(f))

    def to_dict(self) -> dict:
        """Plain-dict view — used to dump the resolved config next to the run."""
        return asdict(self)
