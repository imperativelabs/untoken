import argparse
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class LossWeights:
    reconstruction: float = 1.0
    semantic: float = 0.5
    adversarial: float = 0.0
    compression_ratio: float = 0.3
    gating: float = 0.05
    supervised: float = 0.0


@dataclass
class PhaseConfig:
    epochs: int = 5
    batch_size: int = 128
    lr: float = 2e-5
    grad_accum: int = 1
    scheduler: str = "one_cycle"
    warmup_ratio: float = 0.1
    disc_lr: float = 1e-4
    cosine_restarts: int = 3
    gumbel_temperature: float = 0.8
    gumbel_t_start: float = 1.0
    gumbel_t_end: float = 0.1
    gumbel_hard: bool = False
    freeze_encoder: bool = False
    losses: LossWeights = field(default_factory=LossWeights)


@dataclass
class TrainConfig:
    device: str = "cuda"
    precision: str = "bf16"
    compile: bool = True
    num_workers: int = 16
    seed: int = 42
    data_dir: str = "/data/untoken"
    checkpoint_dir: str = "/data/untoken/checkpoints"
    target_compression_ratio: float = 0.3
    optimizer: str = "adamw"
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.999)
    max_grad_norm: float = 1.0
    semantic_loss_interval: int = 10
    discriminator_real_samples: int = 50000
    phase1: PhaseConfig = field(default_factory=PhaseConfig)
    phase2: PhaseConfig = field(default_factory=lambda: PhaseConfig(
        epochs=15, batch_size=64, grad_accum=2, lr=5e-6,
        scheduler="cosine_with_restarts", losses=LossWeights(adversarial=0.1, supervised=0.0)
    ))
    phase3: PhaseConfig = field(default_factory=lambda: PhaseConfig(
        epochs=5, batch_size=128, lr=1e-6, scheduler="linear_decay",
        gumbel_hard=True, losses=LossWeights(adversarial=0.0, gating=0.0, supervised=0.0)
    ))

    @classmethod
    def from_yaml(cls, path: str) -> "TrainConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)

        def _make_phase(d: dict) -> PhaseConfig:
            losses_d = d.pop("losses", {})
            p = PhaseConfig(**{k: v for k, v in d.items() if k in PhaseConfig.__dataclass_fields__})
            p.losses = LossWeights(**losses_d)
            return p

        cfg = cls()
        for k, v in raw.items():
            if k in ("phase1", "phase2", "phase3"):
                setattr(cfg, k, _make_phase(v))
            elif k == "betas":
                cfg.betas = tuple(v)
            elif k in cls.__dataclass_fields__:
                setattr(cfg, k, v)
        return cfg

    @classmethod
    def from_yaml_with_args(cls, path: str, args: argparse.Namespace) -> "TrainConfig":
        cfg = cls.from_yaml(path)
        for k, v in vars(args).items():
            if v is not None and k in cls.__dataclass_fields__:
                setattr(cfg, k, v)
        return cfg
