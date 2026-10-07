from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass(frozen=True)
class SystemConfig:
    bs_antennas: int
    ris_elements: int
    users: int
    noise_power: float
    transmit_power: float
    rician_k: float
    csi_error_scale: float
    phase_bits: int
    temporal_rho: float = 0.0


@dataclass(frozen=True)
class EvaluationConfig:
    episodes: int
    channel_realizations: int
    outage_quantile: float
    seed: int
    trajectory_steps: int = 1
    outage_threshold: float = 0.5
    backend: str = "numpy"
    matched_candidate_budget: int = 36


@dataclass(frozen=True)
class TrainingConfig:
    hidden_dim: int
    learning_rate: float
    steps: int
    train_error_scales: tuple[float, ...] = (0.0, 0.04, 0.08, 0.12)
    held_out_error_scales: tuple[float, ...] = (0.20,)


@dataclass(frozen=True)
class Config:
    system: SystemConfig
    evaluation: EvaluationConfig
    training: TrainingConfig


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return Config(
        system=SystemConfig(**raw["system"]),
        evaluation=EvaluationConfig(**raw["evaluation"]),
        training=TrainingConfig(**raw["training"]),
    )
