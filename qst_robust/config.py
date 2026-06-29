from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict


@dataclass
class DataConfig:
    train_states: int = 50_000
    validation_states: int = 5_000
    test_states: int = 10_000
    shots_per_basis: int = 1_000
    pure_fraction: float = 0.25
    ginibre_fraction: float = 0.50
    depolarized_fraction: float = 0.25

    def validate(self) -> None:
        total = self.pure_fraction + self.ginibre_fraction + self.depolarized_fraction
        if abs(total - 1.0) > 1e-8:
            raise ValueError(f"State fractions must sum to 1, but sum to {total}.")
        if self.shots_per_basis <= 0:
            raise ValueError("shots_per_basis must be positive.")


@dataclass
class AttackConfig:
    alpha_min: float = 0.01
    alpha_max: float = 0.20
    target_min_trace_distance: float = 0.50

    pgd_epsilon_min: float = 0.005
    pgd_epsilon_max: float = 0.050
    pgd_train_steps: int = 10
    pgd_eval_steps: int = 40
    pgd_eval_restarts: int = 5
    adaptive_pgd_gamma: float = 0.50

    def validate(self) -> None:
        if not (0.0 <= self.alpha_min <= self.alpha_max <= 1.0):
            raise ValueError("Require 0 <= alpha_min <= alpha_max <= 1.")
        if not (0.0 <= self.target_min_trace_distance <= 1.0):
            raise ValueError("For qubits, target trace distance must be in [0, 1].")
        if not (0.0 <= self.pgd_epsilon_min <= self.pgd_epsilon_max <= 1.0):
            raise ValueError("Require 0 <= epsilon_min <= epsilon_max <= 1.")
        if self.pgd_train_steps <= 0 or self.pgd_eval_steps <= 0:
            raise ValueError("PGD step counts must be positive.")
        if self.pgd_eval_restarts <= 0:
            raise ValueError("PGD restarts must be positive.")


@dataclass
class ModelConfig:
    input_dimension: int = 6
    hidden_1: int = 128
    hidden_2: int = 256
    hidden_3: int = 128
    head_hidden: int = 64
    density_jitter: float = 1e-8


@dataclass
class TrainingConfig:
    epochs: int = 50
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    clean_loss_weight: float = 1.0
    adversarial_loss_weight: float = 1.0
    detection_loss_weight: float = 0.1
    fidelity_loss_weight: float = 0.1
    gradient_clip_norm: float = 5.0
    seed: int = 2026
    num_workers: int = 0
    checkpoint_path: str = "checkpoints/best_model.pt"


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def validate(self) -> None:
        self.data.validate()
        self.attack.validate()
        if self.model.input_dimension != 6:
            raise ValueError("Pauli X/Y/Z frequency input must have dimension 6.")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ExperimentConfig":
        return cls(
            data=DataConfig(**payload["data"]),
            attack=AttackConfig(**payload["attack"]),
            model=ModelConfig(**payload["model"]),
            training=TrainingConfig(**payload["training"]),
        )
