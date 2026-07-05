from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ExperimentConfig:
    name: str = "one_qubit"
    seed: int = 7
    num_qubits: int = 1
    device: str = "auto"
    output_dir: str = "outputs/one_qubit"


@dataclass
class DataConfig:
    train_states: int = 20000
    val_states: int = 3000
    test_states: int = 3000
    shots_per_setting: int = 1000
    train_resample_measurements: bool = True
    pure_fraction: float = 0.35
    mixed_fraction: float = 0.35
    depolarized_fraction: float = 0.30
    depolarized_visibility_min: float = 0.20
    depolarized_visibility_max: float = 1.00
    ginibre_rank: int | None = None
    num_workers: int = 0


@dataclass
class ModelConfig:
    name: str = "mlp"
    hidden_dims: list[int] = field(default_factory=lambda: [256, 256, 128])
    activation: str = "gelu"
    dropout: float = 0.05
    layer_norm: bool = True
    diagonal_floor: float = 1.0e-4


@dataclass
class LossConfig:
    # L_total = clean_weight * L_clean
    #         + physical_weight * L_physical
    #         + pgd_weight * L_pgd
    #         + consistency_weight * C_combined
    clean_weight: float = 1.0
    physical_weight: float = 0.5
    pgd_weight: float = 0.5
    consistency_weight: float = 0.1
    physical_max_weight: float = 0.7
    fidelity_epsilon: float = 1.0e-9  # evaluation only


@dataclass
class AttackConfig:
    physical_training_types: list[str] = field(
        default_factory=lambda: [
            "random_replacement",
            "targeted_replacement",
            "worst_replacement",
        ]
    )
    alpha_min: float = 0.01
    alpha_max: float = 0.20
    epsilon_physical: float = 0.20
    target_state: str = "random_far"
    target_min_trace_distance: float = 0.50
    epsilon_frequency_min: float = 0.005
    epsilon_frequency_max: float = 0.050
    pgd_train_steps: int = 10
    pgd_eval_steps: int = 40
    pgd_step_size: float = 0.01
    pgd_random_start: bool = True


@dataclass
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 256
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    gradient_clip_norm: float = 1.0
    scheduler_factor: float = 0.5
    scheduler_patience: int = 8
    early_stopping_patience: int = 20
    min_learning_rate: float = 1.0e-6
    log_every: int = 1


@dataclass
class EvaluationConfig:
    batch_size: int = 256
    max_samples: int = 3000
    attacks: list[str] = field(
        default_factory=lambda: [
            "clean",
            "random_replacement",
            "targeted_replacement",
            "worst_replacement",
            "frequency_pgd",
        ]
    )
    default_alpha: float = 0.10
    default_epsilon_frequency: float = 0.03
    alpha_grid: list[float] = field(
        default_factory=lambda: [0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30]
    )
    epsilon_frequency_grid: list[float] = field(
        default_factory=lambda: [0.0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10]
    )
    shots_grid: list[int] = field(default_factory=lambda: [100, 250, 500, 1000, 2000, 5000])
    alpha_epsilon_alpha_grid: list[float] = field(
        default_factory=lambda: [0.0, 0.05, 0.10, 0.15, 0.20]
    )
    alpha_epsilon_frequency_grid: list[float] = field(
        default_factory=lambda: [0.0, 0.01, 0.02, 0.03, 0.05]
    )
    save_predictions: bool = True


@dataclass
class QSTConfig:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    @property
    def dimension(self) -> int:
        return 2 ** self.experiment.num_qubits

    @property
    def num_settings(self) -> int:
        return 3 ** self.experiment.num_qubits

    @property
    def input_dimension(self) -> int:
        return self.num_settings * self.dimension

    def validate(self) -> None:
        if self.experiment.num_qubits not in (1, 2, 3):
            raise ValueError("This project supports one, two, or three qubits.")
        fractions = (
            self.data.pure_fraction
            + self.data.mixed_fraction
            + self.data.depolarized_fraction
        )
        if abs(fractions - 1.0) > 1.0e-6:
            raise ValueError("State-ensemble fractions must sum to one.")

        allowed_physical = {
            "random_replacement",
            "targeted_replacement",
            "fixed_replacement",
            "worst_replacement",
        }
        if not self.attack.physical_training_types:
            raise ValueError("At least one physical training attack is required.")
        unknown = set(self.attack.physical_training_types) - allowed_physical
        if unknown:
            raise ValueError(f"Unknown physical training attacks: {sorted(unknown)}")

        if not (0.0 <= self.attack.alpha_min <= self.attack.alpha_max <= 1.0):
            raise ValueError("Require 0 <= alpha_min <= alpha_max <= 1.")
        if self.attack.epsilon_physical < 0.0:
            raise ValueError("epsilon_physical must be nonnegative.")
        if self.attack.epsilon_frequency_min < 0.0:
            raise ValueError("Frequency epsilon must be nonnegative.")
        if self.attack.epsilon_frequency_min > self.attack.epsilon_frequency_max:
            raise ValueError("Frequency epsilon minimum exceeds maximum.")

        if self.loss.clean_weight < 0.0:
            raise ValueError("clean_weight must be nonnegative.")
        if self.loss.physical_weight < 0.0 or self.loss.pgd_weight < 0.0:
            raise ValueError("Physical and PGD weights must be nonnegative.")
        if abs(self.loss.physical_weight + self.loss.pgd_weight - 1.0) > 1.0e-6:
            raise ValueError("physical_weight and pgd_weight must sum to one.")
        if not (0.0 <= self.loss.physical_max_weight <= 1.0):
            raise ValueError("physical_max_weight must lie in [0,1].")
        if self.loss.consistency_weight < 0.0:
            raise ValueError("consistency_weight must be nonnegative.")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _update_dataclass(instance: Any, values: dict[str, Any]) -> Any:
    for key, value in values.items():
        if not hasattr(instance, key):
            raise KeyError(f"Unknown configuration field: {type(instance).__name__}.{key}")
        setattr(instance, key, value)
    return instance


def load_config(path: str | Path) -> QSTConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    config = QSTConfig()
    for section_name, section_values in raw.items():
        if not hasattr(config, section_name):
            raise KeyError(f"Unknown configuration section: {section_name}")
        section = getattr(config, section_name)
        _update_dataclass(section, section_values or {})

    config.validate()
    return config
