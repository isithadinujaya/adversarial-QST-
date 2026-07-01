from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class QuantumConfig:
    number_qubits: int = 1
    copies_per_setting: int = 1000
    pure_state_probability: float = 0.5
    complex_dtype: str = "complex64"

    @property
    def hilbert_dimension(self) -> int:
        return 2 ** self.number_qubits

    @property
    def number_settings(self) -> int:
        return 3 ** self.number_qubits

    @property
    def outcomes_per_setting(self) -> int:
        return 2 ** self.number_qubits

    @property
    def input_dimension(self) -> int:
        return self.number_settings * self.outcomes_per_setting

    @property
    def cholesky_output_dimension(self) -> int:
        return self.hilbert_dimension ** 2


@dataclass
class PGDConfig:
    enabled: bool = True
    fraction: float = 0.5
    epsilon: float = 0.04
    step_size: float = 0.01
    steps: int = 5
    random_start: bool = True
    detection_evasion_weight: float = 0.1


@dataclass
class AttackConfig:
    probabilities: dict[str, float] = field(
        default_factory=lambda: {
            "clean": 0.25,
            "random_replacement": 0.25,
            "targeted_replacement": 0.20,
            "worst_case_replacement": 0.20,
            "random_frequency": 0.10,
        }
    )
    replacement_fraction_min: float = 0.05
    replacement_fraction_max: float = 0.30
    replacement_pure_probability: float = 0.5
    targeted_state: str = "zero_state"
    worst_case_candidates: int = 16
    random_frequency_epsilon: float = 0.04
    pgd: PGDConfig = field(default_factory=PGDConfig)


@dataclass
class ModelConfig:
    name: str = "residual_mlp"
    hidden_dimensions: list[int] = field(default_factory=lambda: [128, 128])
    activation: str = "gelu"
    dropout: float = 0.05
    residual_blocks: int = 3
    transformer_embedding_dimension: int = 96
    transformer_heads: int = 4
    transformer_layers: int = 3
    transformer_feedforward_dimension: int = 192
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class LossConfig:
    reconstruction: str = "frobenius"
    clean_reconstruction_weight: float = 1.0
    adversarial_reconstruction_weight: float = 1.0
    pgd_reconstruction_weight: float = 1.0
    detection_weight: float = 0.25
    measurement_consistency_weight: float = 0.10


@dataclass
class DataConfig:
    mode: str = "online"
    train_samples: int = 20_000
    validation_samples: int = 3_000
    test_samples: int = 3_000
    train_file: str | None = None
    validation_file: str | None = None
    test_file: str | None = None
    num_workers: int = 0


@dataclass
class TrainingConfig:
    seed: int = 1234
    device: str = "auto"
    epochs: int = 50
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 5.0
    early_stopping_patience: int = 10
    output_directory: str = "outputs/default"
    data: DataConfig = field(default_factory=DataConfig)


@dataclass
class ExperimentConfig:
    experiment_name: str = "experiment"
    quantum: QuantumConfig = field(default_factory=QuantumConfig)
    attacks: AttackConfig = field(default_factory=AttackConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def validate(self) -> None:
        if self.quantum.number_qubits not in {1, 2, 3}:
            raise ValueError("number_qubits must be 1, 2, or 3 for the supplied configurations.")
        if self.quantum.copies_per_setting <= 0:
            raise ValueError("copies_per_setting must be positive.")
        if self.quantum.complex_dtype not in {"complex64", "complex128"}:
            raise ValueError("complex_dtype must be complex64 or complex128.")
        for name, value in self.attacks.probabilities.items():
            if value < 0:
                raise ValueError(f"Attack probability {name!r} cannot be negative.")
        total = sum(self.attacks.probabilities.values())
        if abs(total - 1.0) > 1e-8:
            raise ValueError(f"Attack probabilities must sum to 1.0; received {total}.")
        required = {
            "clean",
            "random_replacement",
            "targeted_replacement",
            "worst_case_replacement",
            "random_frequency",
        }
        missing = required.difference(self.attacks.probabilities)
        if missing:
            raise ValueError(f"Missing attack probability entries: {sorted(missing)}")
        if not 0 <= self.quantum.pure_state_probability <= 1:
            raise ValueError("pure_state_probability must lie in [0, 1].")
        if not 0 <= self.attacks.replacement_pure_probability <= 1:
            raise ValueError("replacement_pure_probability must lie in [0, 1].")
        if not 0 <= self.attacks.replacement_fraction_min <= self.attacks.replacement_fraction_max <= 1:
            raise ValueError("Replacement fractions must satisfy 0 <= min <= max <= 1.")
        if not self.model.hidden_dimensions or any(value <= 0 for value in self.model.hidden_dimensions):
            raise ValueError("model.hidden_dimensions must contain positive dimensions.")
        if not 0 <= self.model.dropout < 1:
            raise ValueError("model.dropout must lie in [0, 1).")
        if self.model.transformer_embedding_dimension % self.model.transformer_heads != 0:
            raise ValueError("transformer_embedding_dimension must be divisible by transformer_heads.")
        if self.training.data.mode not in {"online", "cached"}:
            raise ValueError("training.data.mode must be 'online' or 'cached'.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _construct_config(raw: dict[str, Any]) -> ExperimentConfig:
    quantum = QuantumConfig(**raw.get("quantum", {}))

    attacks_raw = dict(raw.get("attacks", {}))
    pgd = PGDConfig(**attacks_raw.pop("pgd", {}))
    attacks = AttackConfig(pgd=pgd, **attacks_raw)

    model = ModelConfig(**raw.get("model", {}))
    loss = LossConfig(**raw.get("loss", {}))

    training_raw = dict(raw.get("training", {}))
    data = DataConfig(**training_raw.pop("data", {}))
    training = TrainingConfig(data=data, **training_raw)

    config = ExperimentConfig(
        experiment_name=raw.get("experiment_name", "experiment"),
        quantum=quantum,
        attacks=attacks,
        model=model,
        loss=loss,
        training=training,
    )
    config.validate()
    return config


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Configuration file {path} must contain a YAML mapping.")
    return _construct_config(raw)


def config_from_dict(raw: dict[str, Any]) -> ExperimentConfig:
    return _construct_config(raw)
