from __future__ import annotations

import torch

from qst.config import ExperimentConfig
from qst.models.base import QSTModelBase
from qst.models.common import build_feedforward
from qst.models.registry import register_model


@register_model("mlp")
class MLPQSTModel(QSTModelBase):
    def __init__(self, experiment_config: ExperimentConfig) -> None:
        network, feature_dimension = build_feedforward(
            experiment_config.quantum.input_dimension,
            experiment_config.model.hidden_dimensions,
            experiment_config.model.activation,
            experiment_config.model.dropout,
        )
        super().__init__(experiment_config, feature_dimension)
        self.network = network

    def extract_features(self, frequencies: torch.Tensor) -> torch.Tensor:
        return self.network(frequencies)
