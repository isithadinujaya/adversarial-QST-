from __future__ import annotations

import torch
from torch import nn

from qst.config import ExperimentConfig
from qst.models.base import QSTModelBase
from qst.models.common import ResidualBlock, activation_module
from qst.models.registry import register_model


@register_model("residual_mlp")
class ResidualMLPQSTModel(QSTModelBase):
    def __init__(self, experiment_config: ExperimentConfig) -> None:
        hidden = experiment_config.model.hidden_dimensions[0]
        super().__init__(experiment_config, hidden)
        self.input_projection = nn.Sequential(
            nn.Linear(experiment_config.quantum.input_dimension, hidden),
            nn.LayerNorm(hidden),
            activation_module(experiment_config.model.activation),
        )
        self.blocks = nn.Sequential(
            *[
                ResidualBlock(
                    hidden,
                    experiment_config.model.activation,
                    experiment_config.model.dropout,
                )
                for _ in range(experiment_config.model.residual_blocks)
            ]
        )
        trailing = experiment_config.model.hidden_dimensions[1:]
        layers: list[nn.Module] = []
        current = hidden
        for next_dimension in trailing:
            layers.extend(
                [
                    nn.Linear(current, next_dimension),
                    nn.LayerNorm(next_dimension),
                    activation_module(experiment_config.model.activation),
                ]
            )
            if experiment_config.model.dropout > 0:
                layers.append(nn.Dropout(experiment_config.model.dropout))
            current = next_dimension
        self.trailing = nn.Sequential(*layers) if layers else nn.Identity()
        if current != self.feature_dimension:
            # Base heads were built for the first hidden size. Rebuild them for trailing dimension.
            from qst.models.heads import CholeskyDensityHead

            self.feature_dimension = current
            self.density_head = CholeskyDensityHead(
                current, experiment_config.quantum.hilbert_dimension
            )
            self.detection_head = nn.Linear(current, 1)

    def extract_features(self, frequencies: torch.Tensor) -> torch.Tensor:
        return self.trailing(self.blocks(self.input_projection(frequencies)))
