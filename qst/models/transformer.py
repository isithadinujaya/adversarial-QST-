from __future__ import annotations

import torch
from torch import nn

from qst.config import ExperimentConfig
from qst.models.base import QSTModelBase
from qst.models.registry import register_model


@register_model("setting_transformer")
class SettingTransformerQSTModel(QSTModelBase):
    """Treat each Pauli setting and its outcome distribution as one token."""

    def __init__(self, experiment_config: ExperimentConfig) -> None:
        model_config = experiment_config.model
        embedding = model_config.transformer_embedding_dimension
        super().__init__(experiment_config, embedding)
        self.number_settings = experiment_config.quantum.number_settings
        self.outcomes_per_setting = experiment_config.quantum.outcomes_per_setting
        self.token_projection = nn.Linear(self.outcomes_per_setting, embedding)
        self.setting_embedding = nn.Parameter(
            torch.zeros(1, self.number_settings, embedding)
        )
        nn.init.normal_(self.setting_embedding, mean=0.0, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=embedding,
            nhead=model_config.transformer_heads,
            dim_feedforward=model_config.transformer_feedforward_dimension,
            dropout=model_config.dropout,
            activation="gelu" if model_config.activation.lower() == "gelu" else "relu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=model_config.transformer_layers)
        self.output_norm = nn.LayerNorm(embedding)

    def extract_features(self, frequencies: torch.Tensor) -> torch.Tensor:
        tokens = frequencies.reshape(
            frequencies.shape[0], self.number_settings, self.outcomes_per_setting
        )
        tokens = self.token_projection(tokens) + self.setting_embedding
        encoded = self.encoder(tokens)
        return self.output_norm(encoded.mean(dim=1))
