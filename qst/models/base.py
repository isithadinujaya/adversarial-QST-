from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn

from qst.config import ExperimentConfig
from qst.models.heads import CholeskyDensityHead


class QSTModelBase(nn.Module, ABC):
    def __init__(self, experiment_config: ExperimentConfig, feature_dimension: int) -> None:
        super().__init__()
        self.experiment_config = experiment_config
        self.input_dimension = experiment_config.quantum.input_dimension
        self.hilbert_dimension = experiment_config.quantum.hilbert_dimension
        self.feature_dimension = feature_dimension
        self.density_head = CholeskyDensityHead(feature_dimension, self.hilbert_dimension)
        self.detection_head = nn.Linear(feature_dimension, 1)

    def _validate_input(self, frequencies: torch.Tensor) -> None:
        if frequencies.shape[-1] != self.input_dimension:
            raise ValueError(
                f"Expected input dimension {self.input_dimension}, received {frequencies.shape[-1]}."
            )

    @abstractmethod
    def extract_features(self, frequencies: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, frequencies: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_input(frequencies)
        features = self.extract_features(frequencies)
        density = self.density_head(features)
        attack_logit = self.detection_head(features).squeeze(-1)
        return density, attack_logit
