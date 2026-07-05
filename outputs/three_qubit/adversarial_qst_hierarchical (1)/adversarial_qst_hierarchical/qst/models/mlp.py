from __future__ import annotations

import torch
from torch import nn

from .head import raw_cholesky_to_density


def _activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"Unsupported activation: {name}")


class MLPDensityReconstructor(nn.Module):
    """Architecture-specific encoder with an architecture-independent physical output."""

    def __init__(
        self,
        input_dimension: int,
        density_dimension: int,
        hidden_dims: list[int],
        activation: str = "gelu",
        dropout: float = 0.0,
        layer_norm: bool = True,
        diagonal_floor: float = 1.0e-4,
    ) -> None:
        super().__init__()
        dimensions = [input_dimension, *hidden_dims]
        layers: list[nn.Module] = []
        for input_dim, output_dim in zip(dimensions[:-1], dimensions[1:]):
            layers.append(nn.Linear(input_dim, output_dim))
            if layer_norm:
                layers.append(nn.LayerNorm(output_dim))
            layers.append(_activation(activation))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
        self.backbone = nn.Sequential(*layers)
        final_dim = dimensions[-1]
        self.reconstruction_head = nn.Linear(final_dim, density_dimension**2)
        self.density_dimension = density_dimension
        self.diagonal_floor = diagonal_floor

    def forward(self, frequencies: torch.Tensor) -> torch.Tensor:
        latent = self.backbone(frequencies)
        raw = self.reconstruction_head(latent)
        return raw_cholesky_to_density(
            raw,
            self.density_dimension,
            diagonal_floor=self.diagonal_floor,
        )
