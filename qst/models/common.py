from __future__ import annotations

from torch import nn


def activation_module(name: str) -> nn.Module:
    normalized = name.lower()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "gelu":
        return nn.GELU()
    if normalized == "silu":
        return nn.SiLU()
    if normalized == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unknown activation {name!r}.")


def build_feedforward(
    input_dimension: int,
    hidden_dimensions: list[int],
    activation: str,
    dropout: float,
) -> tuple[nn.Sequential, int]:
    if not hidden_dimensions:
        raise ValueError("hidden_dimensions cannot be empty.")
    layers: list[nn.Module] = []
    current = input_dimension
    for hidden in hidden_dimensions:
        layers.extend(
            [
                nn.Linear(current, hidden),
                nn.LayerNorm(hidden),
                activation_module(activation),
            ]
        )
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        current = hidden
    return nn.Sequential(*layers), current


class ResidualBlock(nn.Module):
    def __init__(self, dimension: int, activation: str, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dimension),
            nn.Linear(dimension, dimension),
            activation_module(activation),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(dimension, dimension),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )
        self.activation = activation_module(activation)

    def forward(self, inputs):
        return self.activation(inputs + self.block(inputs))
