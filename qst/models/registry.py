from __future__ import annotations

from collections.abc import Callable

from torch import nn

from qst.config import QSTConfig
from .mlp import MLPDensityReconstructor


ModelBuilder = Callable[[QSTConfig], nn.Module]
_MODEL_REGISTRY: dict[str, ModelBuilder] = {}


def register_model(name: str, builder: ModelBuilder) -> None:
    key = name.lower()
    if key in _MODEL_REGISTRY:
        raise KeyError(f"Model '{name}' is already registered.")
    _MODEL_REGISTRY[key] = builder


def _build_mlp(config: QSTConfig) -> nn.Module:
    return MLPDensityReconstructor(
        input_dimension=config.input_dimension,
        density_dimension=config.dimension,
        hidden_dims=config.model.hidden_dims,
        activation=config.model.activation,
        dropout=config.model.dropout,
        layer_norm=config.model.layer_norm,
        diagonal_floor=config.model.diagonal_floor,
    )


register_model("mlp", _build_mlp)


def build_model(config: QSTConfig) -> nn.Module:
    key = config.model.name.lower()
    if key not in _MODEL_REGISTRY:
        available = ", ".join(sorted(_MODEL_REGISTRY))
        raise KeyError(f"Unknown model '{config.model.name}'. Available: {available}")
    return _MODEL_REGISTRY[key](config)
