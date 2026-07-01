from __future__ import annotations

from collections.abc import Callable

import torch.nn as nn

from qst.config import ExperimentConfig


_MODEL_REGISTRY: dict[str, type[nn.Module]] = {}


def register_model(name: str) -> Callable[[type[nn.Module]], type[nn.Module]]:
    normalized = name.lower()

    def decorator(model_class: type[nn.Module]) -> type[nn.Module]:
        if normalized in _MODEL_REGISTRY:
            raise KeyError(f"A model named {normalized!r} is already registered.")
        _MODEL_REGISTRY[normalized] = model_class
        return model_class

    return decorator


def available_models() -> tuple[str, ...]:
    return tuple(sorted(_MODEL_REGISTRY))


def build_model(config: ExperimentConfig) -> nn.Module:
    # Ensure registration imports occurred even when this module is imported directly.
    import qst.models.mlp  # noqa: F401
    import qst.models.residual_mlp  # noqa: F401
    import qst.models.transformer  # noqa: F401

    name = config.model.name.lower()
    if name not in _MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model {name!r}. Available models: {', '.join(available_models())}."
        )
    return _MODEL_REGISTRY[name](config)
