from pathlib import Path

import torch

from qst.config import load_config
from qst.models import available_models, build_model
from qst.quantum.metrics import density_diagnostics


def test_all_models_return_physical_density_matrices():
    root = Path(__file__).resolve().parents[1]
    for filename in ["one_qubit.yaml", "two_qubit.yaml", "three_qubit.yaml"]:
        config = load_config(root / "configs" / filename)
        inputs = torch.rand(3, config.quantum.input_dimension)
        shaped = inputs.reshape(3, config.quantum.number_settings, config.quantum.outcomes_per_setting)
        inputs = (shaped / shaped.sum(-1, keepdim=True)).reshape(3, -1)
        for model_name in available_models():
            config.model.name = model_name
            model = build_model(config)
            density, logits = model(inputs)
            diagnostics = density_diagnostics(density)
            assert logits.shape == (3,)
            assert density.shape[-2:] == (
                config.quantum.hilbert_dimension,
                config.quantum.hilbert_dimension,
            )
            assert torch.max(diagnostics["trace_error"]) < 1e-5
            assert torch.max(diagnostics["hermiticity_error"]) < 1e-5
            assert torch.min(diagnostics["minimum_eigenvalue"]) > -1e-5
