from __future__ import annotations

from pathlib import Path

import torch

from qst.config import load_config
from qst.data.dataset import TomographyDataset
from qst.models import available_models, build_model
from qst.quantum.metrics import density_diagnostics


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    for number_qubits, filename in [
        (1, "one_qubit.yaml"),
        (2, "two_qubit.yaml"),
        (3, "three_qubit.yaml"),
    ]:
        config = load_config(root / "configs" / filename)
        dataset = TomographyDataset(config, 2, "train")
        batch = torch.stack([dataset[index]["input_frequencies"] for index in range(2)])
        for model_name in available_models():
            config.model.name = model_name
            model = build_model(config)
            density, attack_logit = model(batch)
            diagnostics = density_diagnostics(density)
            assert density.shape == (
                2,
                2 ** number_qubits,
                2 ** number_qubits,
            )
            assert attack_logit.shape == (2,)
            assert float(diagnostics["trace_error"].max().detach()) < 1e-5
            assert float(diagnostics["hermiticity_error"].max().detach()) < 1e-5
            assert float(diagnostics["minimum_eigenvalue"].min().detach()) > -1e-5
            print(
                f"PASS: {number_qubits} qubit(s), {model_name}, "
                f"input={config.quantum.input_dimension}, density={tuple(density.shape)}"
            )
    print("All smoke tests passed.")


if __name__ == "__main__":
    main()
