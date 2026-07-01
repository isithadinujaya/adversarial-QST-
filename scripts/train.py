from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from qst.config import load_config
from qst.data import build_dataset
from qst.models import available_models, build_model
from qst.training.trainer import Trainer
from qst.utils.seed import resolve_device, seed_everything


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train adversarial QST model.")
    parser.add_argument("--config", required=True, help="Path to YAML configuration.")
    parser.add_argument("--model", choices=available_models(), help="Optional model override.")
    parser.add_argument("--epochs", type=int, help="Optional epoch override.")
    parser.add_argument("--output", help="Optional output-directory override.")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    config = load_config(arguments.config)
    if arguments.model is not None:
        config.model.name = arguments.model
    if arguments.epochs is not None:
        config.training.epochs = arguments.epochs
    if arguments.output is not None:
        config.training.output_directory = arguments.output
    config.validate()

    seed_everything(config.training.seed)
    device = resolve_device(config.training.device)
    train_dataset = build_dataset(config, "train")
    validation_dataset = build_dataset(config, "validation")
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.data.num_workers,
        pin_memory=pin_memory,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.data.num_workers,
        pin_memory=pin_memory,
    )

    model = build_model(config)
    if config.quantum.complex_dtype == "complex128":
        model = model.double()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Experiment: {config.experiment_name}")
    print(f"Device: {device}")
    print(f"Model: {config.model.name} ({parameter_count:,} parameters)")
    print(
        "Qubits: "
        f"{config.quantum.number_qubits} | dimension={config.quantum.hilbert_dimension} | "
        f"settings={config.quantum.number_settings} | "
        f"outcomes/setting={config.quantum.outcomes_per_setting} | "
        f"input={config.quantum.input_dimension} | "
        f"density outputs={config.quantum.cholesky_output_dimension}"
    )

    trainer = Trainer(model, config, device)
    best_path = trainer.fit(train_loader, validation_loader)
    print(f"Best checkpoint: {Path(best_path).resolve()}")


if __name__ == "__main__":
    main()
