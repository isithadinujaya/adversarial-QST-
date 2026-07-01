from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import trange

from qst.config import load_config
from qst.data.dataset import TomographyDataset


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a fixed QST dataset file.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=["train", "validation", "test"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, help="Optional sample-count override.")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    config = load_config(arguments.config)
    default_lengths = {
        "train": config.training.data.train_samples,
        "validation": config.training.data.validation_samples,
        "test": config.training.data.test_samples,
    }
    length = arguments.samples or default_lengths[arguments.split]
    dataset = TomographyDataset(config, length, arguments.split)
    accumulated: dict[str, list[torch.Tensor]] = {}
    for index in trange(length, desc=f"generate {arguments.split}"):
        sample = dataset[index]
        for name, tensor in sample.items():
            accumulated.setdefault(name, []).append(tensor)
    samples = {name: torch.stack(values) for name, values in accumulated.items()}
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "samples": samples,
            "metadata": {
                "split": arguments.split,
                "length": length,
                "config": config.to_dict(),
            },
        },
        output,
    )
    print(f"Saved {length} samples to {output.resolve()}")


if __name__ == "__main__":
    main()
