from __future__ import annotations

import argparse

import torch

from qst.attacks.state import ID_TO_ATTACK
from qst.config import load_config
from qst.data.dataset import TomographyDataset


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect one deterministic generated sample.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--split", choices=["train", "validation", "test"], default="train")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    config = load_config(arguments.config)
    dataset = TomographyDataset(config, max(arguments.index + 1, 1), arguments.split)
    sample = dataset[arguments.index]
    attack_id = int(sample["attack_id"])
    print(f"Attack: {ID_TO_ATTACK[attack_id]}")
    print(f"Attack label: {float(sample['attack_label'])}")
    print(f"Attack fraction/epsilon: {float(sample['attack_fraction']):.6f}")
    print(f"State kind: {'pure' if int(sample['state_kind']) == 0 else 'mixed'}")
    print(f"Clean frequency shape: {tuple(sample['clean_frequencies'].shape)}")
    print(f"Input frequency shape: {tuple(sample['input_frequencies'].shape)}")
    print(f"Density shape: {tuple(sample['target_density'].shape)}")
    settings = sample["input_frequencies"].reshape(
        config.quantum.number_settings,
        config.quantum.outcomes_per_setting,
    )
    print(f"Maximum setting normalization error: {float(torch.max(torch.abs(settings.sum(-1)-1))):.3e}")
    print("Target density:")
    print(sample["target_density"])
    print("First frequency entries:")
    print(sample["input_frequencies"][: min(16, config.quantum.input_dimension)])


if __name__ == "__main__":
    main()
