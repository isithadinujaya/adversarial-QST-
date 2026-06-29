from __future__ import annotations

import torch
from torch.utils.data import Dataset

from .config import DataConfig
from .quantum import generate_state_ensemble


class QuantumStateDataset(Dataset[torch.Tensor]):
    def __init__(self, states: torch.Tensor) -> None:
        if states.ndim != 3 or states.shape[-2:] != (2, 2):
            raise ValueError("states must have shape [num_states,2,2].")
        self.states = states.cpu()

    def __len__(self) -> int:
        return self.states.shape[0]

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.states[index]


def generate_split(num_states: int, config: DataConfig, seed: int) -> QuantumStateDataset:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        states = generate_state_ensemble(
            num_states,
            pure_fraction=config.pure_fraction,
            ginibre_fraction=config.ginibre_fraction,
            depolarized_fraction=config.depolarized_fraction,
            device="cpu",
        )
    return QuantumStateDataset(states)


def build_datasets(config: DataConfig, seed: int) -> tuple[QuantumStateDataset, ...]:
    """Split states before measurements/attacks, preventing state leakage."""
    train = generate_split(config.train_states, config, seed)
    validation = generate_split(config.validation_states, config, seed + 1)
    test = generate_split(config.test_states, config, seed + 2)
    return train, validation, test
