from __future__ import annotations

from torch.utils.data import DataLoader

from qst.config import QSTConfig
from qst.states import DensityMatrixDataset, StateMixture


def _mixture(config: QSTConfig) -> StateMixture:
    return StateMixture(
        pure_fraction=config.data.pure_fraction,
        mixed_fraction=config.data.mixed_fraction,
        depolarized_fraction=config.data.depolarized_fraction,
        visibility_min=config.data.depolarized_visibility_min,
        visibility_max=config.data.depolarized_visibility_max,
        ginibre_rank=config.data.ginibre_rank,
    )


def build_datasets(config: QSTConfig) -> dict[str, DensityMatrixDataset]:
    mixture = _mixture(config)
    seed = config.experiment.seed
    return {
        "train": DensityMatrixDataset(
            config.data.train_states, config.dimension, mixture, seed=seed + 101
        ),
        "val": DensityMatrixDataset(
            config.data.val_states, config.dimension, mixture, seed=seed + 202
        ),
        "test": DensityMatrixDataset(
            config.data.test_states, config.dimension, mixture, seed=seed + 303
        ),
    }


def build_loaders(
    config: QSTConfig,
    datasets: dict[str, DensityMatrixDataset],
) -> dict[str, DataLoader]:
    workers = config.data.num_workers
    common = {
        "num_workers": workers,
        "pin_memory": False,
        "persistent_workers": workers > 0,
    }
    return {
        "train": DataLoader(
            datasets["train"],
            batch_size=config.training.batch_size,
            shuffle=True,
            drop_last=False,
            **common,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=config.evaluation.batch_size,
            shuffle=False,
            drop_last=False,
            **common,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=config.evaluation.batch_size,
            shuffle=False,
            drop_last=False,
            **common,
        ),
    }
