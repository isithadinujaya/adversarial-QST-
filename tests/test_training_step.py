from pathlib import Path

import torch
from torch.utils.data import DataLoader

from qst.config import load_config
from qst.data.dataset import TomographyDataset
from qst.models import build_model
from qst.training.trainer import Trainer


def test_one_training_epoch_runs(tmp_path: Path):
    config = load_config("configs/one_qubit.yaml")
    config.training.output_directory = str(tmp_path)
    config.training.batch_size = 4
    config.attacks.pgd.steps = 1
    dataset = TomographyDataset(config, 8, "train")
    loader = DataLoader(dataset, batch_size=4)
    model = build_model(config)
    trainer = Trainer(model, config, torch.device("cpu"))
    metrics = trainer.run_epoch(loader, training=True)
    assert torch.isfinite(torch.tensor(metrics["total"]))
