from __future__ import annotations

from pathlib import Path

from qst.config import QSTConfig, load_config
from qst.data import build_datasets, build_loaders
from qst.evaluator import QSTEvaluator, load_checkpoint
from qst.measurements import PauliCubeMeasurement
from qst.models import build_model
from qst.utils import count_parameters, resolve_device, seed_everything


def prepare(config_path: str, checkpoint_path: str | None = None):
    config = load_config(config_path)
    seed_everything(config.experiment.seed)
    device = resolve_device(config.experiment.device)
    datasets = build_datasets(config)
    loaders = build_loaders(config, datasets)
    measurement = PauliCubeMeasurement(config.experiment.num_qubits, device=device)
    model = build_model(config).to(device)
    if checkpoint_path is not None:
        load_checkpoint(checkpoint_path, model, device)
    print(
        f"Experiment={config.experiment.name} | device={device} | "
        f"input={config.input_dimension} | d={config.dimension} | "
        f"parameters={count_parameters(model):,}"
    )
    return config, device, datasets, loaders, measurement, model
