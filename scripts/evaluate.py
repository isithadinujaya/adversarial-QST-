from __future__ import annotations

import argparse
import json
from pathlib import Path

from torch.utils.data import DataLoader

from qst.config import load_config
from qst.data import build_dataset
from qst.evaluation import evaluate_model
from qst.models import build_model
from qst.training.checkpoint import load_checkpoint
from qst.utils.seed import resolve_device, seed_everything


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate adversarial QST checkpoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", help="Optional JSON report path.")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    config = load_config(arguments.config)
    seed_everything(config.training.seed)
    device = resolve_device(config.training.device)
    model = build_model(config).to(device)
    load_checkpoint(arguments.checkpoint, model, device=device)
    dataset = build_dataset(config, "test")
    loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.data.num_workers,
        pin_memory=device.type == "cuda",
    )
    report = evaluate_model(model, loader, device, config)
    text = json.dumps(report, indent=2)
    print(text)
    output = arguments.output or str(
        Path(config.training.output_directory) / "evaluation.json"
    )
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    print(f"Saved report to {output_path.resolve()}")


if __name__ == "__main__":
    main()
