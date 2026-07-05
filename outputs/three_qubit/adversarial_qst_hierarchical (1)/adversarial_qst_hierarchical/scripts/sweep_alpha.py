from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from qst.evaluator import AttackParameters, QSTEvaluator
from qst.utils import ensure_dir
from scripts.common import prepare


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--attacks",
        nargs="+",
        default=["random_replacement", "targeted_replacement", "worst_replacement"],
    )
    arguments = parser.parse_args()
    config, device, _, loaders, measurement, model = prepare(
        arguments.config, arguments.checkpoint
    )
    evaluator = QSTEvaluator(config, model, measurement, device)
    frames = []
    for attack in arguments.attacks:
        for alpha in config.evaluation.alpha_grid:
            frame, _ = evaluator.evaluate(
                loaders["test"], AttackParameters(attack=attack, alpha=alpha)
            )
            frames.append(frame)
    output = ensure_dir(Path(config.experiment.output_dir) / "sweeps") / "alpha.csv"
    pd.concat(frames, ignore_index=True).to_csv(output, index=False)
    print(output)


if __name__ == "__main__":
    main()
