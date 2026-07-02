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
        "--physical-kind",
        default="random_replacement",
        choices=["random_replacement", "targeted_replacement", "worst_replacement"],
    )
    arguments = parser.parse_args()
    config, device, _, loaders, measurement, model = prepare(
        arguments.config, arguments.checkpoint
    )
    evaluator = QSTEvaluator(config, model, measurement, device)
    frames = []
    for alpha in config.evaluation.alpha_epsilon_alpha_grid:
        for epsilon in config.evaluation.alpha_epsilon_frequency_grid:
            frame, _ = evaluator.evaluate(
                loaders["test"],
                AttackParameters(
                    attack="combined",
                    alpha=alpha,
                    epsilon_frequency=epsilon,
                    combined_physical_kind=arguments.physical_kind,
                ),
            )
            frames.append(frame)
    output = ensure_dir(Path(config.experiment.output_dir) / "sweeps") / "alpha_epsilon.csv"
    pd.concat(frames, ignore_index=True).to_csv(output, index=False)
    print(output)


if __name__ == "__main__":
    main()
