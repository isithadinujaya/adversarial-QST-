from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from qst.evaluator import AttackParameters, QSTEvaluator
from qst.utils import ensure_dir, save_json
from scripts.common import prepare


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    arguments = parser.parse_args()
    config, device, _, loaders, measurement, model = prepare(
        arguments.config, arguments.checkpoint
    )
    evaluator = QSTEvaluator(config, model, measurement, device)
    output_dir = ensure_dir(Path(config.experiment.output_dir) / "evaluation")
    summaries = {}
    frames = []
    for attack in config.evaluation.attacks:
        parameters = AttackParameters(
            attack=attack,
            alpha=config.evaluation.default_alpha,
            epsilon_frequency=config.evaluation.default_epsilon_frequency,
        )
        frame, summary = evaluator.evaluate(
            loaders["test"],
            parameters,
            save_path=output_dir / f"{attack}.csv",
        )
        frames.append(frame)
        summaries[attack] = summary
        print(attack, summary)
    pd.concat(frames, ignore_index=True).to_csv(output_dir / "all_attacks.csv", index=False)
    save_json(output_dir / "summary.json", summaries)


if __name__ == "__main__":
    main()
