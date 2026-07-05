from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from qst.plotting import (
    plot_alpha_epsilon_heatmaps,
    plot_density_matrix_examples,
    plot_fidelity_boxplot,
    plot_fidelity_vs_epsilon,
    plot_fidelity_vs_shots,
    plot_infidelity_cdf,
    plot_metric_vs_alpha,
    plot_purity_vs_fidelity,
    plot_qubit_attack_summary,
    plot_training_history,
)
from qst.utils import ensure_dir


def _read_many(paths: list[Path]) -> pd.DataFrame:
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--figures-dir", default="figures/results")
    arguments = parser.parse_args()
    outputs_root = Path(arguments.outputs_root)
    figures_dir = ensure_dir(arguments.figures_dir)

    histories = sorted(outputs_root.glob("*/history.csv"))
    if histories:
        plot_training_history(histories, figures_dir / "training_curves.png")

    alpha_paths = sorted(outputs_root.glob("*/sweeps/alpha.csv"))
    if alpha_paths:
        alpha = _read_many(alpha_paths)
        plot_metric_vs_alpha(alpha, "fidelity", figures_dir / "fidelity_vs_alpha.png")
        plot_metric_vs_alpha(alpha, "trace_distance", figures_dir / "trace_distance_vs_alpha.png")

    epsilon_paths = sorted(outputs_root.glob("*/sweeps/epsilon_frequency.csv"))
    if epsilon_paths:
        plot_fidelity_vs_epsilon(_read_many(epsilon_paths), figures_dir / "fidelity_vs_epsilon_frequency.png")

    shots_paths = sorted(outputs_root.glob("*/sweeps/shots.csv"))
    if shots_paths:
        plot_fidelity_vs_shots(_read_many(shots_paths), figures_dir / "fidelity_vs_shots.png")

    grid_paths = sorted(outputs_root.glob("*/sweeps/alpha_epsilon.csv"))
    if grid_paths:
        plot_alpha_epsilon_heatmaps(_read_many(grid_paths), figures_dir)

    evaluation_paths = sorted(outputs_root.glob("*/evaluation/all_attacks.csv"))
    if evaluation_paths:
        evaluation = _read_many(evaluation_paths)
        plot_fidelity_boxplot(evaluation, figures_dir / "fidelity_boxplots.png")
        plot_infidelity_cdf(evaluation, figures_dir / "infidelity_cdf.png")
        plot_purity_vs_fidelity(evaluation, figures_dir / "purity_vs_fidelity.png")
        plot_qubit_attack_summary(evaluation, figures_dir / "qubit_attack_summary.png")

    prediction_paths = sorted(outputs_root.glob("*/evaluation/*.predictions.npz"))
    if prediction_paths:
        plot_density_matrix_examples(prediction_paths, figures_dir / "density_examples")

    print(figures_dir)


if __name__ == "__main__":
    main()
