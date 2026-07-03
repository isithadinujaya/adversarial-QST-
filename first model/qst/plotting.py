from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qst.utils import ensure_dir


ENSEMBLE_NAMES = {0: "Haar pure", 1: "Ginibre mixed", 2: "Depolarized pure"}
ATTACK_LABELS = {
    "clean": "Clean",
    "random_replacement": "Random replacement",
    "targeted_replacement": "Targeted replacement",
    "fixed_replacement": "Fixed replacement",
    "worst_replacement": "Worst replacement",
    "frequency_pgd": "Frequency PGD",
    "combined": "Replacement + PGD",
}


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_training_history(history_paths: list[Path], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for path in history_paths:
        frame = pd.read_csv(path)
        label = path.parent.name.replace("_", " ")
        axes[0].plot(frame["epoch"], frame["train_total"], label=f"{label} train")
        axes[0].plot(frame["epoch"], frame["val_total"], linestyle="--", label=f"{label} val")
        axes[1].plot(frame["epoch"], frame["val_clean"], label=f"{label} clean")
        axes[1].plot(frame["epoch"], frame["val_adversarial"], linestyle="--", label=f"{label} adversarial")
        axes[1].plot(frame["epoch"], frame["val_consistency"], linestyle=":", label=f"{label} consistency")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Total loss")
    axes[0].set_title("Training and validation loss")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Infidelity loss")
    axes[1].set_title("Validation loss components")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)
    _save(fig, output)


def plot_metric_vs_alpha(frame: pd.DataFrame, metric: str, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.8))
    grouped = (
        frame.groupby(["num_qubits", "attack", "alpha_requested"])[metric]
        .agg(["mean", "std"])
        .reset_index()
    )
    for (qubits, attack), group in grouped.groupby(["num_qubits", "attack"]):
        group = group.sort_values("alpha_requested")
        label = f"{qubits}q — {ATTACK_LABELS.get(attack, attack)}"
        ax.plot(group["alpha_requested"], group["mean"], marker="o", label=label)
        ax.fill_between(
            group["alpha_requested"],
            group["mean"] - group["std"],
            group["mean"] + group["std"],
            alpha=0.12,
        )
    ax.set_xlabel("Requested replacement fraction α")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"{metric.replace('_', ' ').title()} versus replacement fraction")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    _save(fig, output)


def plot_fidelity_vs_epsilon(frame: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.8))
    grouped = frame.groupby(["num_qubits", "epsilon_frequency"])["fidelity"].agg(["mean", "std"]).reset_index()
    for qubits, group in grouped.groupby("num_qubits"):
        group = group.sort_values("epsilon_frequency")
        ax.plot(group["epsilon_frequency"], group["mean"], marker="o", label=f"{qubits} qubit")
        ax.fill_between(group["epsilon_frequency"], group["mean"] - group["std"], group["mean"] + group["std"], alpha=0.12)
    ax.set_xlabel("Frequency-PGD radius εf (ℓ∞)")
    ax.set_ylabel("Fidelity")
    ax.set_title("Robustness to frequency-space PGD")
    ax.grid(alpha=0.25)
    ax.legend()
    _save(fig, output)


def plot_fidelity_vs_shots(frame: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.8))
    grouped = frame.groupby(["num_qubits", "attack", "shots"])["fidelity"].agg(["mean", "std"]).reset_index()
    for (qubits, attack), group in grouped.groupby(["num_qubits", "attack"]):
        group = group.sort_values("shots")
        label = f"{qubits}q — {ATTACK_LABELS.get(attack, attack)}"
        ax.plot(group["shots"], group["mean"], marker="o", label=label)
    ax.set_xscale("log")
    ax.set_xlabel("Shots per Pauli setting")
    ax.set_ylabel("Mean fidelity")
    ax.set_title("Shot-noise sensitivity")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    _save(fig, output)


def plot_alpha_epsilon_heatmaps(frame: pd.DataFrame, output_dir: Path) -> None:
    for qubits, subset in frame.groupby("num_qubits"):
        pivot = subset.pivot_table(index="epsilon_frequency", columns="alpha_requested", values="fidelity", aggfunc="mean")
        fig, ax = plt.subplots(figsize=(6.2, 5.0))
        image = ax.imshow(pivot.values, origin="lower", aspect="auto")
        ax.set_xticks(range(len(pivot.columns)), [f"{value:.2f}" for value in pivot.columns])
        ax.set_yticks(range(len(pivot.index)), [f"{value:.3f}" for value in pivot.index])
        ax.set_xlabel("Replacement fraction α")
        ax.set_ylabel("Frequency-PGD radius εf")
        ax.set_title(f"{qubits}-qubit combined-attack fidelity")
        colorbar = fig.colorbar(image, ax=ax)
        colorbar.set_label("Mean fidelity")
        for row in range(pivot.shape[0]):
            for col in range(pivot.shape[1]):
                value = pivot.values[row, col]
                ax.text(col, row, f"{value:.3f}", ha="center", va="center", fontsize=7)
        _save(fig, output_dir / f"alpha_epsilon_heatmap_{qubits}q.png")


def plot_fidelity_boxplot(frame: pd.DataFrame, output: Path) -> None:
    categories = []
    data = []
    for (qubits, attack), subset in frame.groupby(["num_qubits", "attack"]):
        categories.append(f"{qubits}q\n{ATTACK_LABELS.get(attack, attack)}")
        data.append(subset["fidelity"].to_numpy())
    fig, ax = plt.subplots(figsize=(max(8, len(categories) * 1.0), 5.0))
    ax.boxplot(data, tick_labels=categories, showfliers=False)
    ax.set_ylabel("Fidelity")
    ax.set_title("Reconstruction-fidelity distributions")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    _save(fig, output)


def plot_infidelity_cdf(frame: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.8))
    for (qubits, attack), subset in frame.groupby(["num_qubits", "attack"]):
        values = np.sort(subset["infidelity"].to_numpy())
        cdf = np.arange(1, len(values) + 1) / len(values)
        ax.plot(values, cdf, label=f"{qubits}q — {ATTACK_LABELS.get(attack, attack)}")
    ax.set_xlabel("Infidelity")
    ax.set_ylabel("Empirical CDF")
    ax.set_title("Infidelity distribution")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    _save(fig, output)


def plot_purity_vs_fidelity(frame: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.8))
    for (qubits, attack), subset in frame.groupby(["num_qubits", "attack"]):
        sampled = subset.sample(min(len(subset), 800), random_state=7)
        ax.scatter(sampled["purity"], sampled["fidelity"], s=10, alpha=0.25, label=f"{qubits}q — {ATTACK_LABELS.get(attack, attack)}")
    ax.set_xlabel("True-state purity Tr(ρ²)")
    ax.set_ylabel("Fidelity")
    ax.set_title("Reconstruction performance versus state purity")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
    _save(fig, output)


def plot_qubit_attack_summary(frame: pd.DataFrame, output: Path) -> None:
    summary = frame.groupby(["num_qubits", "attack"])["fidelity"].mean().unstack("attack")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = np.arange(len(summary.index))
    attacks = list(summary.columns)
    width = 0.8 / max(1, len(attacks))
    for index, attack in enumerate(attacks):
        ax.bar(x + (index - (len(attacks) - 1) / 2) * width, summary[attack], width=width, label=ATTACK_LABELS.get(attack, attack))
    ax.set_xticks(x, [f"{value} qubit" for value in summary.index])
    ax.set_ylabel("Mean fidelity")
    ax.set_title("Performance across system size and attack type")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    _save(fig, output)


def plot_density_matrix_examples(npz_paths: list[Path], output_dir: Path) -> None:
    count = 0
    for path in npz_paths:
        records = np.load(path, allow_pickle=True)["records"]
        for record in records[:2]:
            record = record.item() if hasattr(record, "item") else record
            fig, axes = plt.subplots(2, 2, figsize=(8, 7))
            matrices = [
                (record["rho_true_real"], "True Re(ρ)"),
                (record["rho_pred_real"], "Predicted Re(ρ̂)"),
                (record["rho_true_imag"], "True Im(ρ)"),
                (record["rho_pred_imag"], "Predicted Im(ρ̂)"),
            ]
            maximum = max(float(np.max(np.abs(matrix))) for matrix, _ in matrices)
            for ax, (matrix, title) in zip(axes.flat, matrices):
                image = ax.imshow(matrix, vmin=-maximum, vmax=maximum)
                ax.set_title(title)
                ax.set_xlabel("Column")
                ax.set_ylabel("Row")
                fig.colorbar(image, ax=ax, fraction=0.046)
            _save(fig, output_dir / f"density_matrix_example_{count:02d}.png")
            count += 1
            if count >= 6:
                return
