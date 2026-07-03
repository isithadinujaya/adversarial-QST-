from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from qst.utils import ensure_dir


def box(ax, x, y, width, height, text, fontsize=10):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02",
        linewidth=1.5,
        facecolor="white",
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=fontsize)


def arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=14, linewidth=1.5))


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(Path(path).with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def pipeline(output_dir: Path):
    fig, ax = plt.subplots(figsize=(13, 3.2))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 3)
    ax.axis("off")
    labels = [
        (0.2, "ρ\nHaar / Ginibre /\ndepolarized"),
        (2.3, "Pauli cube\n{X,Y,Z}ⁿ"),
        (4.4, "Multinomial\nfrequencies f"),
        (6.5, "Physical replacement\nor frequency PGD"),
        (8.8, "Interchangeable\nbackbone"),
        (11.0, "Cholesky head\nρ̂=TT†/Tr(TT†)"),
    ]
    for x, label in labels:
        box(ax, x, 1.0, 1.7, 1.0, label)
    for left, right in zip(labels[:-1], labels[1:]):
        arrow(ax, (left[0] + 1.7, 1.5), (right[0], 1.5))
    ax.set_title("Adversarially robust quantum-state tomography pipeline")
    save(fig, output_dir / "method_pipeline.png")


def physical_attack(output_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axis("off")
    box(ax, 0.5, 2.4, 2.0, 0.8, "N copies of ρ")
    box(ax, 3.0, 2.4, 2.0, 0.8, "m=αN copies\nreplaced by σ")
    box(ax, 5.5, 2.4, 2.0, 0.8, "ρeff=(1−α)ρ+ασ")
    arrow(ax, (2.5, 2.8), (3.0, 2.8))
    arrow(ax, (5.0, 2.8), (5.5, 2.8))
    box(ax, 1.0, 0.6, 6.0, 0.9, "Dtr(ρ,ρeff)=αDtr(ρ,σ)≤α and Dtr(ρ,ρeff)≤εp")
    arrow(ax, (6.5, 2.4), (4.0, 1.5))
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 4)
    ax.set_title("Trace-distance-constrained copy-replacement attack")
    save(fig, output_dir / "physical_attack_relation.png")


def measurement_layout(output_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    cases = [(1, 3, 2, 6), (2, 9, 4, 36), (3, 27, 8, 216)]
    for ax, (qubits, settings, outcomes, total) in zip(axes, cases):
        ax.axis("off")
        ax.set_title(f"{qubits} qubit{'s' if qubits > 1 else ''}")
        height = 0.72 / settings
        for setting in range(settings):
            y = 0.12 + setting * height
            ax.add_patch(plt.Rectangle((0.12, y), 0.76, height * 0.72, fill=False))
            if settings <= 9:
                ax.text(0.08, y + height * 0.36, f"s{setting+1}", ha="right", va="center", fontsize=7)
        ax.text(0.5, 0.93, f"{settings} settings × {outcomes} outcomes", ha="center")
        ax.text(0.5, 0.03, f"Input dimension = {total}", ha="center", weight="bold")
    fig.suptitle("Separately normalized Pauli-cube frequency blocks")
    save(fig, output_dir / "pauli_cube_layout.png")


def modularity(output_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.axis("off")
    box(ax, 0.5, 2.6, 2.2, 0.9, "Frequency vector\n6 / 36 / 216")
    for index, label in enumerate(["MLP (current)", "Residual MLP", "Transformer / CNN"]):
        box(ax, 3.5, 3.1 - index * 1.2, 2.4, 0.8, label)
        arrow(ax, (2.7, 3.05), (3.5, 3.5 - index * 1.2))
        arrow(ax, (5.9, 3.5 - index * 1.2), (7.0, 2.6))
    box(ax, 7.0, 2.1, 2.4, 1.0, "Shared physical head\nρ̂=TT†/Tr(TT†)")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.5)
    ax.set_title("Architecture-independent training and evaluation interface")
    save(fig, output_dir / "modular_architecture.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--figures-dir", default="figures/method")
    arguments = parser.parse_args()
    output_dir = ensure_dir(arguments.figures_dir)
    pipeline(output_dir)
    physical_attack(output_dir)
    measurement_layout(output_dir)
    modularity(output_dir)
    print(output_dir)


if __name__ == "__main__":
    main()
