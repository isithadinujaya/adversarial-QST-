from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from qst.config import config_from_dict, load_config
from qst.data import build_dataset
from qst.evaluation import evaluate_model
from qst.models import build_model
from qst.utils.seed import resolve_device, seed_everything


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate an adversarial QST checkpoint and save performance graphs."
        )
    )
    parser.add_argument("--config", required=True, help="Path to YAML configuration.")
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt or last.pt.")
    parser.add_argument(
        "--output-dir",
        help=(
            "Directory for evaluation.json and PNG figures. Defaults to "
            "<training output directory>/plots."
        ),
    )
    parser.add_argument(
        "--report",
        help=(
            "Optional existing evaluation JSON. When supplied, the test set is not "
            "evaluated again; graphs are made from this report."
        ),
    )
    parser.add_argument("--dpi", type=int, default=180, help="PNG resolution.")
    return parser.parse_args()


def _pretty_name(name: str) -> str:
    replacements = {
        "clean": "Clean",
        "random_replacement": "Random replacement",
        "targeted_replacement": "Targeted replacement",
        "worst_case_replacement": "Worst-case replacement",
        "random_frequency": "Random frequency",
        "pgd": "PGD",
    }
    return replacements.get(name, name.replace("_", " ").title())


def _save_current_figure(path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved {path.resolve()}")


def _plot_line_series(
    series: list[tuple[str, list[float]]],
    *,
    title: str,
    ylabel: str,
    output_path: Path,
    dpi: int,
) -> bool:
    nonempty = [(label, values) for label, values in series if values]
    if not nonempty:
        return False

    plt.figure(figsize=(8.5, 5.2))
    for label, values in nonempty:
        epochs = list(range(1, len(values) + 1))
        plt.plot(epochs, values, marker="o", markersize=3, linewidth=1.8, label=label)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    _save_current_figure(output_path, dpi)
    return True


def _plot_training_history(
    history: dict[str, list[float]],
    output_directory: Path,
    dpi: int,
) -> list[Path]:
    generated: list[Path] = []

    total_path = output_directory / "01_training_total_loss.png"
    if _plot_line_series(
        [
            ("Training", history.get("train_total", [])),
            ("Validation", history.get("validation_total", [])),
        ],
        title="Training and validation total loss",
        ylabel="Total loss",
        output_path=total_path,
        dpi=dpi,
    ):
        generated.append(total_path)

    clean_path = output_directory / "02_clean_reconstruction_loss.png"
    if _plot_line_series(
        [
            ("Training", history.get("train_clean_reconstruction", [])),
            ("Validation", history.get("validation_clean_reconstruction", [])),
        ],
        title="Clean-state reconstruction loss",
        ylabel="Reconstruction loss",
        output_path=clean_path,
        dpi=dpi,
    ):
        generated.append(clean_path)

    attacked_path = output_directory / "03_adversarial_reconstruction_loss.png"
    if _plot_line_series(
        [
            ("Training", history.get("train_adversarial_reconstruction", [])),
            ("Validation", history.get("validation_adversarial_reconstruction", [])),
        ],
        title="Fixed-attack reconstruction loss",
        ylabel="Reconstruction loss",
        output_path=attacked_path,
        dpi=dpi,
    ):
        generated.append(attacked_path)

    pgd_path = output_directory / "04_pgd_training_loss.png"
    if _plot_line_series(
        [("Training PGD", history.get("train_pgd_reconstruction", []))],
        title="PGD reconstruction loss during training",
        ylabel="PGD reconstruction loss",
        output_path=pgd_path,
        dpi=dpi,
    ):
        generated.append(pgd_path)

    detection_path = output_directory / "05_detection_loss.png"
    if _plot_line_series(
        [
            ("Training", history.get("train_detection", [])),
            ("Validation", history.get("validation_detection", [])),
        ],
        title="Attack-detection loss",
        ylabel="Binary cross-entropy loss",
        output_path=detection_path,
        dpi=dpi,
    ):
        generated.append(detection_path)

    return generated


def _attack_metric_rows(report: dict[str, Any], metric: str) -> tuple[list[str], list[float]]:
    labels: list[str] = []
    values: list[float] = []

    per_attack = report.get("per_attack", {})
    if isinstance(per_attack, dict):
        preferred_order = [
            "clean",
            "random_replacement",
            "targeted_replacement",
            "worst_case_replacement",
            "random_frequency",
        ]
        for attack_name in preferred_order:
            metrics = per_attack.get(attack_name)
            if isinstance(metrics, dict) and metric in metrics:
                labels.append(_pretty_name(attack_name))
                values.append(float(metrics[metric]))

        for attack_name, metrics in per_attack.items():
            if attack_name in preferred_order:
                continue
            if isinstance(metrics, dict) and metric in metrics:
                labels.append(_pretty_name(attack_name))
                values.append(float(metrics[metric]))

    pgd = report.get("pgd")
    if isinstance(pgd, dict) and metric in pgd:
        labels.append("PGD")
        values.append(float(pgd[metric]))

    return labels, values


def _plot_attack_metric(
    report: dict[str, Any],
    *,
    metric: str,
    title: str,
    ylabel: str,
    output_path: Path,
    dpi: int,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
) -> bool:
    labels, values = _attack_metric_rows(report, metric)
    if not values:
        return False

    plt.figure(figsize=(10.5, 5.8))
    positions = list(range(len(labels)))
    bars = plt.bar(positions, values)
    plt.xticks(positions, labels, rotation=25, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)
    if lower_bound is not None or upper_bound is not None:
        current_lower, current_upper = plt.ylim()
        plt.ylim(
            lower_bound if lower_bound is not None else current_lower,
            upper_bound if upper_bound is not None else current_upper,
        )

    for bar, value in zip(bars, values, strict=True):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    _save_current_figure(output_path, dpi)
    return True


def _plot_detection_metrics(
    report: dict[str, Any], output_path: Path, dpi: int
) -> bool:
    detection = report.get("detection")
    if not isinstance(detection, dict):
        return False

    metric_names = ["accuracy", "precision", "recall", "f1", "auroc"]
    labels = ["Accuracy", "Precision", "Recall", "F1", "AUROC"]
    values = [float(detection[name]) for name in metric_names if name in detection]
    labels = [label for label, name in zip(labels, metric_names, strict=True) if name in detection]
    if not values:
        return False

    plt.figure(figsize=(8.5, 5.2))
    positions = list(range(len(labels)))
    bars = plt.bar(positions, values)
    plt.xticks(positions, labels)
    plt.ylim(0.0, 1.05)
    plt.ylabel("Score")
    plt.title("Attack-detection performance")
    plt.grid(axis="y", alpha=0.3)

    for bar, value in zip(bars, values, strict=True):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    _save_current_figure(output_path, dpi)
    return True


def _load_or_create_report(
    arguments: argparse.Namespace,
    output_directory: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint_path = Path(arguments.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    file_config = load_config(arguments.config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    checkpoint_config_raw = checkpoint.get("config")
    if isinstance(checkpoint_config_raw, dict):
        config = config_from_dict(checkpoint_config_raw)
        print(
            "Using model configuration saved in checkpoint: "
            f"{config.model.name}"
        )
    else:
        config = file_config
        print(
            "Checkpoint has no saved configuration; using the supplied YAML "
            f"model: {config.model.name}"
        )

    # Runtime-only choices may safely come from the supplied YAML. The model,
    # quantum dimensions, attacks, and losses remain exactly as saved.
    config.training.device = file_config.training.device
    config.training.batch_size = file_config.training.batch_size
    config.training.data.num_workers = file_config.training.data.num_workers

    seed_everything(config.training.seed)
    device = resolve_device(config.training.device)
    model = build_model(config)
    if config.quantum.complex_dtype == "complex128":
        model = model.double()
    model = model.to(device)
    model.load_state_dict(checkpoint["model_state"])

    if arguments.report:
        report_path = Path(arguments.report)
        if not report_path.exists():
            raise FileNotFoundError(f"Evaluation report not found: {report_path}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        print(f"Loaded existing report from {report_path.resolve()}")
        return report, checkpoint

    dataset = build_dataset(config, "test")
    loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.data.num_workers,
        pin_memory=device.type == "cuda",
    )
    report = evaluate_model(model, loader, device, config)
    report_path = output_directory / "evaluation.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved report to {report_path.resolve()}")
    return report, checkpoint


def main() -> None:
    arguments = parse_arguments()
    config = load_config(arguments.config)
    output_directory = Path(
        arguments.output_dir
        or (Path(config.training.output_directory) / "plots")
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    report, checkpoint = _load_or_create_report(arguments, output_directory)
    history = checkpoint.get("history", {})
    if not isinstance(history, dict):
        history = {}

    generated = _plot_training_history(history, output_directory, arguments.dpi)

    metric_figures = [
        (
            "fidelity",
            "Reconstruction fidelity by input attack",
            "Mean fidelity",
            output_directory / "06_fidelity_by_attack.png",
            0.0,
            1.0,
        ),
        (
            "trace_distance",
            "Trace distance by input attack",
            "Mean trace distance",
            output_directory / "07_trace_distance_by_attack.png",
            0.0,
            None,
        ),
        (
            "frobenius",
            "Frobenius distance by input attack",
            "Mean Frobenius distance",
            output_directory / "08_frobenius_distance_by_attack.png",
            0.0,
            None,
        ),
    ]
    for metric, title, ylabel, path, lower, upper in metric_figures:
        if _plot_attack_metric(
            report,
            metric=metric,
            title=title,
            ylabel=ylabel,
            output_path=path,
            dpi=arguments.dpi,
            lower_bound=lower,
            upper_bound=upper,
        ):
            generated.append(path)

    detection_path = output_directory / "09_detection_metrics.png"
    if _plot_detection_metrics(report, detection_path, arguments.dpi):
        generated.append(detection_path)

    manifest = {
        "checkpoint": str(Path(arguments.checkpoint).resolve()),
        "best_epoch": checkpoint.get("epoch"),
        "best_validation_loss": checkpoint.get("best_validation_loss"),
        "sample_count": report.get("sample_count"),
        "figures": [path.name for path in generated],
    }
    manifest_path = output_directory / "plot_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved {manifest_path.resolve()}")
    print(f"Generated {len(generated)} graph(s) in {output_directory.resolve()}")


if __name__ == "__main__":
    main()
