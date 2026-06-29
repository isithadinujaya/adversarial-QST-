from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import DataLoader, Subset

from qst_robust.attacks import (
    pgd_frequency_attack,
    random_physical_replacement_attack,
    targeted_physical_replacement_attack,
)
from qst_robust.config import ExperimentConfig
from qst_robust.dataset import build_datasets
from qst_robust.measurements import sample_pauli_frequencies
from qst_robust.metrics import reconstruction_metrics
from qst_robust.model import RobustQSTNetwork
from qst_robust.utils import choose_device, set_seed


@torch.no_grad()
def _accumulate_batch(
    model: RobustQSTNetwork,
    frequencies: torch.Tensor,
    rho: torch.Tensor,
    *,
    attack_label: int,
) -> dict[str, float]:
    estimate, logits = model(frequencies)
    metrics = reconstruction_metrics(estimate, rho)
    predictions = (torch.sigmoid(logits) >= 0.5).to(torch.long)
    labels = torch.full_like(predictions, attack_label)
    return {
        "samples": rho.shape[0],
        "fidelity": float(metrics["fidelity"].sum().item()),
        "infidelity": float(metrics["infidelity"].sum().item()),
        "trace_distance": float(metrics["trace_distance"].sum().item()),
        "frobenius": float(metrics["frobenius"].sum().item()),
        "detection_accuracy": float((predictions == labels).sum().item()),
        "predicted_attack": float(predictions.sum().item()),
    }


def evaluate_condition(
    model: RobustQSTNetwork,
    loader: DataLoader,
    config: ExperimentConfig,
    device: torch.device,
    *,
    attack_label: int,
    attack_builder: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None,
) -> dict[str, float]:
    model.eval()
    totals = {
        "samples": 0.0,
        "fidelity": 0.0,
        "infidelity": 0.0,
        "trace_distance": 0.0,
        "frobenius": 0.0,
        "detection_accuracy": 0.0,
        "predicted_attack": 0.0,
    }

    for rho in loader:
        rho = rho.to(device)
        clean = sample_pauli_frequencies(
            rho,
            shots_per_basis=config.data.shots_per_basis,
        )
        frequencies = clean if attack_builder is None else attack_builder(rho, clean)
        with torch.no_grad():
            batch = _accumulate_batch(
                model,
                frequencies,
                rho,
                attack_label=attack_label,
            )
        for key in totals:
            totals[key] += batch[key]

    samples = max(totals.pop("samples"), 1.0)
    result = {key: value / samples for key, value in totals.items()}
    result["attack_recall"] = result["predicted_attack"] if attack_label == 1 else float("nan")
    result["false_positive_rate"] = result["predicted_attack"] if attack_label == 0 else float("nan")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate robust QST attacks.")
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="results/evaluation.csv")
    parser.add_argument("--max-states", type=int, default=None)
    parser.add_argument("--pgd-restarts", type=int, default=None)
    parser.add_argument("--pgd-steps", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ExperimentConfig.from_dict(checkpoint["config"])
    set_seed(config.training.seed + 100)

    model = RobustQSTNetwork(config.model).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    _, _, test_dataset = build_datasets(config.data, config.training.seed)
    if args.max_states is not None:
        count = min(args.max_states, len(test_dataset))
        test_dataset = Subset(test_dataset, range(count))
    loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
    )

    rows: list[dict[str, float | str]] = []

    clean_metrics = evaluate_condition(
        model,
        loader,
        config,
        device,
        attack_label=0,
        attack_builder=None,
    )
    rows.append({"attack": "clean", "strength": 0.0, **clean_metrics})

    alpha_values = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    for alpha_value in alpha_values:
        def random_builder(rho: torch.Tensor, clean: torch.Tensor, a: float = alpha_value) -> torch.Tensor:
            alpha = torch.full((rho.shape[0],), a, device=rho.device)
            return random_physical_replacement_attack(
                rho,
                alpha,
                shots_per_basis=config.data.shots_per_basis,
            )

        random_metrics = evaluate_condition(
            model,
            loader,
            config,
            device,
            attack_label=1,
            attack_builder=random_builder,
        )
        rows.append({"attack": "random_physical", "strength": alpha_value, **random_metrics})

        def targeted_builder(rho: torch.Tensor, clean: torch.Tensor, a: float = alpha_value) -> torch.Tensor:
            alpha = torch.full((rho.shape[0],), a, device=rho.device)
            frequencies, _ = targeted_physical_replacement_attack(
                rho,
                alpha,
                shots_per_basis=config.data.shots_per_basis,
                minimum_target_trace_distance=config.attack.target_min_trace_distance,
            )
            return frequencies

        targeted_metrics = evaluate_condition(
            model,
            loader,
            config,
            device,
            attack_label=1,
            attack_builder=targeted_builder,
        )
        rows.append({"attack": "targeted_physical", "strength": alpha_value, **targeted_metrics})

    pgd_restarts = args.pgd_restarts or config.attack.pgd_eval_restarts
    pgd_steps = args.pgd_steps or config.attack.pgd_eval_steps
    epsilon_values = [0.005, 0.010, 0.020, 0.030, 0.050]
    for epsilon_value in epsilon_values:
        for mode in ("standard", "adaptive"):
            def pgd_builder(
                rho: torch.Tensor,
                clean: torch.Tensor,
                eps: float = epsilon_value,
                selected_mode: str = mode,
            ) -> torch.Tensor:
                return pgd_frequency_attack(
                    model,
                    clean,
                    rho,
                    eps,
                    steps=pgd_steps,
                    restarts=pgd_restarts,
                    mode=selected_mode,  # type: ignore[arg-type]
                    adaptive_gamma=config.attack.adaptive_pgd_gamma,
                    fidelity_weight=config.training.fidelity_loss_weight,
                )

            pgd_metrics = evaluate_condition(
                model,
                loader,
                config,
                device,
                attack_label=1,
                attack_builder=pgd_builder,
            )
            rows.append(
                {
                    "attack": f"pgd_{mode}",
                    "strength": epsilon_value,
                    **pgd_metrics,
                }
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"{row['attack']:>20} strength={row['strength']:<6} "
            f"fidelity={row['fidelity']:.5f} "
            f"trace={row['trace_distance']:.5f} "
            f"det_acc={row['detection_accuracy']:.3f}"
        )
    print(f"Saved results to {output}")


if __name__ == "__main__":
    main()
