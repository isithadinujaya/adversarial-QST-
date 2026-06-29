from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from qst_robust.attacks import (
    pgd_frequency_attack,
    random_physical_replacement_attack,
    sample_uniform_strength,
    targeted_physical_replacement_attack,
)
from qst_robust.config import ExperimentConfig
from qst_robust.dataset import build_datasets
from qst_robust.losses import reconstruction_loss_per_sample
from qst_robust.measurements import sample_pauli_frequencies
from qst_robust.metrics import reconstruction_metrics
from qst_robust.model import RobustQSTNetwork
from qst_robust.utils import choose_device, save_checkpoint, save_json, set_seed

RANDOM_PHYSICAL = 0
TARGETED_PHYSICAL = 1
PGD_FREQUENCY = 2


def balanced_attack_ids(batch_size: int, device: torch.device) -> torch.Tensor:
    ids = torch.arange(batch_size, device=device) % 3
    return ids[torch.randperm(batch_size, device=device)]


def build_adversarial_batch(
    model: RobustQSTNetwork,
    rho: torch.Tensor,
    clean_frequencies: torch.Tensor,
    config: ExperimentConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = rho.shape[0]
    attack_ids = balanced_attack_ids(batch, rho.device)
    attacked = torch.empty_like(clean_frequencies)

    random_mask = attack_ids == RANDOM_PHYSICAL
    if random_mask.any():
        count = int(random_mask.sum().item())
        alpha = sample_uniform_strength(
            count,
            config.attack.alpha_min,
            config.attack.alpha_max,
            device=rho.device,
        )
        attacked[random_mask] = random_physical_replacement_attack(
            rho[random_mask],
            alpha,
            shots_per_basis=config.data.shots_per_basis,
        )

    targeted_mask = attack_ids == TARGETED_PHYSICAL
    if targeted_mask.any():
        count = int(targeted_mask.sum().item())
        alpha = sample_uniform_strength(
            count,
            config.attack.alpha_min,
            config.attack.alpha_max,
            device=rho.device,
        )
        targeted, _ = targeted_physical_replacement_attack(
            rho[targeted_mask],
            alpha,
            shots_per_basis=config.data.shots_per_basis,
            minimum_target_trace_distance=config.attack.target_min_trace_distance,
        )
        attacked[targeted_mask] = targeted

    pgd_mask = attack_ids == PGD_FREQUENCY
    if pgd_mask.any():
        count = int(pgd_mask.sum().item())
        epsilon = sample_uniform_strength(
            count,
            config.attack.pgd_epsilon_min,
            config.attack.pgd_epsilon_max,
            device=rho.device,
        )
        attacked[pgd_mask] = pgd_frequency_attack(
            model,
            clean_frequencies[pgd_mask],
            rho[pgd_mask],
            epsilon,
            steps=config.attack.pgd_train_steps,
            restarts=1,
            mode="standard",
            fidelity_weight=config.training.fidelity_loss_weight,
        )

    return attacked, attack_ids


def run_epoch(
    model: RobustQSTNetwork,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: ExperimentConfig,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    totals = {
        "loss": 0.0,
        "clean_reconstruction": 0.0,
        "adversarial_reconstruction": 0.0,
        "detection": 0.0,
        "samples": 0,
    }

    for rho in loader:
        rho = rho.to(device, non_blocking=True)
        batch = rho.shape[0]
        clean = sample_pauli_frequencies(
            rho,
            shots_per_basis=config.data.shots_per_basis,
        )
        adversarial, _ = build_adversarial_batch(model, rho, clean, config)

        clean_estimate, clean_logits = model(clean)
        adversarial_estimate, adversarial_logits = model(adversarial)

        clean_loss = reconstruction_loss_per_sample(
            clean_estimate,
            rho,
            fidelity_weight=config.training.fidelity_loss_weight,
        ).mean()
        adversarial_loss = reconstruction_loss_per_sample(
            adversarial_estimate,
            rho,
            fidelity_weight=config.training.fidelity_loss_weight,
        ).mean()

        all_logits = torch.cat((clean_logits, adversarial_logits), dim=0)
        labels = torch.cat(
            (
                torch.zeros(batch, device=device),
                torch.ones(batch, device=device),
            ),
            dim=0,
        )
        detection_loss = F.binary_cross_entropy_with_logits(all_logits, labels)

        total_loss = (
            config.training.clean_loss_weight * clean_loss
            + config.training.adversarial_loss_weight * adversarial_loss
            + config.training.detection_loss_weight * detection_loss
        )

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=config.training.gradient_clip_norm,
        )
        optimizer.step()

        totals["loss"] += float(total_loss.item()) * batch
        totals["clean_reconstruction"] += float(clean_loss.item()) * batch
        totals["adversarial_reconstruction"] += float(adversarial_loss.item()) * batch
        totals["detection"] += float(detection_loss.item()) * batch
        totals["samples"] += batch

    samples = max(totals.pop("samples"), 1)
    return {key: value / samples for key, value in totals.items()}


def validate(
    model: RobustQSTNetwork,
    loader: DataLoader,
    config: ExperimentConfig,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    sums = {
        "clean_infidelity": 0.0,
        "adversarial_infidelity": 0.0,
        "clean_detection_accuracy": 0.0,
        "attack_detection_accuracy": 0.0,
        "samples": 0,
    }

    for rho in loader:
        rho = rho.to(device, non_blocking=True)
        batch = rho.shape[0]
        clean = sample_pauli_frequencies(
            rho,
            shots_per_basis=config.data.shots_per_basis,
        )
        adversarial, _ = build_adversarial_batch(model, rho, clean, config)

        with torch.no_grad():
            clean_estimate, clean_logits = model(clean)
            adversarial_estimate, adversarial_logits = model(adversarial)
            clean_metrics = reconstruction_metrics(clean_estimate, rho)
            adversarial_metrics = reconstruction_metrics(adversarial_estimate, rho)

            clean_predictions = (torch.sigmoid(clean_logits) >= 0.5).float()
            attack_predictions = (torch.sigmoid(adversarial_logits) >= 0.5).float()

        sums["clean_infidelity"] += float(clean_metrics["infidelity"].sum().item())
        sums["adversarial_infidelity"] += float(
            adversarial_metrics["infidelity"].sum().item()
        )
        sums["clean_detection_accuracy"] += float((clean_predictions == 0).sum().item())
        sums["attack_detection_accuracy"] += float((attack_predictions == 1).sum().item())
        sums["samples"] += batch

    samples = max(sums.pop("samples"), 1)
    return {key: value / samples for key, value in sums.items()}


def make_debug_config(config: ExperimentConfig) -> ExperimentConfig:
    debug = copy.deepcopy(config)
    debug.data.train_states = 384
    debug.data.validation_states = 96
    debug.data.test_states = 96
    debug.training.epochs = 1
    debug.training.batch_size = 128
    debug.attack.pgd_train_steps = 2
    debug.training.checkpoint_path = "checkpoints/debug_model.pt"
    return debug


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train adversarially robust qubit QST.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--debug", action="store_true", help="Run a small end-to-end training job.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig()
    if args.debug:
        config = make_debug_config(config)
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.checkpoint is not None:
        config.training.checkpoint_path = args.checkpoint
    config.validate()

    set_seed(config.training.seed)
    device = choose_device(args.device)
    print(f"Using device: {device}")
    print(
        f"Shots: {config.data.shots_per_basis} per Pauli basis "
        f"({3 * config.data.shots_per_basis} total copies per state)."
    )

    train_dataset, validation_dataset, _ = build_datasets(
        config.data,
        config.training.seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = RobustQSTNetwork(config.model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    config_path = Path(config.training.checkpoint_path).with_suffix(".config.json")
    save_json(config_path, config.to_dict())

    best_score = float("inf")
    for epoch in range(1, config.training.epochs + 1):
        training = run_epoch(model, train_loader, optimizer, config, device)
        validation = validate(model, validation_loader, config, device)
        score = validation["adversarial_infidelity"]

        if score < best_score:
            best_score = score
            save_checkpoint(
                config.training.checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                config=config,
                validation_score=score,
            )

        print(
            f"Epoch {epoch:03d} | "
            f"loss={training['loss']:.6f} | "
            f"clean_inf={validation['clean_infidelity']:.6f} | "
            f"adv_inf={validation['adversarial_infidelity']:.6f} | "
            f"clean_det={validation['clean_detection_accuracy']:.3f} | "
            f"attack_det={validation['attack_detection_accuracy']:.3f}"
        )

    print(f"Best checkpoint: {config.training.checkpoint_path}")


if __name__ == "__main__":
    main()
