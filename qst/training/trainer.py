from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from qst.attacks.frequency import pgd_frequency_attack
from qst.config import ExperimentConfig
from qst.quantum.measurements import PauliMeasurementScheme
from qst.training.checkpoint import save_checkpoint
from qst.training.losses import (
    detection_loss,
    measurement_consistency_loss,
    reconstruction_loss,
)


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        config: ExperimentConfig,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.scheme = PauliMeasurementScheme(config.quantum.number_qubits)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        self.history: dict[str, list[float]] = defaultdict(list)

    def _move_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {name: tensor.to(self.device, non_blocking=True) for name, tensor in batch.items()}

    def _batch_losses(
        self,
        batch: dict[str, torch.Tensor],
        training: bool,
    ) -> dict[str, torch.Tensor]:
        loss_config = self.config.loss
        target = batch["target_density"]
        clean_frequencies = batch["clean_frequencies"]
        input_frequencies = batch["input_frequencies"]
        labels = batch["attack_label"]

        clean_density, clean_logit = self.model(clean_frequencies)
        clean_reconstruction = reconstruction_loss(
            clean_density,
            target,
            loss_config.reconstruction,
        )
        clean_detection = detection_loss(clean_logit, torch.zeros_like(clean_logit))
        consistency = measurement_consistency_loss(
            clean_density,
            clean_frequencies,
            self.scheme,
        )

        attacked_mask = labels > 0.5
        zero = clean_reconstruction.new_zeros(())
        adversarial_reconstruction = zero
        adversarial_detection = zero
        if torch.any(attacked_mask):
            attacked_density, attacked_logit = self.model(input_frequencies[attacked_mask])
            adversarial_reconstruction = reconstruction_loss(
                attacked_density,
                target[attacked_mask],
                loss_config.reconstruction,
            )
            adversarial_detection = detection_loss(
                attacked_logit,
                torch.ones_like(attacked_logit),
            )

        pgd_reconstruction = zero
        pgd_detection = zero
        pgd_config = self.config.attacks.pgd
        if training and pgd_config.enabled and pgd_config.fraction > 0:
            count = max(1, int(round(clean_frequencies.shape[0] * pgd_config.fraction)))
            count = min(count, clean_frequencies.shape[0])
            selected = torch.randperm(clean_frequencies.shape[0], device=self.device)[:count]
            pgd_input = pgd_frequency_attack(
                self.model,
                clean_frequencies[selected],
                target[selected],
                epsilon=pgd_config.epsilon,
                step_size=pgd_config.step_size,
                steps=pgd_config.steps,
                number_settings=self.scheme.number_settings,
                outcomes_per_setting=self.scheme.outcomes_per_setting,
                random_start=pgd_config.random_start,
                detection_evasion_weight=pgd_config.detection_evasion_weight,
            )
            pgd_density, pgd_logit = self.model(pgd_input)
            pgd_reconstruction = reconstruction_loss(
                pgd_density,
                target[selected],
                loss_config.reconstruction,
            )
            pgd_detection = detection_loss(pgd_logit, torch.ones_like(pgd_logit))

        combined_detection = clean_detection + adversarial_detection + pgd_detection
        total = (
            loss_config.clean_reconstruction_weight * clean_reconstruction
            + loss_config.adversarial_reconstruction_weight * adversarial_reconstruction
            + loss_config.pgd_reconstruction_weight * pgd_reconstruction
            + loss_config.detection_weight * combined_detection
            + loss_config.measurement_consistency_weight * consistency
        )
        return {
            "total": total,
            "clean_reconstruction": clean_reconstruction,
            "adversarial_reconstruction": adversarial_reconstruction,
            "pgd_reconstruction": pgd_reconstruction,
            "detection": combined_detection,
            "measurement_consistency": consistency,
        }

    def run_epoch(self, loader: DataLoader, training: bool) -> dict[str, float]:
        self.model.train(training)
        totals: dict[str, float] = defaultdict(float)
        sample_count = 0
        description = "train" if training else "validation"
        for batch in tqdm(loader, desc=description, leave=False):
            batch = self._move_batch(batch)
            batch_size = batch["target_density"].shape[0]
            if training:
                self.optimizer.zero_grad(set_to_none=True)
                losses = self._batch_losses(batch, training=True)
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.gradient_clip_norm,
                )
                self.optimizer.step()
            else:
                with torch.no_grad():
                    losses = self._batch_losses(batch, training=False)
            for name, value in losses.items():
                totals[name] += float(value.detach().cpu()) * batch_size
            sample_count += batch_size
        return {name: value / max(sample_count, 1) for name, value in totals.items()}

    def fit(self, train_loader: DataLoader, validation_loader: DataLoader) -> Path:
        output_directory = Path(self.config.training.output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)
        best_path = output_directory / "best.pt"
        best_validation = float("inf")
        epochs_without_improvement = 0

        for epoch in range(1, self.config.training.epochs + 1):
            train_metrics = self.run_epoch(train_loader, training=True)
            validation_metrics = self.run_epoch(validation_loader, training=False)
            for name, value in train_metrics.items():
                self.history[f"train_{name}"].append(value)
            for name, value in validation_metrics.items():
                self.history[f"validation_{name}"].append(value)

            print(
                f"Epoch {epoch:03d} | "
                f"train total={train_metrics['total']:.6f} | "
                f"validation total={validation_metrics['total']:.6f} | "
                f"clean={validation_metrics['clean_reconstruction']:.6f} | "
                f"attacked={validation_metrics['adversarial_reconstruction']:.6f}"
            )

            improved = validation_metrics["total"] < best_validation
            if improved:
                best_validation = validation_metrics["total"]
                epochs_without_improvement = 0
                save_checkpoint(
                    best_path,
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    best_validation_loss=best_validation,
                    config=self.config,
                    history=dict(self.history),
                )
            else:
                epochs_without_improvement += 1

            save_checkpoint(
                output_directory / "last.pt",
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch,
                best_validation_loss=best_validation,
                config=self.config,
                history=dict(self.history),
            )
            with (output_directory / "history.json").open("w", encoding="utf-8") as handle:
                json.dump(dict(self.history), handle, indent=2)

            patience = self.config.training.early_stopping_patience
            if patience > 0 and epochs_without_improvement >= patience:
                print(f"Early stopping after {epoch} epochs.")
                break

        return best_path
