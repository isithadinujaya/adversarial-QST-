from __future__ import annotations

import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from qst.attacks import frequency_pgd_attack, physical_replacement_attack
from qst.config import QSTConfig
from qst.losses import RobustTomographyLoss
from qst.measurements import PauliCubeMeasurement
from qst.metrics import quantum_fidelity
from qst.utils import ensure_dir, save_json


@dataclass
class EpochResult:
    total: float
    clean: float
    adversarial: float
    physical: float
    pgd: float
    consistency: float
    physical_consistency: float
    pgd_consistency: float
    clean_fidelity: float
    physical_fidelity: float
    pgd_fidelity: float
    adversarial_fidelity: float


class RobustQSTTrainer:
    def __init__(
        self,
        config: QSTConfig,
        model: nn.Module,
        measurement: PauliCubeMeasurement,
        device: torch.device,
    ) -> None:
        self.config = config
        self.model = model.to(device)
        self.measurement = measurement.to(device)
        self.device = device
        self.output_dir = ensure_dir(config.experiment.output_dir)
        self.loss_function = RobustTomographyLoss(
            clean_weight=config.loss.clean_weight,
            physical_weight=config.loss.physical_weight,
            pgd_weight=config.loss.pgd_weight,
            consistency_weight=config.loss.consistency_weight,
            physical_max_weight=config.loss.physical_max_weight,
        )
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=config.training.scheduler_factor,
            patience=config.training.scheduler_patience,
            min_lr=config.training.min_learning_rate,
        )
        self.train_generator = torch.Generator(device=device.type)
        self.train_generator.manual_seed(config.experiment.seed + 404)
        self.validation_seed = config.experiment.seed + 505
        self.history_path = self.output_dir / "history.csv"

    def _uniform_batch(
        self,
        minimum: float,
        maximum: float,
        batch_size: int,
        *,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """Draw one independent attack strength for every state."""
        if minimum == maximum:
            return torch.full(
                (batch_size,),
                minimum,
                device=self.device,
                dtype=torch.float32,
            )
        values = torch.rand(
            batch_size,
            generator=generator,
            device=self.device,
            dtype=torch.float32,
        )
        return minimum + (maximum - minimum) * values

    def _make_physical_frequencies(
        self,
        rho: torch.Tensor,
        attack_kind: str,
        *,
        generator: torch.Generator,
    ) -> torch.Tensor:
        alpha = self._uniform_batch(
            self.config.attack.alpha_min,
            self.config.attack.alpha_max,
            rho.shape[0],
            generator=generator,
        )
        result = physical_replacement_attack(
            rho,
            alpha=alpha,
            epsilon_physical=self.config.attack.epsilon_physical,
            kind=attack_kind,
            target_state=self.config.attack.target_state,
            target_min_trace_distance=self.config.attack.target_min_trace_distance,
            generator=generator,
        )
        return self.measurement.sample_frequencies(
            result.attacked_state,
            self.config.data.shots_per_setting,
            generator=generator,
        )

    def _make_pgd_frequencies(
        self,
        rho: torch.Tensor,
        clean_frequencies: torch.Tensor,
        *,
        training: bool,
        generator: torch.Generator,
    ) -> torch.Tensor:
        epsilon = self._uniform_batch(
            self.config.attack.epsilon_frequency_min,
            self.config.attack.epsilon_frequency_max,
            rho.shape[0],
            generator=generator,
        )
        steps = (
            self.config.attack.pgd_train_steps
            if training
            else self.config.attack.pgd_eval_steps
        )
        return frequency_pgd_attack(
            self.model,
            clean_frequencies,
            rho,
            epsilon=epsilon,
            num_settings=self.config.num_settings,
            outcomes_per_setting=self.config.dimension,
            steps=steps,
            step_size=self.config.attack.pgd_step_size,
            random_start=self.config.attack.pgd_random_start,
            generator=generator,
        )

    def _run_epoch(
        self,
        loader: DataLoader,
        *,
        training: bool,
        epoch: int,
    ) -> EpochResult:
        self.model.train(training)
        totals = {
            "total": 0.0,
            "clean": 0.0,
            "adversarial": 0.0,
            "physical": 0.0,
            "pgd": 0.0,
            "consistency": 0.0,
            "physical_consistency": 0.0,
            "pgd_consistency": 0.0,
            "clean_fidelity": 0.0,
            "physical_fidelity": 0.0,
            "pgd_fidelity": 0.0,
            "adversarial_fidelity": 0.0,
        }
        examples = 0

        if training:
            generator = self.train_generator
        else:
            generator = torch.Generator(device=self.device.type)
            generator.manual_seed(self.validation_seed + epoch)

        iterator = tqdm(loader, leave=False, desc="train" if training else "val")
        for batch in iterator:
            rho = batch["rho"].to(self.device)
            batch_size = rho.shape[0]

            clean_frequencies = self.measurement.sample_frequencies(
                rho,
                self.config.data.shots_per_setting,
                generator=generator,
            )

            # Every state receives every configured physical attack.
            physical_frequency_sets = [
                self._make_physical_frequencies(
                    rho,
                    attack_kind,
                    generator=generator,
                )
                for attack_kind in self.config.attack.physical_training_types
            ]

            # PGD is a separate frequency-space threat family.
            pgd_frequencies = self._make_pgd_frequencies(
                rho,
                clean_frequencies,
                training=training,
                generator=generator,
            )

            if training:
                self.optimizer.zero_grad(set_to_none=True)
                clean_prediction = self.model(clean_frequencies)
                physical_prediction_list = [
                    self.model(frequencies)
                    for frequencies in physical_frequency_sets
                ]
                physical_predictions = torch.stack(
                    physical_prediction_list,
                    dim=1,
                )
                pgd_prediction = self.model(pgd_frequencies)
                losses = self.loss_function(
                    rho,
                    clean_prediction,
                    physical_predictions,
                    pgd_prediction,
                )
                losses.total.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.gradient_clip_norm,
                )
                self.optimizer.step()
            else:
                with torch.no_grad():
                    clean_prediction = self.model(clean_frequencies)
                    physical_prediction_list = [
                        self.model(frequencies)
                        for frequencies in physical_frequency_sets
                    ]
                    physical_predictions = torch.stack(
                        physical_prediction_list,
                        dim=1,
                    )
                    pgd_prediction = self.model(pgd_frequencies)
                    losses = self.loss_function(
                        rho,
                        clean_prediction,
                        physical_predictions,
                        pgd_prediction,
                    )

            with torch.no_grad():
                clean_fidelity = quantum_fidelity(
                    rho,
                    clean_prediction,
                    epsilon=self.config.loss.fidelity_epsilon,
                ).mean()

                physical_fidelities = torch.stack(
                    [
                        quantum_fidelity(
                            rho,
                            prediction,
                            epsilon=self.config.loss.fidelity_epsilon,
                        )
                        for prediction in physical_prediction_list
                    ],
                    dim=1,
                )
                # Fidelity is larger when better, so the worst attack is the minimum.
                physical_fidelity = physical_fidelities.min(dim=1).values.mean()
                pgd_fidelity = quantum_fidelity(
                    rho,
                    pgd_prediction,
                    epsilon=self.config.loss.fidelity_epsilon,
                ).mean()
                adversarial_fidelity = (
                    self.config.loss.physical_weight * physical_fidelity
                    + self.config.loss.pgd_weight * pgd_fidelity
                )

                values = losses.detached_dict()
                for key in (
                    "total",
                    "clean",
                    "adversarial",
                    "physical",
                    "pgd",
                    "consistency",
                    "physical_consistency",
                    "pgd_consistency",
                ):
                    totals[key] += values[key] * batch_size
                totals["clean_fidelity"] += float(clean_fidelity.cpu()) * batch_size
                totals["physical_fidelity"] += float(physical_fidelity.cpu()) * batch_size
                totals["pgd_fidelity"] += float(pgd_fidelity.cpu()) * batch_size
                totals["adversarial_fidelity"] += (
                    float(adversarial_fidelity.cpu()) * batch_size
                )
                examples += batch_size

            iterator.set_postfix(total=f"{totals['total'] / examples:.4f}")

        return EpochResult(
            **{key: value / examples for key, value in totals.items()}
        )

    def _save_checkpoint(
        self,
        path: Path,
        epoch: int,
        best_validation: float,
    ) -> None:
        torch.save(
            {
                "epoch": epoch,
                "best_validation": best_validation,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "config": self.config.as_dict(),
            },
            path,
        )

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> dict[str, float]:
        save_json(self.output_dir / "resolved_config.json", self.config.as_dict())
        best_validation = math.inf
        patience = 0
        history_rows: list[dict[str, float | int]] = []
        start_time = time.time()

        for epoch in range(1, self.config.training.epochs + 1):
            train_result = self._run_epoch(train_loader, training=True, epoch=epoch)
            validation_result = self._run_epoch(
                val_loader,
                training=False,
                epoch=epoch,
            )
            self.scheduler.step(validation_result.total)
            learning_rate = self.optimizer.param_groups[0]["lr"]
            row = {
                "epoch": epoch,
                "learning_rate": learning_rate,
                **{f"train_{k}": v for k, v in train_result.__dict__.items()},
                **{f"val_{k}": v for k, v in validation_result.__dict__.items()},
            }
            history_rows.append(row)
            with self.history_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
                writer.writeheader()
                writer.writerows(history_rows)

            self._save_checkpoint(
                self.output_dir / "last.pt",
                epoch,
                best_validation,
            )
            if validation_result.total < best_validation:
                best_validation = validation_result.total
                patience = 0
                self._save_checkpoint(
                    self.output_dir / "best.pt",
                    epoch,
                    best_validation,
                )
            else:
                patience += 1

            if epoch % self.config.training.log_every == 0:
                print(
                    f"Epoch {epoch:03d} | "
                    f"train={train_result.total:.6f} | "
                    f"val={validation_result.total:.6f} | "
                    f"clean F={validation_result.clean_fidelity:.6f} | "
                    f"physical F={validation_result.physical_fidelity:.6f} | "
                    f"PGD F={validation_result.pgd_fidelity:.6f} | "
                    f"lr={learning_rate:.2e}"
                )
            if patience >= self.config.training.early_stopping_patience:
                print(f"Early stopping at epoch {epoch}.")
                break

        summary = {
            "best_validation_total": best_validation,
            "epochs_completed": len(history_rows),
            "elapsed_seconds": time.time() - start_time,
            "best_checkpoint": str(self.output_dir / "best.pt"),
        }
        save_json(self.output_dir / "training_summary.json", summary)
        return summary
