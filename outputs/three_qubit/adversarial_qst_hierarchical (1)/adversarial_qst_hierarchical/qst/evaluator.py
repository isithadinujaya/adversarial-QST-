from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from qst.attacks import frequency_pgd_attack, physical_replacement_attack
from qst.config import QSTConfig
from qst.measurements import PauliCubeMeasurement
from qst.metrics import (
    frobenius_distance,
    infidelity,
    physicality_metrics,
    purity,
    quantum_fidelity,
    trace_distance,
)
from qst.utils import ensure_dir, save_json


@dataclass
class AttackParameters:
    attack: str
    alpha: float = 0.0
    epsilon_frequency: float = 0.0
    shots: int | None = None
    combined_physical_kind: str = "random_replacement"


class QSTEvaluator:
    def __init__(
        self,
        config: QSTConfig,
        model: nn.Module,
        measurement: PauliCubeMeasurement,
        device: torch.device,
    ) -> None:
        self.config = config
        self.model = model.to(device).eval()
        self.measurement = measurement.to(device)
        self.device = device
        self.generator = torch.Generator(device=device.type)
        self.generator.manual_seed(config.experiment.seed + 606)

    def _limit_loader(self, loader: DataLoader) -> Iterable[dict[str, torch.Tensor]]:
        seen = 0
        maximum = self.config.evaluation.max_samples
        for batch in loader:
            if seen >= maximum:
                break
            remaining = maximum - seen
            if batch["rho"].shape[0] > remaining:
                batch = {key: value[:remaining] for key, value in batch.items()}
            seen += batch["rho"].shape[0]
            yield batch

    def _make_input(
        self,
        rho: torch.Tensor,
        parameters: AttackParameters,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | None]]:
        shots = parameters.shots or self.config.data.shots_per_setting
        clean = self.measurement.sample_frequencies(
            rho, shots, generator=self.generator
        )
        metadata: dict[str, torch.Tensor | None] = {
            "alpha_requested": torch.zeros(rho.shape[0], device=self.device),
            "alpha_effective": torch.zeros(rho.shape[0], device=self.device),
            "epsilon_physical_actual": torch.zeros(rho.shape[0], device=self.device),
            "target_state": None,
        }
        if parameters.attack == "clean":
            return clean, metadata
        if parameters.attack in {
            "random_replacement",
            "targeted_replacement",
            "fixed_replacement",
            "worst_replacement",
        }:
            result = physical_replacement_attack(
                rho,
                alpha=parameters.alpha,
                epsilon_physical=self.config.attack.epsilon_physical,
                kind=parameters.attack,
                target_state=self.config.attack.target_state,
                target_min_trace_distance=self.config.attack.target_min_trace_distance,
                generator=self.generator,
            )
            attacked = self.measurement.sample_frequencies(
                result.attacked_state, shots, generator=self.generator
            )
            metadata.update(
                {
                    "alpha_requested": result.alpha_requested,
                    "alpha_effective": result.alpha_effective,
                    "epsilon_physical_actual": result.epsilon_actual,
                    "target_state": result.target_state,
                }
            )
            return attacked, metadata
        if parameters.attack == "frequency_pgd":
            attacked = frequency_pgd_attack(
                self.model,
                clean,
                rho,
                epsilon=parameters.epsilon_frequency,
                num_settings=self.config.num_settings,
                outcomes_per_setting=self.config.dimension,
                steps=self.config.attack.pgd_eval_steps,
                step_size=self.config.attack.pgd_step_size,
                random_start=self.config.attack.pgd_random_start,
                generator=self.generator,
            )
            return attacked, metadata
        if parameters.attack == "combined":
            physical = physical_replacement_attack(
                rho,
                alpha=parameters.alpha,
                epsilon_physical=self.config.attack.epsilon_physical,
                kind=parameters.combined_physical_kind,
                target_state=self.config.attack.target_state,
                target_min_trace_distance=self.config.attack.target_min_trace_distance,
                generator=self.generator,
            )
            base = self.measurement.sample_frequencies(
                physical.attacked_state, shots, generator=self.generator
            )
            attacked = frequency_pgd_attack(
                self.model,
                base,
                rho,
                epsilon=parameters.epsilon_frequency,
                num_settings=self.config.num_settings,
                outcomes_per_setting=self.config.dimension,
                steps=self.config.attack.pgd_eval_steps,
                step_size=self.config.attack.pgd_step_size,
                random_start=self.config.attack.pgd_random_start,
                generator=self.generator,
            )
            metadata.update(
                {
                    "alpha_requested": physical.alpha_requested,
                    "alpha_effective": physical.alpha_effective,
                    "epsilon_physical_actual": physical.epsilon_actual,
                    "target_state": physical.target_state,
                }
            )
            return attacked, metadata
        raise ValueError(f"Unknown evaluation attack: {parameters.attack}")

    def evaluate(
        self,
        loader: DataLoader,
        parameters: AttackParameters,
        *,
        save_path: str | Path | None = None,
    ) -> tuple[pd.DataFrame, dict[str, float]]:
        rows: list[dict[str, float | int | str]] = []
        prediction_records: list[dict[str, object]] = []
        offset = 0
        for batch in tqdm(
            self._limit_loader(loader),
            desc=f"evaluate:{parameters.attack}",
            leave=False,
        ):
            rho = batch["rho"].to(self.device)
            ensemble = batch["ensemble"].cpu().numpy()
            state_purity = batch["purity"].cpu().numpy()
            sample_indices = batch["sample_index"].cpu().numpy()
            frequencies, attack_metadata = self._make_input(rho, parameters)
            with torch.no_grad():
                prediction = self.model(frequencies)
                fidelities = quantum_fidelity(
                    rho, prediction, epsilon=self.config.loss.fidelity_epsilon
                )
                trace_distances = trace_distance(rho, prediction)
                frobenius = frobenius_distance(rho, prediction)
                predicted_probabilities = self.measurement.flatten_probabilities(prediction)
                measurement_l2 = torch.linalg.vector_norm(
                    predicted_probabilities - frequencies, dim=-1
                )
                physical = physicality_metrics(prediction)
                target = attack_metadata["target_state"]
                if target is not None:
                    target_fidelity = quantum_fidelity(
                        target, prediction, epsilon=self.config.loss.fidelity_epsilon
                    )
                else:
                    target_fidelity = torch.full_like(fidelities, float("nan"))

            batch_size = rho.shape[0]
            for local in range(batch_size):
                rows.append(
                    {
                        "sample_index": int(sample_indices[local]),
                        "num_qubits": self.config.experiment.num_qubits,
                        "dimension": self.config.dimension,
                        "attack": parameters.attack,
                        "ensemble": int(ensemble[local]),
                        "purity": float(state_purity[local]),
                        "shots": int(parameters.shots or self.config.data.shots_per_setting),
                        "alpha_requested": float(
                            attack_metadata["alpha_requested"][local].detach().cpu()
                        ),
                        "alpha_effective": float(
                            attack_metadata["alpha_effective"][local].detach().cpu()
                        ),
                        "epsilon_physical_actual": float(
                            attack_metadata["epsilon_physical_actual"][local].detach().cpu()
                        ),
                        "epsilon_frequency": float(parameters.epsilon_frequency),
                        "fidelity": float(fidelities[local].detach().cpu()),
                        "infidelity": float(1.0 - fidelities[local].detach().cpu()),
                        "trace_distance": float(trace_distances[local].detach().cpu()),
                        "frobenius_distance": float(frobenius[local].detach().cpu()),
                        "measurement_l2": float(measurement_l2[local].detach().cpu()),
                        "target_fidelity": float(target_fidelity[local].detach().cpu()),
                        "minimum_eigenvalue": float(
                            physical["minimum_eigenvalue"][local].detach().cpu()
                        ),
                        "trace_error": float(
                            physical["trace_error"][local].detach().cpu()
                        ),
                        "hermitian_error": float(
                            physical["hermitian_error"][local].detach().cpu()
                        ),
                    }
                )
                if self.config.evaluation.save_predictions and len(prediction_records) < 32:
                    prediction_records.append(
                        {
                            "sample_index": int(sample_indices[local]),
                            "attack": parameters.attack,
                            "rho_true_real": rho[local].real.detach().cpu().numpy(),
                            "rho_true_imag": rho[local].imag.detach().cpu().numpy(),
                            "rho_pred_real": prediction[local].real.detach().cpu().numpy(),
                            "rho_pred_imag": prediction[local].imag.detach().cpu().numpy(),
                        }
                    )
            offset += batch_size

        frame = pd.DataFrame(rows)
        summary = {
            "num_samples": int(len(frame)),
            "mean_fidelity": float(frame["fidelity"].mean()),
            "std_fidelity": float(frame["fidelity"].std(ddof=0)),
            "median_fidelity": float(frame["fidelity"].median()),
            "mean_infidelity": float(frame["infidelity"].mean()),
            "mean_trace_distance": float(frame["trace_distance"].mean()),
            "mean_frobenius_distance": float(frame["frobenius_distance"].mean()),
            "mean_measurement_l2": float(frame["measurement_l2"].mean()),
            "minimum_predicted_eigenvalue": float(frame["minimum_eigenvalue"].min()),
            "maximum_trace_error": float(frame["trace_error"].max()),
        }
        if frame["target_fidelity"].notna().any():
            summary["mean_target_fidelity"] = float(frame["target_fidelity"].mean())
        if save_path is not None:
            save_path = Path(save_path)
            ensure_dir(save_path.parent)
            frame.to_csv(save_path, index=False)
            save_json(save_path.with_suffix(".summary.json"), summary)
            if prediction_records:
                np.savez_compressed(
                    save_path.with_suffix(".predictions.npz"),
                    records=np.array(prediction_records, dtype=object),
                )
        return frame, summary


def load_checkpoint(
    checkpoint_path: str | Path,
    model: nn.Module,
    device: torch.device,
) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    return checkpoint
