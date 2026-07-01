from __future__ import annotations

import torch
import torch.nn.functional as functional

from qst.quantum.measurements import PauliMeasurementScheme


def per_sample_reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    normalized = mode.lower()
    difference = prediction - target
    if normalized == "frobenius":
        return torch.sum(torch.abs(difference) ** 2, dim=(-2, -1))
    if normalized == "trace_distance":
        hermitian = 0.5 * (difference + difference.conj().transpose(-2, -1))
        return 0.5 * torch.sum(torch.abs(torch.linalg.eigvalsh(hermitian)), dim=-1)
    if normalized == "hybrid":
        frobenius = torch.sum(torch.abs(difference) ** 2, dim=(-2, -1))
        hermitian = 0.5 * (difference + difference.conj().transpose(-2, -1))
        trace = 0.5 * torch.sum(torch.abs(torch.linalg.eigvalsh(hermitian)), dim=-1)
        return frobenius + trace
    raise ValueError(f"Unknown reconstruction loss {mode!r}.")


def reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    return per_sample_reconstruction_loss(prediction, target, mode).mean()


def detection_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return functional.binary_cross_entropy_with_logits(logits, labels)


def measurement_consistency_loss(
    predicted_density: torch.Tensor,
    clean_frequencies: torch.Tensor,
    scheme: PauliMeasurementScheme,
) -> torch.Tensor:
    probabilities = scheme.probabilities_torch(predicted_density)
    observed = clean_frequencies.reshape(
        -1, scheme.number_settings, scheme.outcomes_per_setting
    )
    return functional.mse_loss(probabilities, observed)
