from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as functional


def _project_bounded_simplex_torch(
    values: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    iterations: int = 40,
) -> torch.Tensor:
    """Euclidean projection onto sum(x)=1 with elementwise lower/upper bounds."""
    low_tau = torch.amin(values - upper, dim=-1, keepdim=True)
    high_tau = torch.amax(values - lower, dim=-1, keepdim=True)
    for _ in range(iterations):
        tau = 0.5 * (low_tau + high_tau)
        projected = torch.clamp(values - tau, min=lower, max=upper)
        sums = projected.sum(dim=-1, keepdim=True)
        low_tau = torch.where(sums > 1.0, tau, low_tau)
        high_tau = torch.where(sums > 1.0, high_tau, tau)
    tau = 0.5 * (low_tau + high_tau)
    return torch.clamp(values - tau, min=lower, max=upper)


def project_frequency_ball(
    values: torch.Tensor,
    original: torch.Tensor,
    epsilon: float,
    number_settings: int,
    outcomes_per_setting: int,
) -> torch.Tensor:
    shaped_values = values.reshape(-1, number_settings, outcomes_per_setting)
    shaped_original = original.reshape(-1, number_settings, outcomes_per_setting)
    lower = torch.clamp(shaped_original - epsilon, min=0.0)
    upper = torch.clamp(shaped_original + epsilon, max=1.0)
    projected = _project_bounded_simplex_torch(shaped_values, lower, upper)
    return projected.reshape_as(values)


def _project_bounded_simplex_numpy(
    values: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    iterations: int = 50,
) -> np.ndarray:
    low_tau = np.min(values - upper)
    high_tau = np.max(values - lower)
    for _ in range(iterations):
        tau = 0.5 * (low_tau + high_tau)
        projected = np.clip(values - tau, lower, upper)
        if projected.sum() > 1.0:
            low_tau = tau
        else:
            high_tau = tau
    return np.clip(values - 0.5 * (low_tau + high_tau), lower, upper)


def random_frequency_attack_numpy(
    clean_frequencies: np.ndarray,
    epsilon: float,
    number_settings: int,
    outcomes_per_setting: int,
    rng: np.random.Generator,
) -> np.ndarray:
    original = clean_frequencies.reshape(number_settings, outcomes_per_setting)
    attacked = np.empty_like(original)
    for index, row in enumerate(original):
        perturbation = rng.uniform(-epsilon, epsilon, size=row.shape)
        candidate = row + perturbation
        lower = np.clip(row - epsilon, 0.0, None)
        upper = np.clip(row + epsilon, None, 1.0)
        attacked[index] = _project_bounded_simplex_numpy(candidate, lower, upper)
    return attacked.astype(np.float32).reshape(-1)


def _per_sample_frobenius_squared(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    return torch.sum(torch.abs(prediction - target) ** 2, dim=(-2, -1))


def pgd_frequency_attack(
    model: torch.nn.Module,
    clean_frequencies: torch.Tensor,
    target_density: torch.Tensor,
    *,
    epsilon: float,
    step_size: float,
    steps: int,
    number_settings: int,
    outcomes_per_setting: int,
    random_start: bool,
    detection_evasion_weight: float,
) -> torch.Tensor:
    """Untargeted PGD that degrades reconstruction and optionally attempts detection evasion."""
    original = clean_frequencies.detach()
    was_training = model.training
    model.eval()

    if random_start:
        candidate = original + torch.empty_like(original).uniform_(-epsilon, epsilon)
        candidate = project_frequency_ball(
            candidate,
            original,
            epsilon,
            number_settings,
            outcomes_per_setting,
        )
    else:
        candidate = original.clone()

    with torch.enable_grad():
        for _ in range(steps):
            candidate = candidate.detach().requires_grad_(True)
            prediction, attack_logit = model(candidate)
            reconstruction_objective = _per_sample_frobenius_squared(
                prediction, target_density
            ).mean()
            if detection_evasion_weight > 0:
                clean_labels = torch.zeros_like(attack_logit)
                detection_as_clean_loss = functional.binary_cross_entropy_with_logits(
                    attack_logit,
                    clean_labels,
                )
                objective = reconstruction_objective - (
                    detection_evasion_weight * detection_as_clean_loss
                )
            else:
                objective = reconstruction_objective
            gradient = torch.autograd.grad(objective, candidate, only_inputs=True)[0]
            candidate = candidate + step_size * gradient.sign()
            candidate = project_frequency_ball(
                candidate,
                original,
                epsilon,
                number_settings,
                outcomes_per_setting,
            )

    model.train(was_training)
    return candidate.detach()
