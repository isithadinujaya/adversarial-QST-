from __future__ import annotations

import torch
from torch import nn

from qst.metrics import frobenius_distance


def _expand_epsilon(
    epsilon: float | torch.Tensor,
    batch_size: int,
    num_settings: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if isinstance(epsilon, torch.Tensor):
        value = epsilon.to(device=device, dtype=dtype).reshape(-1)
        if value.numel() == 1:
            value = value.expand(batch_size)
        if value.numel() != batch_size:
            raise ValueError("epsilon tensor must be scalar or have one value per batch item.")
    else:
        value = torch.full((batch_size,), float(epsilon), device=device, dtype=dtype)
    return value[:, None].expand(batch_size, num_settings).reshape(-1, 1)


def _bounded_simplex_projection(
    values: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    iterations: int = 50,
) -> torch.Tensor:
    """Euclidean projection onto {x: sum x=1, lower<=x<=upper}."""
    if torch.any(lower.sum(dim=-1) > 1.0 + 1e-6):
        raise ValueError("Infeasible lower bounds for simplex projection.")
    if torch.any(upper.sum(dim=-1) < 1.0 - 1e-6):
        raise ValueError("Infeasible upper bounds for simplex projection.")

    low_tau = (values - upper).min(dim=-1, keepdim=True).values - 1.0
    high_tau = (values - lower).max(dim=-1, keepdim=True).values + 1.0
    for _ in range(iterations):
        tau = (low_tau + high_tau) / 2.0
        projected = torch.clamp(values - tau, min=lower, max=upper)
        too_large = projected.sum(dim=-1, keepdim=True) > 1.0
        low_tau = torch.where(too_large, tau, low_tau)
        high_tau = torch.where(too_large, high_tau, tau)
    tau = (low_tau + high_tau) / 2.0
    projected = torch.clamp(values - tau, min=lower, max=upper)
    residual = 1.0 - projected.sum(dim=-1, keepdim=True)
    free = ((projected > lower + 1e-7) & (projected < upper - 1e-7)).to(projected.dtype)
    free_count = free.sum(dim=-1, keepdim=True)
    projected = torch.where(
        free_count > 0,
        projected + free * residual / free_count.clamp_min(1.0),
        projected,
    )
    return torch.clamp(projected, min=lower, max=upper)


def project_linf_product_simplex(
    values: torch.Tensor,
    center: torch.Tensor,
    *,
    epsilon: float | torch.Tensor,
    num_settings: int,
    outcomes_per_setting: int,
) -> torch.Tensor:
    if values.shape != center.shape:
        raise ValueError("values and center must have the same shape.")
    batch_size = values.shape[0]
    values_blocks = values.reshape(-1, outcomes_per_setting)
    center_blocks = center.reshape(-1, outcomes_per_setting)
    epsilon_blocks = _expand_epsilon(
        epsilon,
        batch_size,
        num_settings,
        values.device,
        values.dtype,
    )
    lower = (center_blocks - epsilon_blocks).clamp(0.0, 1.0)
    upper = (center_blocks + epsilon_blocks).clamp(0.0, 1.0)
    projected = _bounded_simplex_projection(values_blocks, lower, upper)
    return projected.reshape_as(values)


def frequency_pgd_attack(
    model: nn.Module,
    clean_frequencies: torch.Tensor,
    target_rho: torch.Tensor,
    *,
    epsilon: float | torch.Tensor,
    num_settings: int,
    outcomes_per_setting: int,
    steps: int,
    step_size: float,
    random_start: bool = True,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if steps <= 0 or (
        not isinstance(epsilon, torch.Tensor) and float(epsilon) == 0.0
    ):
        return clean_frequencies.detach().clone()

    was_training = model.training
    model.eval()
    center = clean_frequencies.detach()
    if random_start:
        noise = torch.empty_like(center).uniform_(-1.0, 1.0, generator=generator)
        if isinstance(epsilon, torch.Tensor):
            scaled = noise * epsilon.to(center.device, center.dtype).reshape(-1, 1)
        else:
            scaled = noise * float(epsilon)
        adversarial = project_linf_product_simplex(
            center + scaled,
            center,
            epsilon=epsilon,
            num_settings=num_settings,
            outcomes_per_setting=outcomes_per_setting,
        )
    else:
        adversarial = center.clone()

    for _ in range(steps):
        adversarial = adversarial.detach().requires_grad_(True)
        prediction = model(adversarial)
        objective = frobenius_distance(
            target_rho,
            prediction,
        ).square().mean()
        gradient = torch.autograd.grad(objective, adversarial, only_inputs=True)[0]
        with torch.no_grad():
            adversarial = adversarial + step_size * gradient.sign()
            adversarial = project_linf_product_simplex(
                adversarial,
                center,
                epsilon=epsilon,
                num_settings=num_settings,
                outcomes_per_setting=outcomes_per_setting,
            )

    model.train(was_training)
    return adversarial.detach()
