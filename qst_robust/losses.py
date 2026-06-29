from __future__ import annotations

import torch


def frobenius_squared_per_sample(
    estimate: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    difference = estimate - target
    return difference.abs().square().sum(dim=(-2, -1))


def qubit_fidelity(
    rho: torch.Tensor,
    sigma: torch.Tensor,
    *,
    numerical_epsilon: float = 1e-12,
) -> torch.Tensor:
    """
    Squared Uhlmann fidelity for 2x2 density matrices:
    F(rho,sigma) = Tr(rho sigma) + 2 sqrt(det(rho) det(sigma)).
    """
    if rho.shape[-2:] != (2, 2) or sigma.shape[-2:] != (2, 2):
        raise ValueError("This stable closed form is specific to qubits.")

    trace_product = torch.einsum("...ij,...ji->...", rho, sigma).real
    det_rho = torch.linalg.det(rho).real.clamp_min(0.0)
    det_sigma = torch.linalg.det(sigma).real.clamp_min(0.0)
    determinant_term = 2.0 * torch.sqrt(
        (det_rho * det_sigma).clamp_min(0.0) + numerical_epsilon
    )
    return (trace_product + determinant_term).clamp(0.0, 1.0)


def reconstruction_loss_per_sample(
    estimate: torch.Tensor,
    target: torch.Tensor,
    *,
    fidelity_weight: float = 0.1,
) -> torch.Tensor:
    loss = frobenius_squared_per_sample(estimate, target)
    if fidelity_weight:
        loss = loss + fidelity_weight * (1.0 - qubit_fidelity(estimate, target))
    return loss
