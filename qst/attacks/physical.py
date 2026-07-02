from __future__ import annotations

from dataclasses import dataclass

import torch

from qst.metrics import trace_distance
from qst.states import StateMixture, haar_pure_states, sample_density_matrices


@dataclass
class PhysicalAttackResult:
    attacked_state: torch.Tensor
    replacement_state: torch.Tensor
    alpha_requested: torch.Tensor
    alpha_effective: torch.Tensor
    source_replacement_distance: torch.Tensor
    epsilon_actual: torch.Tensor
    target_state: torch.Tensor | None = None


def _basis_state(
    batch_size: int,
    dimension: int,
    index: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    vector = torch.zeros(batch_size, dimension, device=device, dtype=dtype)
    vector[:, index] = 1.0
    return vector.unsqueeze(-1) @ vector.conj().unsqueeze(-2)


def _ghz_state(
    batch_size: int,
    dimension: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    vector = torch.zeros(batch_size, dimension, device=device, dtype=dtype)
    vector[:, 0] = 1.0 / (2.0**0.5)
    vector[:, -1] = 1.0 / (2.0**0.5)
    return vector.unsqueeze(-1) @ vector.conj().unsqueeze(-2)


def _worst_eigenstate(rho: torch.Tensor) -> torch.Tensor:
    _, eigenvectors = torch.linalg.eigh(rho)
    vector = eigenvectors[..., 0]
    return vector.unsqueeze(-1) @ vector.conj().unsqueeze(-2)


def _random_far_target(
    rho: torch.Tensor,
    min_distance: float,
    generator: torch.Generator | None,
    attempts: int = 8,
) -> torch.Tensor:
    batch_size, dimension, _ = rho.shape
    target = haar_pure_states(
        batch_size,
        dimension,
        device=rho.device,
        dtype=rho.dtype,
        generator=generator,
    )
    distance = trace_distance(rho, target)
    for _ in range(attempts - 1):
        mask = distance < min_distance
        if not mask.any():
            break
        candidates = haar_pure_states(
            int(mask.sum().item()),
            dimension,
            device=rho.device,
            dtype=rho.dtype,
            generator=generator,
        )
        target[mask] = candidates
        distance = trace_distance(rho, target)
    fallback = distance < min_distance
    if fallback.any():
        target[fallback] = _worst_eigenstate(rho[fallback])
    return target


def _replacement_state(
    rho: torch.Tensor,
    kind: str,
    target_state: str,
    target_min_trace_distance: float,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    batch_size, dimension, _ = rho.shape
    device, dtype = rho.device, rho.dtype
    if kind == "random_replacement":
        mixture = StateMixture(0.5, 0.5, 0.0, 0.2, 1.0)
        replacement, _ = sample_density_matrices(
            batch_size,
            dimension,
            mixture,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        return replacement, None
    if kind == "fixed_replacement":
        return _basis_state(batch_size, dimension, 0, device, dtype), None
    if kind == "worst_replacement":
        return _worst_eigenstate(rho), None
    if kind == "targeted_replacement":
        if target_state == "zero":
            target = _basis_state(batch_size, dimension, 0, device, dtype)
        elif target_state == "one":
            target = _basis_state(batch_size, dimension, dimension - 1, device, dtype)
        elif target_state == "ghz":
            target = _ghz_state(batch_size, dimension, device, dtype)
        elif target_state == "random_far":
            target = _random_far_target(
                rho, target_min_trace_distance, generator=generator
            )
        else:
            raise ValueError(f"Unknown target state: {target_state}")
        return target, target
    raise ValueError(f"Unsupported physical attack kind: {kind}")


def physical_replacement_attack(
    rho: torch.Tensor,
    *,
    alpha: float | torch.Tensor,
    epsilon_physical: float,
    kind: str,
    target_state: str = "random_far",
    target_min_trace_distance: float = 0.5,
    generator: torch.Generator | None = None,
) -> PhysicalAttackResult:
    replacement, target = _replacement_state(
        rho,
        kind,
        target_state,
        target_min_trace_distance,
        generator,
    )
    distance = trace_distance(rho, replacement)
    if isinstance(alpha, torch.Tensor):
        alpha_requested = alpha.to(device=rho.device, dtype=distance.dtype).reshape(-1)
        if alpha_requested.numel() == 1:
            alpha_requested = alpha_requested.expand(rho.shape[0])
    else:
        alpha_requested = torch.full_like(distance, float(alpha))
    alpha_requested = alpha_requested.clamp(0.0, 1.0)

    if epsilon_physical < 0.0:
        raise ValueError("epsilon_physical must be nonnegative.")
    cap = torch.where(
        distance > 1.0e-12,
        torch.full_like(distance, epsilon_physical) / distance.clamp_min(1.0e-12),
        torch.ones_like(distance),
    )
    alpha_effective = torch.minimum(alpha_requested, cap).clamp(0.0, 1.0)
    attacked = (
        (1.0 - alpha_effective)[:, None, None] * rho
        + alpha_effective[:, None, None] * replacement
    )
    epsilon_actual = trace_distance(rho, attacked)
    return PhysicalAttackResult(
        attacked_state=attacked,
        replacement_state=replacement,
        alpha_requested=alpha_requested,
        alpha_effective=alpha_effective,
        source_replacement_distance=distance,
        epsilon_actual=epsilon_actual,
        target_state=target,
    )
