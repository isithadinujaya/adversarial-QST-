from __future__ import annotations

import torch


def pauli_operators(
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.complex64,
) -> torch.Tensor:
    """Return Pauli X, Y, Z with shape [3,2,2]."""
    x = torch.tensor([[0, 1], [1, 0]], device=device, dtype=dtype)
    y = torch.tensor([[0, -1j], [1j, 0]], device=device, dtype=dtype)
    z = torch.tensor([[1, 0], [0, -1]], device=device, dtype=dtype)
    return torch.stack((x, y, z), dim=0)


def pauli_projectors(
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.complex64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return positive and negative projectors for X,Y,Z."""
    identity = torch.eye(2, device=device, dtype=dtype)
    paulis = pauli_operators(device=device, dtype=dtype)
    plus = 0.5 * (identity.unsqueeze(0) + paulis)
    minus = 0.5 * (identity.unsqueeze(0) - paulis)
    return plus, minus


def pauli_plus_probabilities(rho: torch.Tensor) -> torch.Tensor:
    """Return p(X+), p(Y+), p(Z+) for rho with shape [...,2,2]."""
    if rho.shape[-2:] != (2, 2):
        raise ValueError("Pauli measurement implementation expects qubit states.")

    original_shape = rho.shape[:-2]
    flattened = rho.reshape(-1, 2, 2)
    plus, _ = pauli_projectors(device=rho.device, dtype=rho.dtype)
    probabilities = torch.einsum("bij,kji->bk", flattened, plus).real
    probabilities = probabilities.clamp(0.0, 1.0)
    return probabilities.reshape(*original_shape, 3)


def plus_probabilities_to_frequency_vector(plus: torch.Tensor) -> torch.Tensor:
    """Map [...,3] positive probabilities to [...,6] interleaved +/- pairs."""
    if plus.shape[-1] != 3:
        raise ValueError("Expected final dimension 3 for X+,Y+,Z+.")
    return torch.stack((plus, 1.0 - plus), dim=-1).reshape(*plus.shape[:-1], 6)


def frequency_vector_to_plus(frequencies: torch.Tensor) -> torch.Tensor:
    """Extract X+,Y+,Z+ from [...,6] interleaved Pauli frequencies."""
    if frequencies.shape[-1] != 6:
        raise ValueError("Expected a six-dimensional Pauli frequency vector.")
    return frequencies[..., (0, 2, 4)]


@torch.no_grad()
def sample_pauli_frequencies(
    rho: torch.Tensor,
    *,
    shots_per_basis: int = 1_000,
) -> torch.Tensor:
    """Sample 1000 (configurable) independent shots in each Pauli basis."""
    if shots_per_basis <= 0:
        raise ValueError("shots_per_basis must be positive.")
    plus_probabilities = pauli_plus_probabilities(rho)
    distribution = torch.distributions.Binomial(
        total_count=float(shots_per_basis),
        probs=plus_probabilities,
    )
    plus_counts = distribution.sample()
    plus_frequencies = plus_counts / float(shots_per_basis)
    return plus_probabilities_to_frequency_vector(plus_frequencies)


def validate_pauli_frequency_vector(
    frequencies: torch.Tensor,
    *,
    atol: float = 1e-6,
) -> None:
    """Raise ValueError unless each (+,-) pair is nonnegative and sums to one."""
    if frequencies.shape[-1] != 6:
        raise ValueError("Expected final dimension 6.")
    if frequencies.amin().item() < -atol or frequencies.amax().item() > 1.0 + atol:
        raise ValueError("Frequencies must lie in [0,1].")
    pairs = frequencies.reshape(*frequencies.shape[:-1], 3, 2)
    pair_sums = pairs.sum(dim=-1)
    if not torch.allclose(pair_sums, torch.ones_like(pair_sums), atol=atol, rtol=0.0):
        raise ValueError("Each Pauli (+,-) pair must sum to one.")
