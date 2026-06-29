from __future__ import annotations

from typing import Tuple

import torch

COMPLEX_DTYPE = torch.complex64
REAL_DTYPE = torch.float32


def _complex_normal(
    shape: Tuple[int, ...],
    *,
    device: torch.device | str,
    dtype: torch.dtype = COMPLEX_DTYPE,
) -> torch.Tensor:
    """Return a circular complex normal tensor with E[|z|^2] = 1."""
    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
    scale = 2.0 ** -0.5
    real = torch.randn(shape, device=device, dtype=real_dtype)
    imag = torch.randn(shape, device=device, dtype=real_dtype)
    return (scale * (real + 1j * imag)).to(dtype)


def haar_pure_states(
    num_states: int,
    *,
    dimension: int = 2,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = COMPLEX_DTYPE,
) -> torch.Tensor:
    """Generate Haar-random pure-state density matrices."""
    if num_states < 0:
        raise ValueError("num_states cannot be negative.")
    if num_states == 0:
        return torch.empty((0, dimension, dimension), device=device, dtype=dtype)

    vectors = _complex_normal((num_states, dimension), device=device, dtype=dtype)
    vectors = vectors / torch.linalg.vector_norm(vectors, dim=-1, keepdim=True).clamp_min(1e-12)
    return vectors.unsqueeze(-1) @ vectors.conj().unsqueeze(-2)


def ginibre_mixed_states(
    num_states: int,
    *,
    dimension: int = 2,
    rank_parameter: int | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = COMPLEX_DTYPE,
) -> torch.Tensor:
    """Generate mixed states rho = G G^dagger / Tr(G G^dagger)."""
    if num_states < 0:
        raise ValueError("num_states cannot be negative.")
    if num_states == 0:
        return torch.empty((0, dimension, dimension), device=device, dtype=dtype)

    rank_parameter = dimension if rank_parameter is None else rank_parameter
    if rank_parameter <= 0:
        raise ValueError("rank_parameter must be positive.")

    g = _complex_normal(
        (num_states, dimension, rank_parameter),
        device=device,
        dtype=dtype,
    )
    rho = g @ g.conj().transpose(-1, -2)
    trace = rho.diagonal(dim1=-2, dim2=-1).real.sum(dim=-1, keepdim=True)
    return rho / trace.unsqueeze(-1).clamp_min(1e-12)


def depolarized_pure_states(
    num_states: int,
    *,
    dimension: int = 2,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = COMPLEX_DTYPE,
) -> torch.Tensor:
    """Generate rho = lambda |psi><psi| + (1-lambda) I/d, lambda ~ U(0,1)."""
    if num_states == 0:
        return torch.empty((0, dimension, dimension), device=device, dtype=dtype)

    pure = haar_pure_states(num_states, dimension=dimension, device=device, dtype=dtype)
    mixing = torch.rand((num_states, 1, 1), device=device, dtype=pure.real.dtype)
    identity = torch.eye(dimension, device=device, dtype=dtype) / dimension
    return mixing * pure + (1.0 - mixing) * identity


def random_pure_or_mixed_states(
    num_states: int,
    *,
    pure_probability: float = 0.5,
    dimension: int = 2,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = COMPLEX_DTYPE,
) -> torch.Tensor:
    """Generate an independent Bernoulli mixture of Haar-pure and Ginibre-mixed states."""
    if not 0.0 <= pure_probability <= 1.0:
        raise ValueError("pure_probability must lie in [0,1].")
    if num_states == 0:
        return torch.empty((0, dimension, dimension), device=device, dtype=dtype)

    choose_pure = torch.rand(num_states, device=device) < pure_probability
    result = torch.empty((num_states, dimension, dimension), device=device, dtype=dtype)

    num_pure = int(choose_pure.sum().item())
    num_mixed = num_states - num_pure
    if num_pure:
        result[choose_pure] = haar_pure_states(
            num_pure, dimension=dimension, device=device, dtype=dtype
        )
    if num_mixed:
        result[~choose_pure] = ginibre_mixed_states(
            num_mixed, dimension=dimension, device=device, dtype=dtype
        )
    return result


def generate_state_ensemble(
    num_states: int,
    *,
    pure_fraction: float = 0.25,
    ginibre_fraction: float = 0.50,
    depolarized_fraction: float = 0.25,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = COMPLEX_DTYPE,
) -> torch.Tensor:
    """Generate and shuffle the agreed pure/Ginibre/depolarized state mixture."""
    fractions = pure_fraction + ginibre_fraction + depolarized_fraction
    if abs(fractions - 1.0) > 1e-8:
        raise ValueError("State fractions must sum to 1.")

    num_pure = round(num_states * pure_fraction)
    num_ginibre = round(num_states * ginibre_fraction)
    num_depolarized = num_states - num_pure - num_ginibre

    states = torch.cat(
        [
            haar_pure_states(num_pure, device=device, dtype=dtype),
            ginibre_mixed_states(num_ginibre, device=device, dtype=dtype),
            depolarized_pure_states(num_depolarized, device=device, dtype=dtype),
        ],
        dim=0,
    )
    if num_states:
        states = states[torch.randperm(num_states, device=device)]
    return states


def trace_distance(rho: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Batch trace distance 0.5 ||rho-sigma||_1 for Hermitian matrices."""
    difference = rho - sigma
    eigenvalues = torch.linalg.eigvalsh(difference)
    return 0.5 * eigenvalues.abs().sum(dim=-1)


@torch.no_grad()
def sample_distant_target_states(
    rho: torch.Tensor,
    *,
    minimum_trace_distance: float = 0.5,
    max_attempts: int = 64,
    pure_probability: float = 0.5,
) -> torch.Tensor:
    """
    Sample one pure/mixed target per rho, requiring the requested trace distance.

    If rejection sampling does not succeed, the fallback is the projector onto
    rho's minimum-eigenvalue eigenvector. For a qubit this target is at trace
    distance equal to rho's largest eigenvalue, which is at least 0.5.
    """
    if rho.ndim != 3 or rho.shape[-2:] != (2, 2):
        raise ValueError("rho must have shape [batch, 2, 2].")
    if not 0.0 <= minimum_trace_distance <= 1.0:
        raise ValueError("minimum_trace_distance must lie in [0,1].")

    batch = rho.shape[0]
    targets = torch.empty_like(rho)
    unresolved = torch.ones(batch, dtype=torch.bool, device=rho.device)
    tolerance = 1e-6

    for _ in range(max_attempts):
        indices = unresolved.nonzero(as_tuple=False).squeeze(-1)
        if indices.numel() == 0:
            break
        candidates = random_pure_or_mixed_states(
            indices.numel(),
            pure_probability=pure_probability,
            device=rho.device,
            dtype=rho.dtype,
        )
        distances = trace_distance(rho[indices], candidates)
        accepted = distances >= minimum_trace_distance - tolerance
        if accepted.any():
            accepted_indices = indices[accepted]
            targets[accepted_indices] = candidates[accepted]
            unresolved[accepted_indices] = False

    if unresolved.any():
        indices = unresolved.nonzero(as_tuple=False).squeeze(-1)
        _, eigenvectors = torch.linalg.eigh(rho[indices])
        minimum_vectors = eigenvectors[..., :, 0]
        fallback = minimum_vectors.unsqueeze(-1) @ minimum_vectors.conj().unsqueeze(-2)
        targets[indices] = fallback

    return targets


def validate_density_matrices(
    rho: torch.Tensor,
    *,
    atol: float = 1e-5,
) -> None:
    """Raise ValueError if a batch contains nonphysical density matrices."""
    if rho.ndim != 3 or rho.shape[-1] != rho.shape[-2]:
        raise ValueError("Expected rho with shape [batch, d, d].")
    hermitian_error = (rho - rho.conj().transpose(-1, -2)).abs().amax().item()
    traces = rho.diagonal(dim1=-2, dim2=-1).real.sum(dim=-1)
    minimum_eigenvalue = torch.linalg.eigvalsh(rho).amin().item()
    if hermitian_error > atol:
        raise ValueError(f"Hermiticity error {hermitian_error} exceeds tolerance.")
    if not torch.allclose(traces, torch.ones_like(traces), atol=atol, rtol=0.0):
        raise ValueError("At least one density matrix does not have trace one.")
    if minimum_eigenvalue < -atol:
        raise ValueError(f"Minimum eigenvalue {minimum_eigenvalue} is negative.")
