from __future__ import annotations

import torch


def hermitian_part(matrix: torch.Tensor) -> torch.Tensor:
    return (matrix + matrix.conj().transpose(-1, -2)) / 2


def matrix_sqrt_psd(matrix: torch.Tensor, epsilon: float = 0.0) -> torch.Tensor:
    matrix = hermitian_part(matrix)
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    eigenvalues = eigenvalues.real.clamp_min(epsilon)
    root = torch.sqrt(eigenvalues)
    return (eigenvectors * root.unsqueeze(-2)) @ eigenvectors.conj().transpose(-1, -2)


def quantum_fidelity(
    rho: torch.Tensor,
    sigma: torch.Tensor,
    *,
    epsilon: float = 1.0e-9,
) -> torch.Tensor:
    """Squared Uhlmann fidelity in [0,1]."""
    sqrt_rho = matrix_sqrt_psd(rho, epsilon=0.0)
    middle = hermitian_part(sqrt_rho @ sigma @ sqrt_rho)
    eigenvalues = torch.linalg.eigvalsh(middle).real.clamp_min(epsilon)
    root_trace = torch.sqrt(eigenvalues).sum(dim=-1)
    return root_trace.square().clamp(0.0, 1.0)


def infidelity(
    rho: torch.Tensor,
    sigma: torch.Tensor,
    *,
    epsilon: float = 1.0e-9,
) -> torch.Tensor:
    return 1.0 - quantum_fidelity(rho, sigma, epsilon=epsilon)


def trace_distance(rho: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    difference = hermitian_part(rho - sigma)
    singular_values = torch.linalg.eigvalsh(difference).real.abs()
    return 0.5 * singular_values.sum(dim=-1)


def frobenius_distance(rho: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    difference = rho - sigma
    return torch.sqrt((difference.conj() * difference).real.sum(dim=(-2, -1)).clamp_min(0.0))


def purity(rho: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bij,bji->b", rho, rho).real


def physicality_metrics(rho: torch.Tensor) -> dict[str, torch.Tensor]:
    eigenvalues = torch.linalg.eigvalsh(hermitian_part(rho)).real
    trace = torch.diagonal(rho, dim1=-2, dim2=-1).sum(dim=-1)
    hermitian_error = frobenius_distance(rho, hermitian_part(rho))
    return {
        "minimum_eigenvalue": eigenvalues.min(dim=-1).values,
        "trace_error": (trace.real - 1.0).abs(),
        "trace_imaginary": trace.imag.abs(),
        "hermitian_error": hermitian_error,
    }
