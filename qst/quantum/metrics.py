from __future__ import annotations

import torch


def frobenius_distance(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(
        torch.sum(torch.abs(prediction - target) ** 2, dim=(-2, -1)).clamp_min(0.0)
    )


def trace_distance(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    difference = 0.5 * ((prediction - target) + (prediction - target).conj().transpose(-2, -1))
    eigenvalues = torch.linalg.eigvalsh(difference)
    return 0.5 * torch.sum(torch.abs(eigenvalues), dim=-1)


def _matrix_square_root_psd(matrix: torch.Tensor) -> torch.Tensor:
    hermitian = 0.5 * (matrix + matrix.conj().transpose(-2, -1))
    eigenvalues, eigenvectors = torch.linalg.eigh(hermitian)
    eigenvalues = eigenvalues.clamp_min(0.0)
    sqrt_diagonal = torch.diag_embed(torch.sqrt(eigenvalues)).to(matrix.dtype)
    return eigenvectors @ sqrt_diagonal @ eigenvectors.conj().transpose(-2, -1)


def fidelity(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Squared Uhlmann fidelity in [0, 1]."""
    sqrt_target = _matrix_square_root_psd(target)
    middle = sqrt_target @ prediction @ sqrt_target
    middle = 0.5 * (middle + middle.conj().transpose(-2, -1))
    eigenvalues = torch.linalg.eigvalsh(middle).clamp_min(0.0)
    value = torch.sum(torch.sqrt(eigenvalues), dim=-1) ** 2
    return value.real.clamp(0.0, 1.0)


def density_diagnostics(density: torch.Tensor) -> dict[str, torch.Tensor]:
    hermiticity_error = torch.amax(
        torch.abs(density - density.conj().transpose(-2, -1)), dim=(-2, -1)
    )
    trace_error = torch.abs(
        torch.diagonal(density, dim1=-2, dim2=-1).sum(dim=-1).real - 1.0
    )
    minimum_eigenvalue = torch.linalg.eigvalsh(
        0.5 * (density + density.conj().transpose(-2, -1))
    ).amin(dim=-1)
    return {
        "hermiticity_error": hermiticity_error,
        "trace_error": trace_error,
        "minimum_eigenvalue": minimum_eigenvalue,
    }
