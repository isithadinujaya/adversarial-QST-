from __future__ import annotations

import torch
import torch.nn.functional as F


def raw_cholesky_to_density(
    raw: torch.Tensor,
    dimension: int,
    *,
    diagonal_floor: float = 1.0e-4,
) -> torch.Tensor:
    expected = dimension * dimension
    if raw.shape[-1] != expected:
        raise ValueError(f"Expected {expected} real outputs, received {raw.shape[-1]}.")

    complex_dtype = torch.complex128 if raw.dtype == torch.float64 else torch.complex64
    batch_shape = raw.shape[:-1]
    lower = torch.zeros(
        (*batch_shape, dimension, dimension),
        device=raw.device,
        dtype=complex_dtype,
    )

    diagonal = F.softplus(raw[..., :dimension]) + diagonal_floor
    indices = torch.arange(dimension, device=raw.device)
    lower[..., indices, indices] = diagonal.to(complex_dtype)

    cursor = dimension
    for row in range(1, dimension):
        for column in range(row):
            real = raw[..., cursor]
            imag = raw[..., cursor + 1]
            lower[..., row, column] = torch.complex(real, imag)
            cursor += 2

    unnormalized = lower @ lower.conj().transpose(-1, -2)
    trace = torch.diagonal(unnormalized, dim1=-2, dim2=-1).real.sum(dim=-1)
    return unnormalized / trace[..., None, None].clamp_min(1.0e-12)
