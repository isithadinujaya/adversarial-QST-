from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import nn


class CholeskyDensityHead(nn.Module):
    """Map real features to a valid d x d complex density matrix."""

    def __init__(self, feature_dimension: int, hilbert_dimension: int) -> None:
        super().__init__()
        self.hilbert_dimension = hilbert_dimension
        self.parameter_dimension = hilbert_dimension ** 2
        self.linear = nn.Linear(feature_dimension, self.parameter_dimension)

    def raw_to_density(self, raw: torch.Tensor) -> torch.Tensor:
        d = self.hilbert_dimension
        if raw.shape[-1] != d ** 2:
            raise ValueError(f"Expected {d ** 2} real Cholesky parameters.")
        complex_dtype = torch.complex128 if raw.dtype == torch.float64 else torch.complex64
        matrix = torch.zeros(
            (*raw.shape[:-1], d, d),
            dtype=complex_dtype,
            device=raw.device,
        )

        index = 0
        diagonal = functional.softplus(raw[..., :d]) + torch.finfo(raw.dtype).eps
        for row in range(d):
            matrix[..., row, row] = diagonal[..., row].to(complex_dtype)
        index = d

        for row in range(1, d):
            for column in range(row):
                real_part = raw[..., index]
                imaginary_part = raw[..., index + 1]
                matrix[..., row, column] = torch.complex(real_part, imaginary_part)
                index += 2

        if index != d ** 2:
            raise RuntimeError("Internal Cholesky parameter indexing error.")

        unnormalized = matrix @ matrix.conj().transpose(-2, -1)
        trace = torch.diagonal(unnormalized, dim1=-2, dim2=-1).real.sum(dim=-1)
        trace = trace.clamp_min(torch.finfo(raw.dtype).eps)
        return unnormalized / trace[..., None, None]

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.raw_to_density(self.linear(features))
