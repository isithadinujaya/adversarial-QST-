from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


def _complex_normal(
    shape: tuple[int, ...],
    *,
    device: torch.device | str,
    dtype: torch.dtype,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
    real = torch.randn(shape, device=device, dtype=real_dtype, generator=generator)
    imag = torch.randn(shape, device=device, dtype=real_dtype, generator=generator)
    return (real + 1j * imag).to(dtype)


def haar_pure_states(
    batch_size: int,
    dimension: int,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.complex64,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    vectors = _complex_normal(
        (batch_size, dimension), device=device, dtype=dtype, generator=generator
    )
    vectors = vectors / torch.linalg.vector_norm(vectors, dim=-1, keepdim=True).clamp_min(1e-12)
    return vectors.unsqueeze(-1) @ vectors.conj().unsqueeze(-2)


def ginibre_states(
    batch_size: int,
    dimension: int,
    *,
    rank: int | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.complex64,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    rank = dimension if rank is None else int(rank)
    if not (1 <= rank <= dimension):
        raise ValueError("Ginibre rank must be between one and the Hilbert-space dimension.")
    matrix = _complex_normal(
        (batch_size, dimension, rank),
        device=device,
        dtype=dtype,
        generator=generator,
    )
    rho = matrix @ matrix.conj().transpose(-1, -2)
    trace = torch.diagonal(rho, dim1=-2, dim2=-1).real.sum(dim=-1)
    return rho / trace[:, None, None].clamp_min(1e-12)


def depolarized_pure_states(
    batch_size: int,
    dimension: int,
    visibility_min: float,
    visibility_max: float,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.complex64,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    pure = haar_pure_states(
        batch_size, dimension, device=device, dtype=dtype, generator=generator
    )
    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
    visibility = torch.empty(
        batch_size, device=device, dtype=real_dtype
    ).uniform_(visibility_min, visibility_max, generator=generator)
    identity = torch.eye(dimension, device=device, dtype=dtype) / dimension
    return visibility[:, None, None] * pure + (1.0 - visibility[:, None, None]) * identity


def state_purity(rho: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bij,bji->b", rho, rho).real


@dataclass(frozen=True)
class StateMixture:
    pure_fraction: float
    mixed_fraction: float
    depolarized_fraction: float
    visibility_min: float
    visibility_max: float
    ginibre_rank: int | None = None


def sample_density_matrices(
    batch_size: int,
    dimension: int,
    mixture: StateMixture,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.complex64,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
    uniforms = torch.rand(batch_size, device=device, dtype=real_dtype, generator=generator)
    pure_cut = mixture.pure_fraction
    mixed_cut = pure_cut + mixture.mixed_fraction
    labels = torch.empty(batch_size, dtype=torch.long, device=device)
    labels[uniforms < pure_cut] = 0
    labels[(uniforms >= pure_cut) & (uniforms < mixed_cut)] = 1
    labels[uniforms >= mixed_cut] = 2

    states = torch.empty(
        (batch_size, dimension, dimension), device=device, dtype=dtype
    )
    for label in (0, 1, 2):
        mask = labels == label
        count = int(mask.sum().item())
        if count == 0:
            continue
        if label == 0:
            generated = haar_pure_states(
                count, dimension, device=device, dtype=dtype, generator=generator
            )
        elif label == 1:
            generated = ginibre_states(
                count,
                dimension,
                rank=mixture.ginibre_rank,
                device=device,
                dtype=dtype,
                generator=generator,
            )
        else:
            generated = depolarized_pure_states(
                count,
                dimension,
                mixture.visibility_min,
                mixture.visibility_max,
                device=device,
                dtype=dtype,
                generator=generator,
            )
        states[mask] = generated
    return states, labels


class DensityMatrixDataset(Dataset):
    def __init__(
        self,
        num_states: int,
        dimension: int,
        mixture: StateMixture,
        seed: int,
        chunk_size: int = 4096,
    ) -> None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        state_chunks: list[torch.Tensor] = []
        label_chunks: list[torch.Tensor] = []
        remaining = num_states
        while remaining > 0:
            current = min(chunk_size, remaining)
            states, labels = sample_density_matrices(
                current,
                dimension,
                mixture,
                device="cpu",
                dtype=torch.complex64,
                generator=generator,
            )
            state_chunks.append(states)
            label_chunks.append(labels)
            remaining -= current
        self.states = torch.cat(state_chunks, dim=0)
        self.labels = torch.cat(label_chunks, dim=0)
        self.purities = state_purity(self.states).to(torch.float32)

    def __len__(self) -> int:
        return self.states.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "rho": self.states[index],
            "ensemble": self.labels[index],
            "purity": self.purities[index],
            "sample_index": torch.tensor(index, dtype=torch.long),
        }
