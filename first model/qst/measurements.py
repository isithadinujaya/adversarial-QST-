from __future__ import annotations

import itertools

import torch


PAULI_LABELS = ("X", "Y", "Z")


def _kron_all(matrices: list[torch.Tensor]) -> torch.Tensor:
    result = matrices[0]
    for matrix in matrices[1:]:
        result = torch.kron(result, matrix)
    return result


def _single_qubit_projectors(dtype: torch.dtype) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    identity = torch.eye(2, dtype=dtype)
    x = torch.tensor([[0, 1], [1, 0]], dtype=dtype)
    y = torch.tensor([[0, -1j], [1j, 0]], dtype=dtype)
    z = torch.tensor([[1, 0], [0, -1]], dtype=dtype)
    paulis = {"X": x, "Y": y, "Z": z}
    output: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for label, matrix in paulis.items():
        output[label] = ((identity + matrix) / 2, (identity - matrix) / 2)
    return output


class PauliCubeMeasurement:
    """All local Pauli settings with one normalized frequency block per setting."""

    def __init__(
        self,
        num_qubits: int,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.complex64,
    ) -> None:
        if num_qubits not in (1, 2, 3):
            raise ValueError("This implementation supports one to three qubits.")
        self.num_qubits = num_qubits
        self.dimension = 2**num_qubits
        self.setting_labels = [
            "".join(setting)
            for setting in itertools.product(PAULI_LABELS, repeat=num_qubits)
        ]
        self.outcome_labels = [
            "".join(map(str, outcome))
            for outcome in itertools.product((0, 1), repeat=num_qubits)
        ]
        local = _single_qubit_projectors(dtype)
        projectors: list[torch.Tensor] = []
        for setting in itertools.product(PAULI_LABELS, repeat=num_qubits):
            setting_projectors: list[torch.Tensor] = []
            for outcome in itertools.product((0, 1), repeat=num_qubits):
                factors = [local[axis][bit] for axis, bit in zip(setting, outcome)]
                setting_projectors.append(_kron_all(factors))
            projectors.append(torch.stack(setting_projectors, dim=0))
        self.projectors = torch.stack(projectors, dim=0).to(device=device)

    @property
    def num_settings(self) -> int:
        return len(self.setting_labels)

    @property
    def input_dimension(self) -> int:
        return self.num_settings * self.dimension

    def to(self, device: torch.device | str) -> "PauliCubeMeasurement":
        self.projectors = self.projectors.to(device)
        return self

    def probabilities(self, rho: torch.Tensor) -> torch.Tensor:
        if rho.shape[-2:] != (self.dimension, self.dimension):
            raise ValueError("Density-matrix dimension does not match the measurement.")
        flat_projectors = self.projectors.reshape(
            self.input_dimension, self.dimension, self.dimension
        )
        probabilities = torch.einsum(
            "bij,kji->bk", rho, flat_projectors
        ).real
        probabilities = probabilities.reshape(
            rho.shape[0], self.num_settings, self.dimension
        )
        probabilities = probabilities.clamp_min(0.0)
        probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return probabilities

    @staticmethod
    def _multinomial_counts(
        probabilities: torch.Tensor,
        shots: int,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if shots <= 0:
            raise ValueError("shots must be positive.")
        original_shape = probabilities.shape
        categories = original_shape[-1]
        flat = probabilities.reshape(-1, categories)
        counts = torch.zeros_like(flat)
        remaining_count = torch.full(
            (flat.shape[0],),
            float(shots),
            device=flat.device,
            dtype=flat.dtype,
        )
        remaining_probability = torch.ones_like(remaining_count)
        for category in range(categories - 1):
            conditional = (
                flat[:, category] / remaining_probability.clamp_min(1e-12)
            ).clamp(0.0, 1.0)
            draw = torch.binomial(remaining_count, conditional, generator=generator)
            counts[:, category] = draw
            remaining_count = remaining_count - draw
            remaining_probability = remaining_probability - flat[:, category]
        counts[:, -1] = remaining_count
        return counts.reshape(original_shape)

    def sample_frequencies(
        self,
        rho: torch.Tensor,
        shots: int,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        probabilities = self.probabilities(rho)
        counts = self._multinomial_counts(probabilities, shots, generator=generator)
        return (counts / float(shots)).reshape(rho.shape[0], self.input_dimension)

    def flatten_probabilities(self, rho: torch.Tensor) -> torch.Tensor:
        return self.probabilities(rho).reshape(rho.shape[0], self.input_dimension)

    def reshape_frequency_vector(self, frequencies: torch.Tensor) -> torch.Tensor:
        if frequencies.shape[-1] != self.input_dimension:
            raise ValueError(
                f"Expected frequency dimension {self.input_dimension}, got {frequencies.shape[-1]}."
            )
        return frequencies.reshape(-1, self.num_settings, self.dimension)
