from __future__ import annotations

from itertools import product

import numpy as np
import torch


_SINGLE_QUBIT_STATES = {
    "X": (
        np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2),
        np.array([1.0, -1.0], dtype=np.complex128) / np.sqrt(2),
    ),
    "Y": (
        np.array([1.0, 1.0j], dtype=np.complex128) / np.sqrt(2),
        np.array([1.0, -1.0j], dtype=np.complex128) / np.sqrt(2),
    ),
    "Z": (
        np.array([1.0, 0.0], dtype=np.complex128),
        np.array([0.0, 1.0], dtype=np.complex128),
    ),
}


def _kron_all(vectors: tuple[np.ndarray, ...]) -> np.ndarray:
    result = vectors[0]
    for vector in vectors[1:]:
        result = np.kron(result, vector)
    return result


class PauliMeasurementScheme:
    """Full local Pauli X/Y/Z projective measurements for n qubits."""

    def __init__(self, number_qubits: int):
        if number_qubits < 1:
            raise ValueError("number_qubits must be positive.")
        self.number_qubits = number_qubits
        self.dimension = 2 ** number_qubits
        self.settings = tuple("".join(items) for items in product("XYZ", repeat=number_qubits))
        self.outcomes = tuple(product((0, 1), repeat=number_qubits))
        self.number_settings = len(self.settings)
        self.outcomes_per_setting = len(self.outcomes)
        self.input_dimension = self.number_settings * self.outcomes_per_setting
        self.projectors = self._build_projectors()

    def _build_projectors(self) -> np.ndarray:
        projectors = np.empty(
            (
                self.number_settings,
                self.outcomes_per_setting,
                self.dimension,
                self.dimension,
            ),
            dtype=np.complex128,
        )
        for setting_index, setting in enumerate(self.settings):
            for outcome_index, outcome in enumerate(self.outcomes):
                vectors = tuple(
                    _SINGLE_QUBIT_STATES[axis][bit]
                    for axis, bit in zip(setting, outcome, strict=True)
                )
                vector = _kron_all(vectors)
                projectors[setting_index, outcome_index] = np.outer(vector, vector.conj())
        return projectors

    def probabilities_numpy(self, density: np.ndarray) -> np.ndarray:
        if density.shape != (self.dimension, self.dimension):
            raise ValueError(
                f"Expected density shape {(self.dimension, self.dimension)}, received {density.shape}."
            )
        probabilities = np.einsum("soij,ji->so", self.projectors, density).real
        probabilities = np.clip(probabilities, 0.0, None)
        totals = probabilities.sum(axis=-1, keepdims=True)
        if np.any(totals <= 0):
            raise RuntimeError("A measurement setting produced zero total probability.")
        return probabilities / totals

    def probabilities_torch(self, density: torch.Tensor) -> torch.Tensor:
        if density.shape[-2:] != (self.dimension, self.dimension):
            raise ValueError(
                f"Expected final density dimensions {(self.dimension, self.dimension)}, "
                f"received {density.shape[-2:]}."
            )
        projectors = torch.as_tensor(
            self.projectors,
            dtype=density.dtype,
            device=density.device,
        )
        probabilities = torch.einsum("soij,...ji->...so", projectors, density).real
        probabilities = probabilities.clamp_min(0.0)
        return probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    def sample_frequencies(
        self,
        density: np.ndarray,
        copies_per_setting: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        probabilities = self.probabilities_numpy(density)
        counts = np.stack(
            [rng.multinomial(copies_per_setting, row) for row in probabilities],
            axis=0,
        )
        return (counts / copies_per_setting).astype(np.float32).reshape(-1)

    def sample_replacement_frequencies(
        self,
        clean_density: np.ndarray,
        replacement_density: np.ndarray,
        replacement_fraction: float,
        copies_per_setting: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Simulate exactly K replacement copies per setting rather than only mixing probabilities."""
        replacement_copies = int(round(replacement_fraction * copies_per_setting))
        replacement_copies = min(max(replacement_copies, 0), copies_per_setting)
        clean_copies = copies_per_setting - replacement_copies

        clean_probabilities = self.probabilities_numpy(clean_density)
        replacement_probabilities = self.probabilities_numpy(replacement_density)
        all_counts = []
        for clean_row, replacement_row in zip(
            clean_probabilities, replacement_probabilities, strict=True
        ):
            clean_counts = rng.multinomial(clean_copies, clean_row)
            replacement_counts = rng.multinomial(replacement_copies, replacement_row)
            all_counts.append(clean_counts + replacement_counts)
        counts = np.stack(all_counts, axis=0)
        return (counts / copies_per_setting).astype(np.float32).reshape(-1)

    def reshape_frequencies_numpy(self, frequencies: np.ndarray) -> np.ndarray:
        return frequencies.reshape(self.number_settings, self.outcomes_per_setting)

    def reshape_frequencies_torch(self, frequencies: torch.Tensor) -> torch.Tensor:
        return frequencies.reshape(
            *frequencies.shape[:-1], self.number_settings, self.outcomes_per_setting
        )
