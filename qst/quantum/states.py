from __future__ import annotations

import numpy as np


def _normalize_density(matrix: np.ndarray) -> np.ndarray:
    matrix = 0.5 * (matrix + matrix.conj().T)
    trace = np.trace(matrix).real
    if trace <= 0:
        raise ValueError("Cannot normalize a matrix with non-positive trace.")
    return matrix / trace


def haar_pure_density(dimension: int, rng: np.random.Generator) -> np.ndarray:
    vector = rng.normal(size=dimension) + 1j * rng.normal(size=dimension)
    norm = np.linalg.norm(vector)
    if norm == 0:
        vector[0] = 1.0
        norm = 1.0
    vector = vector / norm
    return np.outer(vector, vector.conj()).astype(np.complex128)


def ginibre_mixed_density(dimension: int, rng: np.random.Generator) -> np.ndarray:
    matrix = rng.normal(size=(dimension, dimension)) + 1j * rng.normal(
        size=(dimension, dimension)
    )
    density = matrix @ matrix.conj().T
    return _normalize_density(density).astype(np.complex128)


def generate_density_matrix(
    dimension: int,
    pure_probability: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, str]:
    if rng.random() < pure_probability:
        return haar_pure_density(dimension, rng), "pure"
    return ginibre_mixed_density(dimension, rng), "mixed"


def computational_basis_density(dimension: int, index: int) -> np.ndarray:
    if not 0 <= index < dimension:
        raise ValueError("Computational-basis index is out of range.")
    vector = np.zeros(dimension, dtype=np.complex128)
    vector[index] = 1.0
    return np.outer(vector, vector.conj())


def ghz_density(number_qubits: int) -> np.ndarray:
    dimension = 2 ** number_qubits
    vector = np.zeros(dimension, dtype=np.complex128)
    vector[0] = 1 / np.sqrt(2)
    vector[-1] = 1 / np.sqrt(2)
    return np.outer(vector, vector.conj())


def target_density(
    mode: str,
    number_qubits: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    dimension = 2 ** number_qubits
    normalized_mode = mode.lower()
    if normalized_mode == "zero_state":
        return computational_basis_density(dimension, 0)
    if normalized_mode == "one_state":
        return computational_basis_density(dimension, dimension - 1)
    if normalized_mode == "maximally_mixed":
        return np.eye(dimension, dtype=np.complex128) / dimension
    if normalized_mode == "ghz":
        return ghz_density(number_qubits)
    if normalized_mode == "random_pure":
        if rng is None:
            raise ValueError("random_pure target requires a random generator.")
        return haar_pure_density(dimension, rng)
    if normalized_mode == "random_mixed":
        if rng is None:
            raise ValueError("random_mixed target requires a random generator.")
        return ginibre_mixed_density(dimension, rng)
    raise ValueError(
        f"Unknown targeted state mode {mode!r}. "
        "Use zero_state, one_state, maximally_mixed, ghz, random_pure, or random_mixed."
    )
