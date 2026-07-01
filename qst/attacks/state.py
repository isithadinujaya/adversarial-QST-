from __future__ import annotations

import numpy as np

from qst.config import AttackConfig
from qst.quantum.states import generate_density_matrix, target_density


ATTACK_IDS = {
    "clean": 0,
    "random_replacement": 1,
    "targeted_replacement": 2,
    "worst_case_replacement": 3,
    "random_frequency": 4,
    "pgd": 5,
}
ID_TO_ATTACK = {value: key for key, value in ATTACK_IDS.items()}


def trace_distance_numpy(first: np.ndarray, second: np.ndarray) -> float:
    difference = 0.5 * ((first - second) + (first - second).conj().T)
    eigenvalues = np.linalg.eigvalsh(difference)
    return float(0.5 * np.sum(np.abs(eigenvalues)))


def random_replacement_state(
    dimension: int,
    pure_probability: float,
    rng: np.random.Generator,
) -> np.ndarray:
    state, _ = generate_density_matrix(dimension, pure_probability, rng)
    return state


def worst_case_replacement_state(
    clean_density: np.ndarray,
    dimension: int,
    pure_probability: float,
    number_candidates: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if number_candidates <= 0:
        raise ValueError("number_candidates must be positive.")
    best_state: np.ndarray | None = None
    best_distance = -np.inf
    for _ in range(number_candidates):
        candidate = random_replacement_state(dimension, pure_probability, rng)
        distance = trace_distance_numpy(clean_density, candidate)
        if distance > best_distance:
            best_state = candidate
            best_distance = distance
    assert best_state is not None
    return best_state


def choose_replacement_state(
    attack_name: str,
    clean_density: np.ndarray,
    number_qubits: int,
    config: AttackConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    dimension = 2 ** number_qubits
    if attack_name == "random_replacement":
        return random_replacement_state(
            dimension,
            config.replacement_pure_probability,
            rng,
        )
    if attack_name == "targeted_replacement":
        return target_density(config.targeted_state, number_qubits, rng)
    if attack_name == "worst_case_replacement":
        return worst_case_replacement_state(
            clean_density,
            dimension,
            config.replacement_pure_probability,
            config.worst_case_candidates,
            rng,
        )
    raise ValueError(f"{attack_name!r} is not a physical replacement attack.")
