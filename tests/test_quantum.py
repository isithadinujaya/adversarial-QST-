import numpy as np
import torch

from qst.quantum.measurements import PauliMeasurementScheme
from qst.quantum.states import generate_density_matrix


def test_measurement_dimensions_and_normalization():
    for qubits, expected_input in [(1, 6), (2, 36), (3, 216)]:
        scheme = PauliMeasurementScheme(qubits)
        rng = np.random.default_rng(7)
        density, _ = generate_density_matrix(2**qubits, 0.5, rng)
        probabilities = scheme.probabilities_numpy(density)
        assert probabilities.shape == (3**qubits, 2**qubits)
        assert np.allclose(probabilities.sum(axis=-1), 1.0)
        frequencies = scheme.sample_frequencies(density, 1000, rng)
        assert frequencies.shape == (expected_input,)
        assert np.allclose(frequencies.reshape(3**qubits, 2**qubits).sum(-1), 1.0)


def test_exact_replacement_frequency_normalization():
    scheme = PauliMeasurementScheme(2)
    rng = np.random.default_rng(12)
    clean, _ = generate_density_matrix(4, 1.0, rng)
    replacement, _ = generate_density_matrix(4, 0.0, rng)
    frequencies = scheme.sample_replacement_frequencies(clean, replacement, 0.2, 1000, rng)
    sums = torch.from_numpy(frequencies).reshape(9, 4).sum(-1)
    assert torch.allclose(sums, torch.ones_like(sums))
