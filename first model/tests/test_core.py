from __future__ import annotations

import torch

from qst.attacks.frequency_pgd import project_linf_product_simplex
from qst.attacks.physical import physical_replacement_attack
from qst.config import load_config
from qst.losses import RobustTomographyLoss
from qst.measurements import PauliCubeMeasurement
from qst.metrics import quantum_fidelity, trace_distance
from qst.models import build_model
from qst.states import StateMixture, sample_density_matrices


def test_measurement_dimensions_and_block_normalization():
    for qubits, expected in [(1, 6), (2, 36), (3, 216)]:
        measurement = PauliCubeMeasurement(qubits)
        dimension = 2**qubits
        states, _ = sample_density_matrices(
            4,
            dimension,
            StateMixture(0.34, 0.33, 0.33, 0.2, 1.0),
        )
        frequencies = measurement.sample_frequencies(states, 100)
        assert frequencies.shape == (4, expected)
        blocks = frequencies.reshape(4, 3**qubits, dimension)
        assert torch.allclose(blocks.sum(dim=-1), torch.ones(4, 3**qubits))


def test_cholesky_model_outputs_physical_states():
    config = load_config("configs/smoke_one_qubit.yaml")
    model = build_model(config)
    input_tensor = torch.rand(8, config.input_dimension)
    input_tensor = input_tensor.reshape(8, config.num_settings, config.dimension)
    input_tensor = input_tensor / input_tensor.sum(dim=-1, keepdim=True)
    prediction = model(input_tensor.reshape(8, -1))
    traces = torch.diagonal(prediction, dim1=-2, dim2=-1).sum(dim=-1)
    eigenvalues = torch.linalg.eigvalsh(prediction)
    assert torch.allclose(traces.real, torch.ones(8), atol=1e-5)
    assert eigenvalues.min() > -1e-6


def test_physical_attack_obeys_trace_budget():
    states, _ = sample_density_matrices(
        16, 4, StateMixture(0.34, 0.33, 0.33, 0.2, 1.0)
    )
    result = physical_replacement_attack(
        states,
        alpha=0.8,
        epsilon_physical=0.1,
        kind="random_replacement",
    )
    distance = trace_distance(states, result.attacked_state)
    assert torch.all(distance <= 0.10001)
    expected = result.alpha_effective * result.source_replacement_distance
    assert torch.allclose(distance, expected, atol=1e-5)


def test_product_simplex_linf_projection():
    center = torch.tensor([[0.2, 0.3, 0.5, 0.4, 0.1, 0.5]])
    values = center + torch.tensor([[0.5, -0.4, 0.1, -0.5, 0.8, -0.3]])
    projected = project_linf_product_simplex(
        values,
        center,
        epsilon=0.1,
        num_settings=2,
        outcomes_per_setting=3,
    )
    blocks = projected.reshape(1, 2, 3)
    assert torch.allclose(blocks.sum(dim=-1), torch.ones(1, 2), atol=1e-6)
    assert torch.max(torch.abs(projected - center)) <= 0.10001
    assert projected.min() >= 0.0


def test_latest_loss_backward():
    config = load_config("configs/smoke_one_qubit.yaml")
    model = build_model(config)
    measurement = PauliCubeMeasurement(1)
    states, _ = sample_density_matrices(
        8, 2, StateMixture(0.34, 0.33, 0.33, 0.2, 1.0)
    )
    clean = measurement.sample_frequencies(states, 100)
    adversarial = project_linf_product_simplex(
        clean + 0.02 * torch.randn_like(clean),
        clean,
        epsilon=0.03,
        num_settings=3,
        outcomes_per_setting=2,
    )
    clean_prediction = model(clean)
    adversarial_prediction = model(adversarial)
    loss = RobustTomographyLoss()(states, clean_prediction, adversarial_prediction)
    loss.total.backward()
    assert torch.isfinite(loss.total)
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_fidelity_identity():
    states, _ = sample_density_matrices(
        8, 2, StateMixture(0.34, 0.33, 0.33, 0.2, 1.0)
    )
    fidelity = quantum_fidelity(states, states)
    assert torch.allclose(fidelity, torch.ones_like(fidelity), atol=2e-4)
