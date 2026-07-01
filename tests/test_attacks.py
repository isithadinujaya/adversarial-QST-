import numpy as np
import torch

from qst.attacks.frequency import (
    pgd_frequency_attack,
    random_frequency_attack_numpy,
)
from qst.config import load_config
from qst.models import build_model
from qst.quantum.measurements import PauliMeasurementScheme
from qst.quantum.states import generate_density_matrix


def test_random_frequency_attack_obeys_simplex_and_epsilon():
    scheme = PauliMeasurementScheme(3)
    rng = np.random.default_rng(8)
    density, _ = generate_density_matrix(8, 0.5, rng)
    clean = scheme.sample_frequencies(density, 1000, rng)
    epsilon = 0.03
    attacked = random_frequency_attack_numpy(
        clean, epsilon, scheme.number_settings, scheme.outcomes_per_setting, rng
    )
    assert np.max(np.abs(attacked - clean)) <= epsilon + 1e-5
    assert np.allclose(attacked.reshape(27, 8).sum(-1), 1.0, atol=1e-5)
    assert np.min(attacked) >= -1e-7


def test_pgd_attack_obeys_constraints():
    config = load_config("configs/one_qubit.yaml")
    config.model.name = "mlp"
    model = build_model(config)
    scheme = PauliMeasurementScheme(1)
    rng = np.random.default_rng(9)
    density, _ = generate_density_matrix(2, 0.5, rng)
    clean_np = scheme.sample_frequencies(density, 1000, rng)
    clean = torch.from_numpy(clean_np).unsqueeze(0)
    target = torch.from_numpy(density.astype(np.complex64)).unsqueeze(0)
    epsilon = 0.04
    attacked = pgd_frequency_attack(
        model,
        clean,
        target,
        epsilon=epsilon,
        step_size=0.01,
        steps=2,
        number_settings=3,
        outcomes_per_setting=2,
        random_start=True,
        detection_evasion_weight=0.1,
    )
    assert torch.max(torch.abs(attacked - clean)) <= epsilon + 1e-5
    sums = attacked.reshape(1, 3, 2).sum(-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
    assert torch.min(attacked) >= -1e-7
