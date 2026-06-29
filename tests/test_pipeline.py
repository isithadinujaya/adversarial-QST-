from __future__ import annotations

import torch
import torch.nn.functional as F

from qst_robust.attacks import (
    pgd_frequency_attack,
    random_physical_replacement_attack,
    targeted_physical_replacement_attack,
)
from qst_robust.losses import reconstruction_loss_per_sample
from qst_robust.measurements import (
    frequency_vector_to_plus,
    sample_pauli_frequencies,
    validate_pauli_frequency_vector,
)
from qst_robust.model import RobustQSTNetwork
from qst_robust.quantum import (
    generate_state_ensemble,
    sample_distant_target_states,
    trace_distance,
    validate_density_matrices,
)


def test_state_generation_is_physical() -> None:
    torch.manual_seed(1)
    states = generate_state_ensemble(128)
    validate_density_matrices(states)


def test_pauli_sampling_is_valid() -> None:
    torch.manual_seed(2)
    states = generate_state_ensemble(32)
    frequencies = sample_pauli_frequencies(states, shots_per_basis=1_000)
    assert frequencies.shape == (32, 6)
    validate_pauli_frequency_vector(frequencies)
    scaled = frequencies * 1_000
    assert torch.allclose(scaled, scaled.round(), atol=1e-5)


def test_random_physical_attack_is_valid() -> None:
    torch.manual_seed(3)
    states = generate_state_ensemble(16)
    alpha = torch.linspace(0.01, 0.20, 16)
    attacked = random_physical_replacement_attack(
        states,
        alpha,
        shots_per_basis=200,
    )
    validate_pauli_frequency_vector(attacked)




def test_explicit_random_physical_attack_is_valid() -> None:
    torch.manual_seed(33)
    states = generate_state_ensemble(4)
    alpha = torch.full((4,), 0.20)
    attacked = random_physical_replacement_attack(
        states,
        alpha,
        shots_per_basis=20,
        simulation_mode="explicit",
    )
    validate_pauli_frequency_vector(attacked)


def test_targeted_attack_uses_distant_targets() -> None:
    torch.manual_seed(4)
    states = generate_state_ensemble(32)
    targets = sample_distant_target_states(states, minimum_trace_distance=0.5)
    distances = trace_distance(states, targets)
    assert torch.all(distances >= 0.5 - 1e-5)

    alpha = torch.full((32,), 0.2)
    attacked, returned_targets = targeted_physical_replacement_attack(
        states,
        alpha,
        shots_per_basis=200,
        target_states=targets,
    )
    validate_pauli_frequency_vector(attacked)
    assert torch.allclose(targets, returned_targets)


def test_model_outputs_physical_density_and_logit() -> None:
    torch.manual_seed(5)
    model = RobustQSTNetwork()
    states = generate_state_ensemble(24)
    frequencies = sample_pauli_frequencies(states)
    estimates, logits = model(frequencies)
    assert estimates.shape == (24, 2, 2)
    assert logits.shape == (24,)
    validate_density_matrices(estimates, atol=2e-5)


def test_pgd_respects_pair_and_linf_constraints() -> None:
    torch.manual_seed(6)
    model = RobustQSTNetwork()
    states = generate_state_ensemble(12)
    clean = sample_pauli_frequencies(states)
    epsilon = torch.linspace(0.005, 0.03, 12)
    attacked = pgd_frequency_attack(
        model,
        clean,
        states,
        epsilon,
        steps=3,
        restarts=2,
        mode="standard",
    )
    validate_pauli_frequency_vector(attacked)
    difference = (frequency_vector_to_plus(attacked) - frequency_vector_to_plus(clean)).abs()
    assert torch.all(difference <= epsilon.unsqueeze(-1) + 1e-6)


def test_one_training_backward_pass() -> None:
    torch.manual_seed(7)
    model = RobustQSTNetwork()
    states = generate_state_ensemble(18)
    clean = sample_pauli_frequencies(states, shots_per_basis=100)
    alpha = torch.full((18,), 0.1)
    adversarial = random_physical_replacement_attack(
        states,
        alpha,
        shots_per_basis=100,
    )

    clean_estimate, clean_logits = model(clean)
    adversarial_estimate, adversarial_logits = model(adversarial)
    reconstruction = reconstruction_loss_per_sample(clean_estimate, states).mean()
    reconstruction = reconstruction + reconstruction_loss_per_sample(
        adversarial_estimate, states
    ).mean()
    logits = torch.cat((clean_logits, adversarial_logits))
    labels = torch.cat((torch.zeros(18), torch.ones(18)))
    loss = reconstruction + 0.1 * F.binary_cross_entropy_with_logits(logits, labels)
    loss.backward()

    gradient_norm = sum(
        parameter.grad.abs().sum().item()
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    assert gradient_norm > 0.0
