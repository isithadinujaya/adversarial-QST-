from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from .losses import reconstruction_loss_per_sample
from .measurements import (
    frequency_vector_to_plus,
    pauli_plus_probabilities,
    plus_probabilities_to_frequency_vector,
)
from .quantum import random_pure_or_mixed_states, sample_distant_target_states


def sample_uniform_strength(
    batch_size: int,
    minimum: float,
    maximum: float,
    *,
    device: torch.device | str,
) -> torch.Tensor:
    return minimum + (maximum - minimum) * torch.rand(batch_size, device=device)


def _sample_binomial_counts(total_count: torch.Tensor, probabilities: torch.Tensor) -> torch.Tensor:
    distribution = torch.distributions.Binomial(
        total_count=total_count.to(dtype=probabilities.dtype),
        probs=probabilities,
    )
    return distribution.sample()


@torch.no_grad()
def random_physical_replacement_attack(
    rho: torch.Tensor,
    alpha: torch.Tensor,
    *,
    shots_per_basis: int = 1_000,
    replacement_pure_probability: float = 0.5,
    simulation_mode: Literal["marginal", "explicit"] = "marginal",
) -> torch.Tensor:
    """
    Replace each corrupted copy by an independently sampled pure/mixed state.

    ``marginal`` is the efficient, statistically exact implementation. Haar-pure
    and Ginibre-mixed ensembles are unitarily invariant and both average to I/2.
    Therefore an independently redrawn replacement copy gives a + result with
    probability 1/2 in every Pauli basis after the random state is marginalized.

    ``explicit`` generates every replacement density matrix and is provided for
    verification or small demonstrations, but is much slower during training.
    """
    if rho.ndim != 3 or rho.shape[-2:] != (2, 2):
        raise ValueError("rho must have shape [batch,2,2].")
    if alpha.shape != (rho.shape[0],):
        raise ValueError("alpha must have shape [batch].")
    if not torch.all((alpha >= 0.0) & (alpha <= 1.0)):
        raise ValueError("alpha must lie in [0,1].")
    if not 0.0 <= replacement_pure_probability <= 1.0:
        raise ValueError("replacement_pure_probability must lie in [0,1].")
    if simulation_mode not in {"marginal", "explicit"}:
        raise ValueError("simulation_mode must be 'marginal' or 'explicit'.")

    batch = rho.shape[0]
    corrupted_shots = torch.floor(alpha * shots_per_basis).to(torch.long)
    clean_shots = shots_per_basis - corrupted_shots
    clean_probabilities = pauli_plus_probabilities(rho)

    clean_plus = _sample_binomial_counts(
        clean_shots.unsqueeze(-1),
        clean_probabilities,
    )

    if simulation_mode == "marginal":
        random_probabilities = torch.full_like(clean_probabilities, 0.5)
        corrupted_plus = _sample_binomial_counts(
            corrupted_shots.unsqueeze(-1),
            random_probabilities,
        )
    else:
        max_corrupted = int(corrupted_shots.max().item()) if batch else 0
        corrupted_plus = torch.zeros(
            (batch, 3), device=rho.device, dtype=rho.real.dtype
        )
        if max_corrupted:
            copy_indices = torch.arange(max_corrupted, device=rho.device).unsqueeze(0)
            valid_copy_mask = copy_indices < corrupted_shots.unsqueeze(1)
            for basis in range(3):
                replacement_states = random_pure_or_mixed_states(
                    batch * max_corrupted,
                    pure_probability=replacement_pure_probability,
                    device=rho.device,
                    dtype=rho.dtype,
                ).reshape(batch, max_corrupted, 2, 2)
                replacement_probabilities = pauli_plus_probabilities(
                    replacement_states.reshape(-1, 2, 2)
                )[:, basis].reshape(batch, max_corrupted)
                replacement_outcomes = torch.bernoulli(replacement_probabilities)
                corrupted_plus[:, basis] = (
                    replacement_outcomes * valid_copy_mask
                ).sum(dim=1)

    plus_frequencies = (clean_plus + corrupted_plus) / float(shots_per_basis)
    return plus_probabilities_to_frequency_vector(plus_frequencies)


@torch.no_grad()
def targeted_physical_replacement_attack(
    rho: torch.Tensor,
    alpha: torch.Tensor,
    *,
    shots_per_basis: int = 1_000,
    target_states: torch.Tensor | None = None,
    minimum_target_trace_distance: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replace every corrupted copy by the same distant target state."""
    if alpha.shape != (rho.shape[0],):
        raise ValueError("alpha must have shape [batch].")
    if not torch.all((alpha >= 0.0) & (alpha <= 1.0)):
        raise ValueError("alpha must lie in [0,1].")
    if target_states is None:
        target_states = sample_distant_target_states(
            rho,
            minimum_trace_distance=minimum_target_trace_distance,
        )
    if target_states.shape != rho.shape:
        raise ValueError("target_states must have the same shape as rho.")

    corrupted_shots = torch.floor(alpha * shots_per_basis).to(torch.long)
    clean_shots = shots_per_basis - corrupted_shots
    clean_probabilities = pauli_plus_probabilities(rho)
    target_probabilities = pauli_plus_probabilities(target_states)

    clean_plus = _sample_binomial_counts(clean_shots.unsqueeze(-1), clean_probabilities)
    target_plus = _sample_binomial_counts(corrupted_shots.unsqueeze(-1), target_probabilities)
    plus_frequencies = (clean_plus + target_plus) / float(shots_per_basis)
    return plus_probabilities_to_frequency_vector(plus_frequencies), target_states


def _pgd_objective_per_sample(
    model: nn.Module,
    frequencies: torch.Tensor,
    target_rho: torch.Tensor,
    *,
    fidelity_weight: float,
    mode: Literal["standard", "adaptive"],
    adaptive_gamma: float,
) -> torch.Tensor:
    estimate, attack_logits = model(frequencies)
    reconstruction = reconstruction_loss_per_sample(
        estimate,
        target_rho,
        fidelity_weight=fidelity_weight,
    )
    if mode == "standard":
        return reconstruction
    if mode == "adaptive":
        # log(sigmoid(-logit)) is largest (approaches zero) when the detector
        # predicts clean. Maximizing this together with reconstruction error
        # therefore attacks reconstruction while evading detection.
        clean_evasion_reward = -F.softplus(attack_logits)
        return reconstruction + adaptive_gamma * clean_evasion_reward
    raise ValueError(f"Unknown PGD mode: {mode}")


def pgd_frequency_attack(
    model: nn.Module,
    clean_frequencies: torch.Tensor,
    target_rho: torch.Tensor,
    epsilon: float | torch.Tensor,
    *,
    steps: int = 10,
    step_size: float | torch.Tensor | None = None,
    restarts: int = 1,
    mode: Literal["standard", "adaptive"] = "standard",
    adaptive_gamma: float = 0.5,
    fidelity_weight: float = 0.1,
    random_start: bool = True,
) -> torch.Tensor:
    """
    L-infinity PGD on X+,Y+,Z+ while enforcing each (+,-) pair exactly.

    A perturbation to B+ is paired with the opposite perturbation to B-, so
    every basis remains a valid two-outcome frequency distribution.
    """
    if steps <= 0 or restarts <= 0:
        raise ValueError("steps and restarts must be positive.")
    if clean_frequencies.shape[-1] != 6:
        raise ValueError("clean_frequencies must have shape [batch,6].")
    if clean_frequencies.shape[0] != target_rho.shape[0]:
        raise ValueError("Frequency and state batch sizes must match.")

    batch = clean_frequencies.shape[0]
    if batch == 0:
        return clean_frequencies.clone()

    clean_plus = frequency_vector_to_plus(clean_frequencies).detach()
    epsilon_tensor = torch.as_tensor(
        epsilon,
        device=clean_frequencies.device,
        dtype=clean_frequencies.dtype,
    )
    if epsilon_tensor.ndim == 0:
        epsilon_tensor = epsilon_tensor.expand(batch)
    if epsilon_tensor.shape != (batch,):
        raise ValueError("epsilon must be scalar or have shape [batch].")
    if not torch.all((epsilon_tensor >= 0.0) & (epsilon_tensor <= 1.0)):
        raise ValueError("epsilon must lie in [0,1].")
    epsilon_column = epsilon_tensor.unsqueeze(-1)

    lower = torch.maximum(clean_plus - epsilon_column, torch.zeros_like(clean_plus))
    upper = torch.minimum(clean_plus + epsilon_column, torch.ones_like(clean_plus))

    if step_size is None:
        step_column = 2.0 * epsilon_column / float(steps)
    else:
        step_tensor = torch.as_tensor(
            step_size,
            device=clean_frequencies.device,
            dtype=clean_frequencies.dtype,
        )
        if step_tensor.ndim == 0:
            step_tensor = step_tensor.expand(batch)
        if step_tensor.shape != (batch,):
            raise ValueError("step_size must be scalar or have shape [batch].")
        step_column = step_tensor.unsqueeze(-1)

    best_frequencies = clean_frequencies.detach().clone()
    best_scores = torch.full(
        (batch,),
        -torch.inf,
        device=clean_frequencies.device,
        dtype=clean_frequencies.dtype,
    )

    was_training = model.training
    model.eval()
    try:
        for _ in range(restarts):
            if random_start:
                random_delta = (2.0 * torch.rand_like(clean_plus) - 1.0) * epsilon_column
                adversarial_plus = (clean_plus + random_delta).clamp(min=lower, max=upper)
            else:
                adversarial_plus = clean_plus.clone()

            for _ in range(steps):
                adversarial_plus = adversarial_plus.detach().requires_grad_(True)
                adversarial_frequencies = plus_probabilities_to_frequency_vector(adversarial_plus)
                objective_per_sample = _pgd_objective_per_sample(
                    model,
                    adversarial_frequencies,
                    target_rho,
                    fidelity_weight=fidelity_weight,
                    mode=mode,
                    adaptive_gamma=adaptive_gamma,
                )
                gradient = torch.autograd.grad(
                    objective_per_sample.mean(),
                    adversarial_plus,
                    only_inputs=True,
                )[0]
                with torch.no_grad():
                    adversarial_plus = adversarial_plus + step_column * gradient.sign()
                    adversarial_plus = adversarial_plus.clamp(min=lower, max=upper)

            candidate = plus_probabilities_to_frequency_vector(adversarial_plus.detach())
            with torch.no_grad():
                candidate_scores = _pgd_objective_per_sample(
                    model,
                    candidate,
                    target_rho,
                    fidelity_weight=fidelity_weight,
                    mode=mode,
                    adaptive_gamma=adaptive_gamma,
                )
                improved = candidate_scores > best_scores
                best_scores[improved] = candidate_scores[improved]
                best_frequencies[improved] = candidate[improved]
    finally:
        model.train(was_training)

    return best_frequencies.detach()
