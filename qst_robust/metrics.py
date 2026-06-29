from __future__ import annotations

from typing import Dict

import torch

from .losses import frobenius_squared_per_sample, qubit_fidelity
from .quantum import trace_distance


@torch.no_grad()
def reconstruction_metrics(
    estimate: torch.Tensor,
    target: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    fidelity = qubit_fidelity(estimate, target)
    return {
        "fidelity": fidelity,
        "infidelity": 1.0 - fidelity,
        "trace_distance": trace_distance(estimate, target),
        "frobenius": torch.sqrt(frobenius_squared_per_sample(estimate, target)),
    }


@torch.no_grad()
def detection_predictions(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    return (torch.sigmoid(logits) >= threshold).to(torch.float32)
