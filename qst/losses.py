from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from qst.metrics import frobenius_distance


@dataclass
class LossOutput:
    total: torch.Tensor
    clean: torch.Tensor
    adversarial: torch.Tensor
    consistency: torch.Tensor

    def detached_dict(self) -> dict[str, float]:
        return {
            "total": float(self.total.detach().cpu()),
            "clean": float(self.clean.detach().cpu()),
            "adversarial": float(self.adversarial.detach().cpu()),
            "consistency": float(self.consistency.detach().cpu()),
        }


class RobustTomographyLoss(nn.Module):
    """Squared Frobenius loss for robust quantum-state reconstruction."""

    def __init__(
        self,
        *,
        clean_weight: float = 1.0,
        adversarial_weight: float = 1.0,
        consistency_weight: float = 0.1,
    ) -> None:
        super().__init__()

        self.clean_weight = clean_weight
        self.adversarial_weight = adversarial_weight
        self.consistency_weight = consistency_weight

    def forward(
        self,
        target_rho: torch.Tensor,
        clean_prediction: torch.Tensor,
        adversarial_prediction: torch.Tensor,
    ) -> LossOutput:

        # ||rho - rho_hat_clean||_F^2
        clean = frobenius_distance(
            target_rho,
            clean_prediction,
        ).square().mean()

        # ||rho - rho_hat_adv||_F^2
        adversarial = frobenius_distance(
            target_rho,
            adversarial_prediction,
        ).square().mean()

        # ||stopgrad(rho_hat_clean) - rho_hat_adv||_F^2
        consistency = frobenius_distance(
            clean_prediction.detach(),
            adversarial_prediction,
        ).square().mean()

        total = (
            self.clean_weight * clean
            + self.adversarial_weight * adversarial
            + self.consistency_weight * consistency
        )

        return LossOutput(
            total=total,
            clean=clean,
            adversarial=adversarial,
            consistency=consistency,
        )