from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


def _squared_frobenius_per_sample(
    first: torch.Tensor,
    second: torch.Tensor,
) -> torch.Tensor:
    """Return ||first-second||_F^2 over the final two matrix dimensions."""
    difference = first - second
    return (difference.conj() * difference).real.sum(dim=(-2, -1))


@dataclass
class LossOutput:
    total: torch.Tensor
    clean: torch.Tensor
    adversarial: torch.Tensor
    physical: torch.Tensor
    pgd: torch.Tensor
    consistency: torch.Tensor
    physical_consistency: torch.Tensor
    pgd_consistency: torch.Tensor

    def detached_dict(self) -> dict[str, float]:
        return {
            "total": float(self.total.detach().cpu()),
            "clean": float(self.clean.detach().cpu()),
            "adversarial": float(self.adversarial.detach().cpu()),
            "physical": float(self.physical.detach().cpu()),
            "pgd": float(self.pgd.detach().cpu()),
            "consistency": float(self.consistency.detach().cpu()),
            "physical_consistency": float(
                self.physical_consistency.detach().cpu()
            ),
            "pgd_consistency": float(self.pgd_consistency.detach().cpu()),
        }


class HierarchicalRobustTomographyLoss(nn.Module):
    """Separate physical-family and frequency-PGD squared-Frobenius losses."""

    def __init__(
        self,
        *,
        clean_weight: float = 1.0,
        physical_weight: float = 0.5,
        pgd_weight: float = 0.5,
        consistency_weight: float = 0.1,
        physical_max_weight: float = 0.7,
    ) -> None:
        super().__init__()
        self.clean_weight = clean_weight
        self.physical_weight = physical_weight
        self.pgd_weight = pgd_weight
        self.consistency_weight = consistency_weight
        self.physical_max_weight = physical_max_weight

    def forward(
        self,
        target_rho: torch.Tensor,
        clean_prediction: torch.Tensor,
        physical_predictions: torch.Tensor,
        pgd_prediction: torch.Tensor,
    ) -> LossOutput:
        """
        Args:
            target_rho: [B,d,d]
            clean_prediction: [B,d,d]
            physical_predictions: [B,K,d,d]
            pgd_prediction: [B,d,d]
        """
        if physical_predictions.ndim != target_rho.ndim + 1:
            raise ValueError("physical_predictions must have shape [B,K,d,d].")
        if physical_predictions.shape[0] != target_rho.shape[0]:
            raise ValueError("Physical predictions and targets have different batches.")

        beta = self.physical_max_weight

        clean_per_sample = _squared_frobenius_per_sample(
            target_rho,
            clean_prediction,
        )
        physical_per_sample = _squared_frobenius_per_sample(
            target_rho[:, None, :, :],
            physical_predictions,
        )
        pgd_per_sample = _squared_frobenius_per_sample(
            target_rho,
            pgd_prediction,
        )

        # The strongest physical attack is selected independently for each state.
        worst_physical_indices = physical_per_sample.argmax(
            dim=1,
            keepdim=True,
        )
        physical_max_per_sample = physical_per_sample.gather(
            1,
            worst_physical_indices,
        ).squeeze(1)
        physical_avg_per_sample = physical_per_sample.mean(dim=1)

        clean_loss = clean_per_sample.mean()
        physical_loss = (
            beta * physical_max_per_sample.mean()
            + (1.0 - beta) * physical_avg_per_sample.mean()
        )
        pgd_loss = pgd_per_sample.mean()
        adversarial_loss = (
            self.physical_weight * physical_loss
            + self.pgd_weight * pgd_loss
        )

        clean_reference = clean_prediction.detach()
        physical_consistency_per_sample = _squared_frobenius_per_sample(
            clean_reference[:, None, :, :],
            physical_predictions,
        )
        pgd_consistency_per_sample = _squared_frobenius_per_sample(
            clean_reference,
            pgd_prediction,
        )

        # Use the same physically worst attack chosen by reconstruction error.
        physical_consistency_max = physical_consistency_per_sample.gather(
            1,
            worst_physical_indices,
        ).squeeze(1)
        physical_consistency_avg = physical_consistency_per_sample.mean(dim=1)
        physical_consistency = (
            beta * physical_consistency_max.mean()
            + (1.0 - beta) * physical_consistency_avg.mean()
        )
        pgd_consistency = pgd_consistency_per_sample.mean()
        consistency_loss = (
            self.physical_weight * physical_consistency
            + self.pgd_weight * pgd_consistency
        )

        total = (
            self.clean_weight * clean_loss
            + adversarial_loss
            + self.consistency_weight * consistency_loss
        )

        return LossOutput(
            total=total,
            clean=clean_loss,
            adversarial=adversarial_loss,
            physical=physical_loss,
            pgd=pgd_loss,
            consistency=consistency_loss,
            physical_consistency=physical_consistency,
            pgd_consistency=pgd_consistency,
        )


# Keep this alias so imports elsewhere in the project remain simple.
RobustTomographyLoss = HierarchicalRobustTomographyLoss
