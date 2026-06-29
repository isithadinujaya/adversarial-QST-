from __future__ import annotations

import torch
from torch import nn

from .config import ModelConfig


class RobustQSTNetwork(nn.Module):
    """Shared MLP with density-matrix reconstruction and attack-detection heads."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        config = ModelConfig() if config is None else config
        self.config = config

        self.shared = nn.Sequential(
            nn.Linear(config.input_dimension, config.hidden_1),
            nn.GELU(),
            nn.Linear(config.hidden_1, config.hidden_2),
            nn.GELU(),
            nn.Linear(config.hidden_2, config.hidden_3),
            nn.GELU(),
        )
        self.reconstruction_head = nn.Sequential(
            nn.Linear(config.hidden_3, config.head_hidden),
            nn.GELU(),
            nn.Linear(config.head_hidden, 4),
        )
        self.detection_head = nn.Sequential(
            nn.Linear(config.hidden_3, config.head_hidden),
            nn.GELU(),
            nn.Linear(config.head_hidden, 1),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def raw_cholesky_to_density(self, raw: torch.Tensor) -> torch.Tensor:
        """Convert four real outputs to a physical 2x2 density matrix."""
        if raw.shape[-1] != 4:
            raise ValueError("The qubit reconstruction head must output four real values.")

        diagonal_00, diagonal_11, lower_real, lower_imag = raw.unbind(dim=-1)
        complex_dtype = torch.complex128 if raw.dtype == torch.float64 else torch.complex64
        batch_shape = raw.shape[:-1]
        t = torch.zeros((*batch_shape, 2, 2), device=raw.device, dtype=complex_dtype)
        t[..., 0, 0] = diagonal_00.to(complex_dtype)
        t[..., 1, 1] = diagonal_11.to(complex_dtype)
        t[..., 1, 0] = lower_real.to(complex_dtype) + 1j * lower_imag.to(complex_dtype)

        unnormalized = t @ t.conj().transpose(-1, -2)
        identity = torch.eye(2, device=raw.device, dtype=complex_dtype)
        unnormalized = unnormalized + self.config.density_jitter * identity
        trace = unnormalized.diagonal(dim1=-2, dim2=-1).real.sum(dim=-1)
        return unnormalized / trace[..., None, None].clamp_min(self.config.density_jitter)

    def forward(self, frequencies: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if frequencies.shape[-1] != self.config.input_dimension:
            raise ValueError(
                f"Expected input dimension {self.config.input_dimension}, "
                f"received {frequencies.shape[-1]}."
            )
        latent = self.shared(frequencies)
        raw_density = self.reconstruction_head(latent)
        density = self.raw_cholesky_to_density(raw_density)
        attack_logit = self.detection_head(latent).squeeze(-1)
        return density, attack_logit
