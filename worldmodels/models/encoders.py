"""Visual encoders and projection heads for Chapter 3."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn



class ConvEncoder(nn.Module):
    """Lightweight convolutional encoder for 64x64 visual observations."""

    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 16,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        flat_dim = (base_channels * 8) * 4 * 4
        self.fc = nn.Linear(flat_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("ConvEncoder expects input shaped (B, C, H, W)")
        features = self.conv_net(x)
        return self.fc(torch.flatten(features, start_dim=1))


class ProjectionHead(nn.Module):
    """Small nonlinear projection head used by self-supervised objectives."""

    def __init__(self, in_dim: int = 16, hidden_dim: int = 64, out_dim: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Encoder(nn.Module):
    """Unified encoder interface used by the Chapter 1 agent loop."""

    def __init__(self, backbone: nn.Module, recurrent_hidden_dim: Optional[int] = None) -> None:
        super().__init__()
        self.backbone = backbone
        self.recurrent_hidden_dim = recurrent_hidden_dim
        if recurrent_hidden_dim is not None:
            latent_dim = getattr(backbone, "latent_dim", 16)
            self.gru_cell = nn.GRUCell(latent_dim, recurrent_hidden_dim)
        else:
            self.gru_cell = None

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.ndim == 3:
            obs = obs.unsqueeze(0)
        return self.backbone(obs)

    def update(self, state: Optional[torch.Tensor], obs: torch.Tensor) -> torch.Tensor:
        z = self.encode(obs)
        if self.gru_cell is None:
            return z
        if state is None:
            state = torch.zeros(
                z.shape[0],
                self.recurrent_hidden_dim,
                device=z.device,
                dtype=z.dtype,
            )
        return self.gru_cell(z, state)

# Backward-compatible exports; implementations live in masked_autoencoder.py.
from worldmodels.models.masked_autoencoder import MAEEncoder, SimpleMAE
