"""Autoencoder used to pretrain the three-axis tactile-map encoder."""

from __future__ import annotations

import torch
from torch import nn

from .tactile_map_encoder import TactileMapEncoder


class _MapDecoder(nn.Module):
    def __init__(self, token_dim: int) -> None:
        super().__init__()
        self.project = nn.Linear(token_dim, 128 * 3 * 3)
        self.conv = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, fingers, width = tokens.shape
        features = self.project(tokens.reshape(batch * fingers, width)).reshape(batch * fingers, 128, 3, 3)
        features = torch.nn.functional.interpolate(features, size=(7, 5), mode="bilinear", align_corners=False)
        return self.conv(features).reshape(batch, fingers, 3, 7, 5)


class TactileMapAutoencoder(nn.Module):
    """Reconstruct normalized maps through one 256D token per fingertip."""

    def __init__(self, token_dim: int = 256) -> None:
        super().__init__()
        self.encoder = TactileMapEncoder(token_dim=token_dim)
        self.decoder = _MapDecoder(token_dim=token_dim)

    def forward(self, maps: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(maps))
