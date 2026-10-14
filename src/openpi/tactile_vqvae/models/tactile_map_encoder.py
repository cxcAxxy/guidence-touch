"""Encode small three-axis tactile maps into one token per fingertip.

The poker sidecar stores normal, tangent-0, and tangent-1 taxel values in
``[B, F, 3, 7, 5]`` order. Each axis has its own shallow stem; the resulting
feature maps are fused before shared residual processing. Unlike the T-Rex
deformation encoder, no spatial downsampling is used on the 7x5 grid.
"""

from __future__ import annotations

import torch
from torch import nn


class _ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.act(self.norm1(self.conv1(x)))
        return self.act(self.norm2(self.conv2(x)) + residual)


class TactileMapEncoder(nn.Module):
    """Return one embedding per fingertip, preserving the finger dimension.

    Input must be normalized channel-wise with the sidecar's tactile-map
    q01/q99 ranges, in normal/tangent-0/tangent-1 order. The output shape is
    ``[B, F, token_dim]`` for either five or ten fingertips. A ``finger_mask``
    may suppress missing fingers, such as the unused left hand in poker data.
    """

    def __init__(self, token_dim: int = 256) -> None:
        super().__init__()
        if token_dim <= 0:
            raise ValueError("token_dim must be positive")
        self.channel_stems = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(8, 32),
                nn.SiLU(),
            )
            for _ in range(3)
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(96, 64, kernel_size=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
        )
        self.spatial_encoder = nn.Sequential(
            _ResidualBlock(64, 64),
            _ResidualBlock(64, 128),
            _ResidualBlock(128, 128),
        )
        # Retain coarse taxel location instead of averaging the entire map.
        self.pool = nn.AdaptiveAvgPool2d((3, 3))
        self.token_projection = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.LayerNorm(128 * 3 * 3),
            nn.Linear(128 * 3 * 3, token_dim),
        )

    def forward(
        self,
        tactile_map: torch.Tensor,
        finger_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if tactile_map.ndim != 5 or tactile_map.shape[2:] != (3, 7, 5):
            raise ValueError(
                f"Expected tactile maps [B, F, 3, 7, 5], got {tuple(tactile_map.shape)}"
            )
        batch, fingers = tactile_map.shape[:2]
        if fingers not in (5, 10):
            raise ValueError(f"Expected 5 or 10 fingertips, got {fingers}")
        if finger_mask is not None and finger_mask.shape != (batch, fingers):
            raise ValueError(
                f"Expected finger_mask [B, F] = [{batch}, {fingers}], got {tuple(finger_mask.shape)}"
            )

        flat = tactile_map.reshape(batch * fingers, 3, 7, 5)
        channel_features = [stem(flat[:, i : i + 1]) for i, stem in enumerate(self.channel_stems)]
        features = self.fuse(torch.cat(channel_features, dim=1))
        features = self.spatial_encoder(features)
        tokens = self.token_projection(self.pool(features)).reshape(batch, fingers, -1)
        if finger_mask is not None:
            tokens = tokens * finger_mask.to(dtype=tokens.dtype, device=tokens.device).unsqueeze(-1)
        return tokens
