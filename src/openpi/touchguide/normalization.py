"""CPM-only affine coordinates, independent of the frozen pi0.5 coordinates.

The profile is fitted using training data only. Both modalities share hand
absolute-position ranges, while arm state and arm-delta actions remain distinct.
Apply ``pi05_actions`` AFTER generating flow noise in pi0.5 coordinates, in both
training and inference. This maps the original noise distribution too; do not
replace it with a fresh standard Gaussian in CPM coordinates. No clipping.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn


class CPMNormalization(nn.Module):
    """Differentiable coordinate adapter; outputs FP32 without input clipping."""

    def __init__(self, profile: dict):
        super().__init__()
        if profile["schema"] != "touchguide_cpm_normalization_v1":
            raise ValueError("Unsupported CPM coordinate profile")
        self.profile = profile
        for modality in ("actions", "state"):
            bounds = profile[modality]
            low, high = (
                torch.as_tensor(bounds[key], dtype=torch.float32)
                for key in ("low", "high")
            )
            if low.shape != (27,) or high.shape != (27,):
                raise ValueError("Expected 27-dimensional CPM bounds")
            if (
                not torch.isfinite(low).all()
                or not torch.isfinite(high).all()
                or torch.any(high <= low)
            ):
                raise ValueError("CPM bounds must be finite with positive width")
            original = profile["original_pi05_stats"][modality]
            old_low, old_high = (
                torch.as_tensor(original[key], dtype=torch.float32)
                for key in ("q01", "q99")
            )
            if old_low.shape != (27,) or old_high.shape != (27,):
                raise ValueError("Expected 27-dimensional original quantiles")
            if (
                not torch.isfinite(old_low).all()
                or not torch.isfinite(old_high).all()
                or torch.any(old_high < old_low)
            ):
                raise ValueError("Invalid original pi0.5 quantiles")
            self.register_buffer(f"{modality}_low", low)
            self.register_buffer(f"{modality}_width", high - low)
            self.register_buffer(f"{modality}_original_low", old_low)
            self.register_buffer(
                f"{modality}_original_width", old_high - old_low + 1e-6
            )

    @classmethod
    def from_file(cls, path: str | Path):
        return cls(json.loads(Path(path).read_text()))

    @staticmethod
    def _input(values: torch.Tensor, *, padded: bool = False) -> torch.Tensor:
        if not values.is_floating_point() or values.shape[-1] not in (
            (27, 32) if padded else (27,)
        ):
            raise ValueError(
                "Expected floating 27-dimensional coordinates (or padded pi0.5 32 dimensions)"
            )
        return values[..., :27].float()

    def _normalize(self, values: torch.Tensor, modality: str) -> torch.Tensor:
        return (
            2
            * (values.float() - getattr(self, f"{modality}_low"))
            / getattr(self, f"{modality}_width")
            - 1
        )

    def normalize_state(self, raw_state: torch.Tensor) -> torch.Tensor:
        return self._normalize(self._input(raw_state), "state")

    def normalize_physical_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Input arm delta 7 + hand absolute 20, not 27 absolute targets."""
        return self._normalize(self._input(actions), "actions")

    def _from_pi05(self, values: torch.Tensor, modality: str) -> torch.Tensor:
        values = self._input(values, padded=True)
        physical = (values + 1) * 0.5 * getattr(
            self, f"{modality}_original_width"
        ) + getattr(self, f"{modality}_original_low")
        return self._normalize(physical, modality)

    def pi05_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Convert clean OR already noisy pi0.5 actions, preserving gradients."""
        return self._from_pi05(actions, "actions")

    def pi05_state(self, state: torch.Tensor) -> torch.Tensor:
        return self._from_pi05(state, "state")

    def action_targets(
        self, normalized: torch.Tensor, raw_state: torch.Tensor
    ) -> torch.Tensor:
        """Decode [B,T,27] CPM coordinates to absolute joint targets in radians."""
        normalized = self._input(normalized)
        raw_state = self._input(raw_state)
        if normalized.ndim != 3 or raw_state.shape != (len(normalized), 27):
            raise ValueError("Expected CPM actions [B,T,27] and state [B,27]")
        physical = (normalized + 1) * 0.5 * self.actions_width + self.actions_low
        return torch.cat(
            [physical[..., :7] + raw_state[:, None, :7], physical[..., 7:]], dim=-1
        )
