"""state-CPM: add one normalized, current 27-joint token to the original CPM.

The prepared baseline data/configuration and checkpoint remain valid. The new
variant lives in its own training run; no state history or flow-time encoding is
applied to the current joint observation. Only candidate actions receive noise.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional

from openpi.touchguide.model import CPMConfig
from openpi.touchguide.model import CPMModel
from openpi.touchguide.model import _require_float_tensor

STATE_CPM_SPEC = {
    "schema": "openpi_touchguide_state_cpm_v1",
    "input_key": "state_raw",
    "state_dim": 27,
    "state_kind": "current_absolute_right_arm_7_and_right_hand_20_joint_positions",
    "mlp_dims": [27, 512, 1024],
    "activation": "SiLU",
    "output_normalization": "LayerNorm",
    "tokens": 1,
    "modality_embedding": "independent_learned_state_embedding",
    "normalization": "original_pi05_state_q01_q99_with_epsilon_1e-6_no_clip",
    "statistics_file": "meta/action_norm_stats.json",
    "statistics_key": "norm_stats.state",
    "noise_on_state": False,
    "flow_time_on_state": False,
}


class StateCPMModel(CPMModel):
    uses_state = True

    def __init__(
        self,
        config: CPMConfig,
        *,
        vision_directory: str | Path,
        tactile_rms: tuple[float, float, float],
        state_stats: dict[str, Any],
    ):
        # Construct the baseline first, preserving its seeded initialization.
        super().__init__(config, vision_directory=vision_directory, tactile_rms=tactile_rms)
        low, high = (torch.as_tensor(state_stats[key], dtype=torch.float32).clone() for key in ("q01", "q99"))
        for name, value in (("state q01", low), ("state q99", high)):
            _require_float_tensor(value, (27,), name)
        if torch.any(high < low):
            raise ValueError("Invalid state quantile order")
        self.register_buffer("state_q01", low)
        self.register_buffer("state_q99", high)
        self.state_encoder = nn.Sequential(
            nn.Linear(27, 512),
            nn.SiLU(),
            nn.Linear(512, config.embedding_dim),
            nn.LayerNorm(config.embedding_dim),
        )
        self.state_modality_embedding = nn.Parameter(torch.empty(1, 1, config.embedding_dim))
        nn.init.normal_(self.state_modality_embedding, std=0.02)

    def normalize_state(self, state_raw: torch.Tensor) -> torch.Tensor:
        _require_float_tensor(state_raw, (len(state_raw), 27), "current raw state")
        normalized = (state_raw.float() - self.state_q01) / (self.state_q99 - self.state_q01 + 1e-6) * 2.0 - 1.0
        _require_float_tensor(normalized, (len(state_raw), 27), "normalized current state")
        return normalized

    def encode_observation(
        self,
        head: torch.Tensor,
        wrist: torch.Tensor,
        tactile_map_raw: torch.Tensor,
        state_raw: torch.Tensor | None = None,
        *,
        state_normalized: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (state_raw is None) == (state_normalized is None):
            raise ValueError("state-CPM requires exactly one current raw or normalized state")
        if state_normalized is None:
            state_normalized = self.normalize_state(state_raw)
        _require_float_tensor(state_normalized, (len(head), 27), "normalized current state")
        tokens = self.observation_tokens(head, wrist, tactile_map_raw)
        state_token = self.state_encoder(state_normalized.float())[:, None, :] + self.state_modality_embedding
        tokens = torch.cat([tokens, state_token], dim=1)
        if tokens.shape[1] != 457:
            raise RuntimeError("state-CPM requires the baseline 456 tokens plus one current state token")
        fused = self.fusion(tokens)[:, 0]
        return functional.normalize(self.observation_projection(fused).float(), dim=-1, eps=1e-8)

    def encode_observation_batch(self, batch: dict[str, Any]) -> torch.Tensor:
        if "state_raw" not in batch:
            raise ValueError("state-CPM batches must include current state_raw [B,27]")
        return self.encode_observation(
            batch["head"], batch["wrist"], batch["tactile_map_raw"], state_raw=batch["state_raw"]
        )

    @classmethod
    def from_config(cls, root: str | Path):
        root = Path(root).expanduser().resolve(strict=True)
        stats = json.loads((root / "meta/action_norm_stats.json").read_text())["norm_stats"]["state"]
        return cls(**cls.config_arguments(root), state_stats=stats)
