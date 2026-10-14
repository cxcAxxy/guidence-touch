"""Compact DINO state-CPM for clean and fingertip recovery matched pairs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional

from openpi.touchguide.model import ActionEncoder
from openpi.touchguide.model import CPMModel
from openpi.touchguide.model import _require_float_tensor
from openpi.touchguide.normalization import CPMNormalization


class RecoveryActionEncoder(ActionEncoder):
    def __init__(self, config, profile):
        super().__init__(config)
        self.coordinates = CPMNormalization(profile)

    def forward(self, actions, flow_time):
        # Both clean and noisy inputs remain in original pi0.5 coordinates.
        return super().forward(self.coordinates.pi05_actions(actions), flow_time)


class RecoveryStateCPM(CPMModel):
    uses_state = True
    uses_recovery_normalization = True
    observation_token_count = 120  # fusion CLS + 2*17 camera + 5*17 touch

    def __init__(
        self, config, *, vision_directory, tactile_rms, profile, negative_policy
    ):
        super().__init__(
            config, vision_directory=vision_directory, tactile_rms=tactile_rms
        )
        self.action_encoder = RecoveryActionEncoder(config, profile)
        self.negative_policy = negative_policy
        dim = config.embedding_dim
        self.global_state_encoder = nn.Sequential(
            nn.Linear(27, 512), nn.SiLU(), nn.Linear(512, dim), nn.LayerNorm(dim)
        )
        self.arm_state_encoder = nn.Sequential(
            nn.Linear(7, 256), nn.SiLU(), nn.Linear(256, dim), nn.LayerNorm(dim)
        )
        # One shared encoder, plus the same finger identity used by touch tokens.
        self.finger_state_encoder = nn.Sequential(
            nn.Linear(4, 256), nn.SiLU(), nn.Linear(256, dim), nn.LayerNorm(dim)
        )
        self.state_modality_embedding = nn.Parameter(torch.empty(1, 1, dim))
        self.state_scope_embedding = nn.Parameter(torch.empty(3, dim))
        nn.init.normal_(self.state_modality_embedding, std=0.02)
        nn.init.normal_(self.state_scope_embedding, std=0.02)

    @staticmethod
    def pool_dino_tokens(features):
        if features.ndim != 3 or features.shape[1] != 257:
            raise ValueError("Expected DINO CLS plus 16x16 patch grid")
        patches = (
            features[:, 1:]
            .transpose(1, 2)
            .reshape(len(features), features.shape[2], 16, 16)
        )
        patches = functional.avg_pool2d(patches, 4, 4).flatten(2).transpose(1, 2)
        return torch.cat([features[:, :1], patches], dim=1)

    def observation_tokens(self, head, wrist, tactile_map_raw):
        size = len(head)
        _require_float_tensor(head, (size, 3, 224, 224), "head")
        _require_float_tensor(wrist, (size, 3, 224, 224), "wrist")
        rgb = self.vision.preprocess(torch.cat([head, wrist]))
        touch = self.tactile_preprocessor(tactile_map_raw)
        features = self.pool_dino_tokens(
            self.vision.forward_pixels(torch.cat([rgb, touch]))
        )
        h, w, t = features.split([size, size, size * 5])
        h = self.vision_projection(h) + self.modality_embedding[0]
        w = self.vision_projection(w) + self.modality_embedding[1]
        t = self.tactile_projection(t).reshape(size, 5, 17, self.config.embedding_dim)
        t = t + self.modality_embedding[2] + self.finger_embedding[None, :, None]
        return torch.cat(
            [self.observation_cls.expand(size, -1, -1), h, w, t.flatten(1, 2)], dim=1
        )

    def encode_observation(
        self, head, wrist, tactile_map_raw, state_raw=None, *, state_normalized=None
    ):
        if (state_raw is None) == (state_normalized is None):
            raise ValueError("Provide raw state OR original pi0.5-normalized state")
        coords = self.action_encoder.coordinates
        state = (
            coords.normalize_state(state_raw)
            if state_raw is not None
            else coords.pi05_state(state_normalized)
        )
        _require_float_tensor(state, (len(head), 27), "CPM state")
        global_token = (
            self.global_state_encoder(state)[:, None] + self.state_scope_embedding[0]
        )
        arm_token = (
            self.arm_state_encoder(state[:, :7])[:, None]
            + self.state_scope_embedding[1]
        )
        fingers = self.finger_state_encoder(state[:, 7:].reshape(len(state), 5, 4))
        fingers = fingers + self.state_scope_embedding[2] + self.finger_embedding[None]
        state_tokens = (
            torch.cat([global_token, arm_token, fingers], 1)
            + self.state_modality_embedding
        )
        tokens = torch.cat(
            [self.observation_tokens(head, wrist, tactile_map_raw), state_tokens], 1
        )
        if tokens.shape[1] != 127:
            raise RuntimeError("Recovery state-CPM requires 127 observation tokens")
        fused = self.fusion(tokens)[:, 0]
        return functional.normalize(
            self.observation_projection(fused).float(), dim=-1, eps=1e-8
        )

    def encode_observation_batch(self, batch):
        return self.encode_observation(
            batch["head"],
            batch["wrist"],
            batch["tactile_map_raw"],
            state_raw=batch["state_raw"],
        )

    def forward(self, batch):
        result = super().forward(batch)
        result["matching_actions"] = self.action_encoder.coordinates.pi05_actions(
            batch["actions"]
        ).detach()
        result["negative_policy"] = self.negative_policy
        from openpi.touchguide.negative_pairing import negative_weights  # noqa: PLC0415

        result["negative_weights"] = negative_weights(
            result["matching_actions"], self.negative_policy
        )
        return result

    @classmethod
    def from_config(cls, root):
        root = Path(root).resolve()
        settings = json.loads((root / "config/cpm.json").read_text())
        records = settings["recovery"]
        loaded = {}
        for key in ("normalization", "negative_policy"):
            entry = records[key]
            raw = (root / entry["file"]).read_bytes()
            if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
                raise ValueError(f"Recovery dependency changed: {key}")
            loaded[key] = json.loads(raw)
        return cls(
            **cls.config_arguments(root),
            profile=loaded["normalization"],
            negative_policy=loaded["negative_policy"],
        )
