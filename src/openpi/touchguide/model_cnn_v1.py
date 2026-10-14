"""Paper-inspired CPM for two RGB views, five small force maps and action chunks.

TouchGuide's public checkout does not contain its CPM implementation. The
Transformer dimensions, small-grid tactile CNN and explicit noise-time embedding
below are local choices. No pi0.5 model or training loop is loaded here.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional

from openpi.tactile_vqvae.models.tactile_map_encoder import TactileMapEncoder


@dataclass(frozen=True)
class CPMConfig:
    embedding_dim: int = 256
    fusion_layers: int = 4
    fusion_heads: int = 8
    fusion_ff_dim: int = 1024
    dropout: float = 0.1
    action_horizon: int = 30
    action_dim: int = 27
    action_pool_bins: int = 6
    time_embedding_dim: int = 64
    time_conditioning: bool = True
    initial_temperature: float = 0.07
    min_temperature: float = 0.01
    max_temperature: float = 1.0
    vision_batch_size: int = 16
    touch_contact_threshold_n: float = 1e-6

    def __post_init__(self) -> None:
        dimensions = (
            self.embedding_dim,
            self.fusion_layers,
            self.fusion_heads,
            self.fusion_ff_dim,
            self.action_pool_bins,
            self.time_embedding_dim,
            self.vision_batch_size,
        )
        if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in dimensions):
            raise ValueError("CPM dimensions must be positive integers")
        if self.embedding_dim % self.fusion_heads or self.time_embedding_dim % 2:
            raise ValueError("embedding_dim must divide into heads; time_embedding_dim must be even")
        if (self.action_horizon, self.action_dim) != (30, 27) or self.action_pool_bins > self.action_horizon:
            raise ValueError("This CPM uses 30-step, 27-dimensional actions")
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        if not isinstance(self.time_conditioning, bool):
            raise ValueError("time_conditioning must be boolean")
        if not math.isfinite(self.touch_contact_threshold_n) or self.touch_contact_threshold_n < 0:
            raise ValueError("touch_contact_threshold_n must be finite and nonnegative")
        temperatures = (self.min_temperature, self.initial_temperature, self.max_temperature)
        if (
            any(not math.isfinite(value) for value in temperatures)
            or not 0 < temperatures[0] <= temperatures[1] <= temperatures[2]
        ):
            raise ValueError("Invalid temperature bounds or initialization")


def _require_float_tensor(value: torch.Tensor, shape: tuple[int, ...], name: str) -> None:
    if not isinstance(value, torch.Tensor) or not value.is_floating_point() or tuple(value.shape) != shape:
        raise ValueError(f"Expected floating {name} {shape}")
    if not torch.isfinite(value).all():
        raise ValueError(f"Nonfinite {name}")


class FrozenDinov2(nn.Module):
    """Load verified local DINOv2-small weights, with no network or random fallback."""

    def __init__(self, directory: str | Path, *, batch_size: int = 16):
        super().__init__()
        from transformers import Dinov2Model  # noqa: PLC0415 - defer backbone dependency until model creation

        directory = Path(directory).expanduser().resolve(strict=True)
        manifest = json.loads((directory / "backbone_manifest.json").read_text())
        if (
            manifest["schema"] != "touchguide_public_backbone_v1"
            or manifest["repository"] != "facebook/dinov2-small"
            or manifest["revision"] != "ed25f3a31f01632728cabb09d1542f84ab7b0056"
            or manifest["files"]["model.safetensors"]["sha256"]
            != "ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1"
        ):
            raise ValueError("Use the verified facebook/dinov2-small backbone")
        for name in ("config.json", "model.safetensors", "preprocessor_config.json"):
            path = directory / name
            expected = manifest["files"][name]
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected["sha256"]:
                raise ValueError(f"DINOv2 artifact hash mismatch: {name}")
        backbone, information = Dinov2Model.from_pretrained(
            str(directory),
            local_files_only=True,
            output_loading_info=True,
            attn_implementation="sdpa",
        )
        if any(information[key] for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise ValueError("DINOv2 checkpoint does not load exactly")
        if backbone.config.hidden_size != 384 or backbone.config.patch_size != 14:
            raise ValueError("Expected DINOv2-small with 384 features and 14-pixel patches")
        self.backbone = backbone.requires_grad_(requires_grad=False).eval()
        self.feature_dim = backbone.config.hidden_size
        self.batch_size = batch_size
        self.register_buffer("rgb_mean", torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1))
        self.register_buffer("rgb_std", torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1))
        self.source_manifest = manifest

    def train(self, mode: bool = True):  # noqa: FBT001, FBT002 - preserve nn.Module.train's public signature
        super().train(mode)
        self.backbone.eval()
        return self

    def preprocess(self, rgb: torch.Tensor) -> torch.Tensor:
        _require_float_tensor(rgb, (len(rgb), 3, 224, 224), "RGB")
        if rgb.min() < -1.0001 or rgb.max() > 1.0001:
            raise ValueError("The clean reader must supply RGB in [-1,1]")
        # The reader already resizes/pads to 224. Restore [0,1] exactly once,
        # then use DINOv2's ImageNet normalization; no second resize/crop.
        return ((rgb + 1) * 0.5 - self.rgb_mean) / self.rgb_std

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        pixels = self.preprocess(rgb)
        self.backbone.eval()
        with torch.no_grad():
            values = [self.backbone(pixel_values=chunk).last_hidden_state for chunk in pixels.split(self.batch_size)]
        # CLS plus the 16x16 patch grid. Positional interpolation is handled
        # by the official DINOv2 model when its training resolution differs.
        return torch.cat(values)


class ActionEncoder(nn.Module):
    """Temporal 1D CNN + ordered pooled MLP, with optional local time conditioning."""

    def __init__(self, config: CPMConfig):
        super().__init__()
        self.config = config
        self.temporal = nn.Sequential(
            nn.Conv1d(config.action_dim, 128, kernel_size=5, padding=2),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.GroupNorm(8, 256),
            nn.SiLU(),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.GroupNorm(8, 256),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(config.action_pool_bins),
        )
        self.projection = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(256 * config.action_pool_bins, 512),
            nn.SiLU(),
            nn.Linear(512, config.embedding_dim),
        )
        self.register_buffer("time_frequencies", torch.logspace(0, 3, config.time_embedding_dim // 2))
        self.time_mlp = (
            nn.Sequential(
                nn.Linear(config.time_embedding_dim, config.embedding_dim),
                nn.SiLU(),
                nn.Linear(config.embedding_dim, config.embedding_dim),
            )
            if config.time_conditioning
            else None
        )
        self.output = nn.Sequential(
            nn.LayerNorm(config.embedding_dim), nn.Linear(config.embedding_dim, config.embedding_dim)
        )

    def forward(self, actions: torch.Tensor, flow_time: torch.Tensor | float) -> torch.Tensor:
        size = len(actions)
        if size < 1:
            raise ValueError("An action encoder needs at least one candidate")
        _require_float_tensor(actions, (size, self.config.action_horizon, self.config.action_dim), "noisy_actions")
        time = torch.as_tensor(flow_time, dtype=torch.float32, device=actions.device)
        if time.ndim == 0:
            time = time.expand(size)
        if time.shape != (size,) or not torch.isfinite(time).all() or (time < 0).any() or (time > 1).any():
            raise ValueError("flow_time must be a scalar or [B] in [0,1]")
        features = self.projection(self.temporal(actions.transpose(1, 2)))
        if self.time_mlp is not None:
            phase = 2 * math.pi * time[:, None] * self.time_frequencies[None]
            features = features + self.time_mlp(torch.cat([phase.sin(), phase.cos()], dim=-1))
        return functional.normalize(self.output(features).float(), dim=-1, eps=1e-8)


class CPMModel(nn.Module):
    """Produce normalized features, raw cosine scores and temperature-scaled logits."""

    def __init__(self, config: CPMConfig, *, vision_directory: str | Path, tactile_rms: tuple[float, float, float]):
        super().__init__()
        if len(tactile_rms) != 3 or any(not math.isfinite(value) or value <= 0 for value in tactile_rms):
            raise ValueError("Use three finite positive train-only tactile channel RMS values")
        self.config = config
        self.vision = FrozenDinov2(vision_directory, batch_size=config.vision_batch_size)
        self.vision_projection = nn.Linear(self.vision.feature_dim, config.embedding_dim)
        # Reuse the small-grid architecture, freshly initialized for this CPM.
        # Older quantile/clipping-based encoder weights are not loaded here.
        self.tactile_encoder = TactileMapEncoder(token_dim=config.embedding_dim)
        # GroupNorm stabilizes the spatial CNN but can suppress common force
        # amplitude. Preserve amplitude through statistics from these same taxels.
        self.tactile_magnitude = nn.Sequential(nn.Linear(9, 128), nn.SiLU(), nn.Linear(128, config.embedding_dim))
        self.register_buffer("tactile_rms", torch.tensor(tactile_rms).reshape(1, 1, 3, 1, 1))
        self.modality_embedding = nn.Parameter(torch.empty(3, config.embedding_dim))
        self.finger_embedding = nn.Parameter(torch.empty(5, config.embedding_dim))
        self.no_contact_token = nn.Parameter(torch.empty(1, 1, config.embedding_dim))
        self.observation_cls = nn.Parameter(torch.empty(1, 1, config.embedding_dim))
        layer = nn.TransformerEncoderLayer(
            config.embedding_dim,
            config.fusion_heads,
            dim_feedforward=config.fusion_ff_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.fusion = nn.TransformerEncoder(
            layer, config.fusion_layers, norm=nn.LayerNorm(config.embedding_dim), enable_nested_tensor=False
        )
        # TransformerEncoder clones one prototype. Independently initialize the
        # cloned layers rather than giving all layers identical starting weights.
        for block in self.fusion.layers:
            nn.init.xavier_uniform_(block.self_attn.in_proj_weight)
            nn.init.zeros_(block.self_attn.in_proj_bias)
            for linear in (block.self_attn.out_proj, block.linear1, block.linear2):
                nn.init.xavier_uniform_(linear.weight)
                nn.init.zeros_(linear.bias)
        self.observation_projection = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.action_encoder = ActionEncoder(config)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / config.initial_temperature)))
        nn.init.normal_(self.modality_embedding, std=0.02)
        nn.init.normal_(self.finger_embedding, std=0.02)
        nn.init.normal_(self.no_contact_token, std=0.02)
        nn.init.normal_(self.observation_cls, std=0.02)

    @property
    def temperature(self) -> torch.Tensor:
        lower = math.log(1 / self.config.max_temperature)
        upper = math.log(1 / self.config.min_temperature)
        return self.logit_scale.clamp(lower, upper).exp().reciprocal()

    def scale_tactile(self, tactile: torch.Tensor) -> torch.Tensor:
        _require_float_tensor(tactile, (len(tactile), 5, 3, 7, 5), "tactile_map_raw")
        # Divide by fixed train RMS, preserving physical zero, signs and values
        # beyond the demonstration range; there is no clipping or centering.
        return tactile / self.tactile_rms

    def encode_observation(
        self, head: torch.Tensor, wrist: torch.Tensor, tactile_map_raw: torch.Tensor
    ) -> torch.Tensor:
        size = len(tactile_map_raw)
        if size < 1 or len(head) != size or len(wrist) != size:
            raise ValueError("Current camera and tactile batch sizes must match")
        camera_features = self.vision(torch.cat([head, wrist]))
        if camera_features.shape[1] != 257:
            raise ValueError("224x224 DINOv2 must return CLS plus 256 spatial patch tokens")
        head_tokens, wrist_tokens = self.vision_projection(camera_features).split(size)
        scaled_touch = self.scale_tactile(tactile_map_raw)
        contact = tactile_map_raw.abs().amax(dim=(-3, -2, -1)) > self.config.touch_contact_threshold_n
        # All-zero fingers can amplify gradients through stacked GroupNorm.
        # Mask the spatial path before fusion and learn an explicit empty-contact
        # representation. Contact is derived from current taxels, not a label.
        touch_tokens = self.tactile_encoder(scaled_touch, finger_mask=contact)
        touch_tokens = touch_tokens + (~contact).unsqueeze(-1) * self.no_contact_token
        magnitude = torch.cat(
            [
                scaled_touch.mean(dim=(-2, -1)),
                scaled_touch.abs().mean(dim=(-2, -1)),
                scaled_touch.abs().amax(dim=(-2, -1)),
            ],
            dim=-1,
        )
        touch_tokens = touch_tokens + self.tactile_magnitude(magnitude)
        head_tokens = head_tokens + self.modality_embedding[0]
        wrist_tokens = wrist_tokens + self.modality_embedding[1]
        touch_tokens = touch_tokens + self.modality_embedding[2] + self.finger_embedding
        tokens = torch.cat([self.observation_cls.expand(size, -1, -1), head_tokens, wrist_tokens, touch_tokens], dim=1)
        fused = self.fusion(tokens)[:, 0]
        return functional.normalize(self.observation_projection(fused).float(), dim=-1, eps=1e-8)

    def score_actions(
        self, observation_embedding: torch.Tensor, noisy_actions: torch.Tensor, flow_time: torch.Tensor | float
    ) -> torch.Tensor:
        """Raw [observations,candidates] cosine, without training temperature or softmax."""
        _require_float_tensor(
            observation_embedding, (len(observation_embedding), self.config.embedding_dim), "observation_embedding"
        )
        if not torch.allclose(
            observation_embedding.norm(dim=-1),
            torch.ones(len(observation_embedding), device=observation_embedding.device),
            atol=1e-4,
            rtol=1e-4,
        ):
            raise ValueError("Cache the normalized encode_observation output")
        action = self.action_encoder(noisy_actions, flow_time)
        with torch.autocast(device_type=observation_embedding.device.type, enabled=False):
            return observation_embedding.float() @ action.float().T

    def paired_scores(
        self, observation_embedding: torch.Tensor, noisy_actions: torch.Tensor, flow_time: torch.Tensor | float
    ) -> torch.Tensor:
        """Score aligned pairs or many candidates for one cached observation, with gradients."""
        matrix = self.score_actions(observation_embedding, noisy_actions, flow_time)
        if len(observation_embedding) == 1:
            return matrix[0]
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Paired scoring needs equal counts or one observation")
        return matrix.diagonal()

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        required = ("head", "wrist", "tactile_map_raw", "noisy_actions", "flow_time")
        if any(key not in batch for key in required):
            raise ValueError("CPM requires an augmented batch: RGB, taxels, noisy_actions and flow_time")
        if "negative_actions" in batch or "noisy_negative_actions" in batch:
            raise ValueError("Use shared action columns rather than legacy negative tensors")
        if len(batch["head"]) != len(batch["noisy_actions"]):
            raise ValueError("Contrastive forward requires one matched action column per observation")
        time = torch.as_tensor(batch["flow_time"])
        if time.ndim != 0:
            raise ValueError("Contrastive batches must share a scalar noise time")
        observation = self.encode_observation(batch["head"], batch["wrist"], batch["tactile_map_raw"])
        action = self.action_encoder(batch["noisy_actions"], batch["flow_time"])
        with torch.autocast(device_type=observation.device.type, enabled=False):
            cosine = observation.float() @ action.float().T
        temperature = self.temperature.float()
        return {
            "observation_embedding": observation,
            "action_embedding": action,
            "cosine_scores": cosine,
            "logits": cosine / temperature,
            "temperature": temperature,
        }

    def parameter_counts(self) -> dict[str, int]:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}

    @classmethod
    def from_config(cls, root: str | Path):
        root = Path(root).expanduser().resolve(strict=True)
        settings = json.loads((root / "config/cpm.json").read_text())
        if settings["schema"] != "openpi_touchguide_cpm_config_v1":
            raise ValueError("Unsupported CPM configuration")
        vision_manifest = Path(settings["vision"]["directory"]) / "backbone_manifest.json"
        if hashlib.sha256(vision_manifest.read_bytes()).hexdigest() != settings["vision"]["manifest_sha256"]:
            raise ValueError("The visual dependency changed after CPM preparation")
        scale_path = root / settings["tactile"]["statistics_file"]
        if hashlib.sha256(scale_path.read_bytes()).hexdigest() != settings["tactile"]["statistics_sha256"]:
            raise ValueError("Tactile statistics changed after CPM preparation")
        stats = json.loads(scale_path.read_text())
        if settings["tactile"]["normalization"] != "divide_by_train_channel_rms_no_clip":
            raise ValueError("Use the explicitly configured unclipped tactile RMS transform")
        return cls(
            CPMConfig(**settings["model_kwargs"]),
            vision_directory=settings["vision"]["directory"],
            tactile_rms=tuple(stats["rms"]),
        )
