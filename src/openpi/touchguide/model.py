"""TouchGuide-inspired CPM with one frozen DINOv2-Base for RGB and force maps.

The 8x1024 fusion, action encoder sizes, tactile asinh mapping and spatial
pooling are explicit local choices close to the paper's 118M trainable budget.
No pi0.5 policy is loaded or updated here.
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

from openpi.touchguide.tactile_dino import TactileDinoPreprocessor


@dataclass(frozen=True)
class CPMConfig:
    embedding_dim: int = 1024
    fusion_layers: int = 8
    fusion_heads: int = 16
    fusion_ff_dim: int = 4096
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
    action_channels: tuple[int, int, int] = (256, 512, 1024)
    action_mlp_hidden: int = 2048
    tactile_reference_multiple: float = 32.0

    def __post_init__(self) -> None:
        dimensions = (
            self.embedding_dim,
            self.fusion_layers,
            self.fusion_heads,
            self.fusion_ff_dim,
            self.action_pool_bins,
            self.time_embedding_dim,
            self.vision_batch_size,
            self.action_mlp_hidden,
        )
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in dimensions
        ):
            raise ValueError("CPM dimensions must be positive integers")
        if self.embedding_dim % self.fusion_heads or self.time_embedding_dim % 2:
            raise ValueError(
                "embedding_dim must divide into heads; time_embedding_dim must be even"
            )
        if (self.action_horizon, self.action_dim) != (
            30,
            27,
        ) or self.action_pool_bins > self.action_horizon:
            raise ValueError("This CPM uses 30-step, 27-dimensional actions")
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        if not isinstance(self.time_conditioning, bool):
            raise ValueError("time_conditioning must be boolean")
        if len(self.action_channels) != 3 or any(
            not isinstance(channel, int)
            or isinstance(channel, bool)
            or channel <= 0
            or channel % 8
            for channel in self.action_channels
        ):
            raise ValueError(
                "Use three positive action channel widths divisible by eight"
            )
        object.__setattr__(self, "action_channels", tuple(self.action_channels))
        if (
            not math.isfinite(self.tactile_reference_multiple)
            or self.tactile_reference_multiple <= 0
        ):
            raise ValueError("tactile_reference_multiple must be finite and positive")
        temperatures = (
            self.min_temperature,
            self.initial_temperature,
            self.max_temperature,
        )
        if (
            any(not math.isfinite(value) for value in temperatures)
            or not 0 < temperatures[0] <= temperatures[1] <= temperatures[2]
        ):
            raise ValueError("Invalid temperature bounds or initialization")


def _require_float_tensor(
    value: torch.Tensor, shape: tuple[int, ...], name: str
) -> None:
    if (
        not isinstance(value, torch.Tensor)
        or not value.is_floating_point()
        or tuple(value.shape) != shape
    ):
        raise ValueError(f"Expected floating {name} {shape}")
    if not torch.isfinite(value).all():
        raise ValueError(f"Nonfinite {name}")


class FrozenDinov2(nn.Module):
    """Load verified local DINOv2-Base weights, with no network or random fallback."""

    def __init__(self, directory: str | Path, *, batch_size: int = 16):
        super().__init__()
        from transformers import Dinov2Model  # noqa: PLC0415 - defer backbone dependency until model creation

        directory = Path(directory).expanduser().resolve(strict=True)
        manifest = json.loads((directory / "backbone_manifest.json").read_text())
        if (
            manifest["schema"] != "touchguide_public_backbone_v1"
            or manifest["repository"] != "facebook/dinov2-base"
            or manifest["revision"] != "f9e44c814b77203eaa57a6bdbbd535f21ede1415"
            or manifest["files"]["model.safetensors"]["sha256"]
            != "d73036b56966966d07975d696bde331762f37297e2f095de8cea0040c3aa0841"
        ):
            raise ValueError("Use the verified facebook/dinov2-base backbone")
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
        if any(
            information[key]
            for key in (
                "missing_keys",
                "unexpected_keys",
                "mismatched_keys",
                "error_msgs",
            )
        ):
            raise ValueError("DINOv2 checkpoint does not load exactly")
        if backbone.config.hidden_size != 768 or backbone.config.patch_size != 14:
            raise ValueError(
                "Expected DINOv2-Base with 768 features and 14-pixel patches"
            )
        self.backbone = backbone.requires_grad_(requires_grad=False).eval()
        self.feature_dim = backbone.config.hidden_size
        self.batch_size = batch_size
        self.register_buffer(
            "rgb_mean", torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
        )
        self.register_buffer(
            "rgb_std", torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
        )
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
        return self.forward_pixels(self.preprocess(rgb))

    def forward_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
        # Pixel values are already ImageNet-normalized. Force inputs deliberately
        # allow values outside the usual RGB range; never clamp or normalize twice.
        _require_float_tensor(
            pixels, (len(pixels), 3, 224, 224), "normalized DINO pixels"
        )
        self.backbone.eval()
        with torch.no_grad():
            values = [
                self.backbone(pixel_values=chunk).last_hidden_state
                for chunk in pixels.split(self.batch_size)
            ]
        # CLS plus the 16x16 patch grid. Positional interpolation is handled
        # by the official DINOv2 model when its training resolution differs.
        return torch.cat(values)


class ActionEncoder(nn.Module):
    """Temporal 1D CNN + ordered pooled MLP, with optional local time conditioning."""

    def __init__(self, config: CPMConfig):
        super().__init__()
        self.config = config
        first, second, third = config.action_channels
        self.temporal = nn.Sequential(
            nn.Conv1d(config.action_dim, first, kernel_size=5, padding=2),
            nn.GroupNorm(8, first),
            nn.SiLU(),
            nn.Conv1d(first, second, kernel_size=3, padding=1),
            nn.GroupNorm(8, second),
            nn.SiLU(),
            nn.Conv1d(second, third, kernel_size=3, padding=1),
            nn.GroupNorm(8, third),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(config.action_pool_bins),
        )
        self.projection = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(third * config.action_pool_bins, config.action_mlp_hidden),
            nn.SiLU(),
            nn.Linear(config.action_mlp_hidden, config.embedding_dim),
        )
        self.register_buffer(
            "time_frequencies", torch.logspace(0, 3, config.time_embedding_dim // 2)
        )
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
            nn.LayerNorm(config.embedding_dim),
            nn.Linear(config.embedding_dim, config.embedding_dim),
        )

    def forward(
        self, actions: torch.Tensor, flow_time: torch.Tensor | float
    ) -> torch.Tensor:
        size = len(actions)
        if size < 1:
            raise ValueError("An action encoder needs at least one candidate")
        _require_float_tensor(
            actions,
            (size, self.config.action_horizon, self.config.action_dim),
            "noisy_actions",
        )
        time = torch.as_tensor(flow_time, dtype=torch.float32, device=actions.device)
        if time.ndim == 0:
            time = time.expand(size)
        if (
            time.shape != (size,)
            or not torch.isfinite(time).all()
            or (time < 0).any()
            or (time > 1).any()
        ):
            raise ValueError("flow_time must be a scalar or [B] in [0,1]")
        features = self.projection(self.temporal(actions.transpose(1, 2)))
        if self.time_mlp is not None:
            phase = 2 * math.pi * time[:, None] * self.time_frequencies[None]
            features = features + self.time_mlp(
                torch.cat([phase.sin(), phase.cos()], dim=-1)
            )
        return functional.normalize(self.output(features).float(), dim=-1, eps=1e-8)


class CPMModel(nn.Module):
    """Produce normalized features, raw cosine scores and temperature-scaled logits."""

    uses_state = False
    observation_token_count = 456

    def __init__(
        self,
        config: CPMConfig,
        *,
        vision_directory: str | Path,
        tactile_rms: tuple[float, float, float],
    ):
        super().__init__()
        if len(tactile_rms) != 3 or any(
            not math.isfinite(value) or value <= 0 for value in tactile_rms
        ):
            raise ValueError(
                "Use three finite positive train-only tactile channel RMS values"
            )
        self.config = config
        self.vision = FrozenDinov2(
            vision_directory, batch_size=config.vision_batch_size
        )
        self.vision_projection = nn.Linear(
            self.vision.feature_dim, config.embedding_dim
        )
        self.tactile_projection = nn.Linear(
            self.vision.feature_dim, config.embedding_dim
        )
        self.tactile_preprocessor = TactileDinoPreprocessor(
            tactile_rms, reference_multiple=config.tactile_reference_multiple
        )
        self.modality_embedding = nn.Parameter(torch.empty(3, config.embedding_dim))
        self.finger_embedding = nn.Parameter(torch.empty(5, config.embedding_dim))
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
            layer,
            config.fusion_layers,
            norm=nn.LayerNorm(config.embedding_dim),
            enable_nested_tensor=False,
        )
        # TransformerEncoder clones one prototype. Independently initialize the
        # cloned layers rather than giving all layers identical starting weights.
        for block in self.fusion.layers:
            nn.init.xavier_uniform_(block.self_attn.in_proj_weight)
            nn.init.zeros_(block.self_attn.in_proj_bias)
            for linear in (block.self_attn.out_proj, block.linear1, block.linear2):
                nn.init.xavier_uniform_(linear.weight)
                nn.init.zeros_(linear.bias)
        self.observation_projection = nn.Linear(
            config.embedding_dim, config.embedding_dim
        )
        self.action_encoder = ActionEncoder(config)
        self.logit_scale = nn.Parameter(
            torch.tensor(math.log(1 / config.initial_temperature))
        )
        nn.init.normal_(self.modality_embedding, std=0.02)
        nn.init.normal_(self.finger_embedding, std=0.02)
        nn.init.normal_(self.observation_cls, std=0.02)

    @property
    def temperature(self) -> torch.Tensor:
        lower = math.log(1 / self.config.max_temperature)
        upper = math.log(1 / self.config.min_temperature)
        return self.logit_scale.clamp(lower, upper).exp().reciprocal()

    @staticmethod
    def pool_dino_tokens(features: torch.Tensor) -> torch.Tensor:
        """Keep each view CLS and 2x2-average its 16x16 spatial patch grid."""
        if features.ndim != 3 or features.shape[1] != 257:
            raise ValueError("Expected DINO CLS plus a 16x16 patch grid")
        patches = (
            features[:, 1:]
            .transpose(1, 2)
            .reshape(len(features), features.shape[2], 16, 16)
        )
        patches = (
            functional.avg_pool2d(patches, kernel_size=2, stride=2)
            .flatten(2)
            .transpose(1, 2)
        )
        return torch.cat([features[:, :1], patches], dim=1)

    def observation_tokens(
        self, head: torch.Tensor, wrist: torch.Tensor, tactile_map_raw: torch.Tensor
    ) -> torch.Tensor:
        size = len(tactile_map_raw)
        if size < 1 or len(head) != size or len(wrist) != size:
            raise ValueError("Current camera and tactile batch sizes must match")
        _require_float_tensor(head, (size, 3, 224, 224), "head")
        _require_float_tensor(wrist, (size, 3, 224, 224), "wrist")
        rgb_pixels = self.vision.preprocess(torch.cat([head, wrist]))
        touch_pixels = self.tactile_preprocessor(tactile_map_raw)
        # Exactly one shared frozen backbone; touch flattening is batch-major,
        # finger-minor. Camera blocks remain head batch then wrist batch.
        features = self.pool_dino_tokens(
            self.vision.forward_pixels(torch.cat([rgb_pixels, touch_pixels]))
        )
        head_features, wrist_features, touch_features = features.split(
            [size, size, size * 5]
        )
        head_tokens = self.vision_projection(head_features) + self.modality_embedding[0]
        wrist_tokens = (
            self.vision_projection(wrist_features) + self.modality_embedding[1]
        )
        touch_tokens = self.tactile_projection(touch_features).reshape(
            size, 5, 65, self.config.embedding_dim
        )
        touch_tokens = (
            touch_tokens
            + self.modality_embedding[2]
            + self.finger_embedding[None, :, None, :]
        )
        tokens = torch.cat(
            [
                self.observation_cls.expand(size, -1, -1),
                head_tokens,
                wrist_tokens,
                touch_tokens.flatten(1, 2),
            ],
            dim=1,
        )
        if tokens.shape[1] != 456:
            raise RuntimeError("Expected two cameras, five fingers and one fusion CLS")
        return tokens

    def encode_observation(
        self, head: torch.Tensor, wrist: torch.Tensor, tactile_map_raw: torch.Tensor
    ) -> torch.Tensor:
        tokens = self.observation_tokens(head, wrist, tactile_map_raw)
        fused = self.fusion(tokens)[:, 0]
        return functional.normalize(
            self.observation_projection(fused).float(), dim=-1, eps=1e-8
        )

    def encode_observation_batch(self, batch: dict[str, Any]) -> torch.Tensor:
        return self.encode_observation(
            batch["head"], batch["wrist"], batch["tactile_map_raw"]
        )

    def score_actions(
        self,
        observation_embedding: torch.Tensor,
        noisy_actions: torch.Tensor,
        flow_time: torch.Tensor | float,
    ) -> torch.Tensor:
        """Raw [observations,candidates] cosine, without training temperature or softmax."""
        _require_float_tensor(
            observation_embedding,
            (len(observation_embedding), self.config.embedding_dim),
            "observation_embedding",
        )
        if not torch.allclose(
            observation_embedding.norm(dim=-1),
            torch.ones(len(observation_embedding), device=observation_embedding.device),
            atol=1e-4,
            rtol=1e-4,
        ):
            raise ValueError("Cache the normalized encode_observation output")
        action = self.action_encoder(noisy_actions, flow_time)
        with torch.autocast(
            device_type=observation_embedding.device.type, enabled=False
        ):
            return observation_embedding.float() @ action.float().T

    def paired_scores(
        self,
        observation_embedding: torch.Tensor,
        noisy_actions: torch.Tensor,
        flow_time: torch.Tensor | float,
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
            raise ValueError(
                "CPM requires an augmented batch: RGB, taxels, noisy_actions and flow_time"
            )
        if "negative_actions" in batch or "noisy_negative_actions" in batch:
            raise ValueError(
                "Use shared action columns rather than legacy negative tensors"
            )
        if len(batch["head"]) != len(batch["noisy_actions"]):
            raise ValueError(
                "Contrastive forward requires one matched action column per observation"
            )
        time = torch.as_tensor(batch["flow_time"])
        if time.ndim != 0:
            raise ValueError("Contrastive batches must share a scalar noise time")
        observation = self.encode_observation_batch(batch)
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
        trainable = sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
        return {"total": total, "trainable": trainable, "frozen": total - trainable}

    @classmethod
    def config_arguments(cls, root: str | Path) -> dict[str, Any]:
        root = Path(root).expanduser().resolve(strict=True)
        settings = json.loads((root / "config/cpm.json").read_text())
        if settings["schema"] != "openpi_touchguide_cpm_config_v2":
            raise ValueError(
                "Use the upgraded DINO CPM v2 configuration; old CNN checkpoints are incompatible"
            )
        vision_manifest = (
            Path(settings["vision"]["directory"]) / "backbone_manifest.json"
        )
        if (
            hashlib.sha256(vision_manifest.read_bytes()).hexdigest()
            != settings["vision"]["manifest_sha256"]
        ):
            raise ValueError("The visual dependency changed after CPM preparation")
        scale_path = root / settings["tactile"]["statistics_file"]
        if (
            hashlib.sha256(scale_path.read_bytes()).hexdigest()
            != settings["tactile"]["statistics_sha256"]
        ):
            raise ValueError("Tactile statistics changed after CPM preparation")
        stats = json.loads(scale_path.read_text())
        if settings["tactile"]["normalization"] != "signed_asinh_train_rms_no_clip":
            raise ValueError("Use the configured signed tactile asinh transform")
        if (
            settings["tactile"]["trainable"]
            or not settings["vision"]["frozen"]
            or settings["tactile"]["shared_backbone"] != "vision"
            or settings["observation_tokens"] != cls.observation_token_count
        ):
            raise ValueError(
                f"DINO CPM requires one frozen shared backbone and {cls.observation_token_count} base observation tokens"
            )
        return {
            "config": CPMConfig(**settings["model_kwargs"]),
            "vision_directory": settings["vision"]["directory"],
            "tactile_rms": tuple(stats["rms"]),
        }

    @classmethod
    def from_config(cls, root: str | Path):
        return cls(**cls.config_arguments(root))


def load_cpm_model(root: str | Path, variant: str = "cpm") -> CPMModel:
    """Keep the prepared baseline entry intact; record state-CPM in the run contract."""
    if variant == "cpm":
        return CPMModel.from_config(root)
    if variant == "state-cpm":
        from openpi.touchguide.state_model import StateCPMModel  # noqa: PLC0415

        return StateCPMModel.from_config(root)
    if variant == "recovery-state-cpm":
        from openpi.touchguide.recovery_model import RecoveryStateCPM  # noqa: PLC0415

        return RecoveryStateCPM.from_config(root)
    raise ValueError(f"Unknown CPM variant: {variant}")


def encode_observation_batch(model, batch: dict[str, Any]) -> torch.Tensor:
    """Support the original three-input interface and the optional current state."""
    if getattr(model, "uses_state", False):
        return model.encode_observation_batch(batch)
    return model.encode_observation(
        batch["head"], batch["wrist"], batch["tactile_map_raw"]
    )
