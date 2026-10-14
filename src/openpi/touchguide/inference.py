"""Load the trusted CPM checkpoint and converted 20k pi0.5 for inference only."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from openpi.touchguide.checkpoint import CHECKPOINT_FORMAT
from openpi.touchguide.model import load_cpm_model
from openpi.touchguide.sampler import GuidanceConfig
from openpi.touchguide.sampler import TouchGuideSampler


@dataclass(frozen=True)
class ActionCoordinates:
    """The original checkpoint quantiles: arm delta 7, hand absolute target 20."""

    stats: dict[str, Any]

    def __post_init__(self) -> None:
        for key in ("state", "actions"):
            low = torch.tensor(self.stats[key]["q01"], dtype=torch.float64)
            high = torch.tensor(self.stats[key]["q99"], dtype=torch.float64)
            if low.shape != (27,) or high.shape != (27,) or not torch.isfinite(low).all():
                raise ValueError("Expected 27 finite state/action quantiles")
            if not torch.isfinite(high).all() or torch.any(high < low):
                raise ValueError("Invalid state/action quantile range")

    def _quantiles(self, key: str, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return tuple(
            torch.as_tensor(self.stats[key][name], dtype=reference.dtype, device=reference.device)
            for name in ("q01", "q99")
        )

    def normalize_state(self, state_raw: torch.Tensor) -> torch.Tensor:
        """Normalize 27 dimensions before state tokenization and padding to 32."""
        if state_raw.shape[-1] != 27:
            raise ValueError("Use the right-arm/right-hand 27-dimensional raw state")
        low, high = self._quantiles("state", state_raw)
        return (state_raw - low) / (high - low + 1e-6) * 2.0 - 1.0

    def action_targets(self, actions: torch.Tensor, state_raw: torch.Tensor) -> torch.Tensor:
        """Unnormalize then restore arm absolute targets; discard the five padded dims."""
        if actions.ndim != 3 or tuple(actions.shape[1:]) != (30, 32):
            raise ValueError("Expected normalized pi0.5 actions [B,30,32]")
        if state_raw.shape != (actions.shape[0], 27):
            raise ValueError("Expected current raw state [B,27]")
        low, high = self._quantiles("actions", actions)
        targets = (actions[..., :27] + 1.0) / 2.0 * (high - low + 1e-6) + low
        targets[..., :7] += state_raw.to(device=actions.device, dtype=actions.dtype)[:, None, :7]
        return targets


@dataclass(frozen=True)
class LoadedTouchGuide:
    sampler: TouchGuideSampler
    coordinates: ActionCoordinates
    metadata: dict[str, Any]


def load_touchguide(
    *,
    base_checkpoint: str | Path,
    cpm_checkpoint: str | Path,
    data_root: str | Path,
    guidance: GuidanceConfig | None = None,
    device: str | torch.device = "cuda",
    precision: str = "bfloat16",
) -> LoadedTouchGuide:
    """Use model weights only; never restore the 8-GPU trainer, optimizer or RNGs.

    The CPM file must be this project's own trusted torch.save checkpoint.
    mmap keeps unused training tensors on CPU without transferring them to GPU.
    Config/statistics hashes are checked against the saved training contract.
    """
    guidance = GuidanceConfig() if guidance is None else guidance
    base_path = Path(base_checkpoint).expanduser().resolve(strict=True)
    cpm_path = Path(cpm_checkpoint).expanduser().resolve(strict=True)
    root = Path(data_root).expanduser().resolve(strict=True)
    device = torch.device(device)
    if precision not in ("bfloat16", "float32"):
        raise ValueError("precision must be bfloat16 or float32")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the selected Python environment")
    payload = torch.load(cpm_path, map_location="cpu", weights_only=False, mmap=True)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("Use the project's trained CPM checkpoint format")
    contract = payload["contract"]
    model_variant = contract.get("model_variant", "cpm")
    dependencies = ["config/cpm.json", "meta/tactile_scale.json", "meta/action_norm_stats.json"]
    if model_variant == "recovery-state-cpm":
        settings = json.loads((root / "config/cpm.json").read_text())
        dependencies += [settings["recovery"][key]["file"] for key in ("normalization", "negative_policy")]
    for name in dependencies:
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if actual != contract["data_sha256"].get(name):
            raise ValueError(f"CPM inference dependency differs from its training checkpoint: {name}")
    if guidance.num_steps != contract["noise"]["num_flow_steps"]:
        raise ValueError("Use the denoising grid used to train this CPM (10 steps)")
    settings = json.loads((base_path / "config.json").read_text())
    manifest = json.loads((root / "meta/manifest.json").read_text())
    expected_source = Path(manifest["sources"]["base_checkpoint"]).resolve()
    if Path(settings["source_checkpoint"]).resolve() != expected_source:
        raise ValueError("pi0.5 checkpoint must match the CPM dataset's 20000-step source")
    expected = {
        "pi05": True,
        "action_horizon": 30,
        "action_dim": 32,
        "max_token_len": 200,
        "discrete_state_input": True,
    }
    if any(settings.get(key) != value for key, value in expected.items()):
        raise ValueError("Converted pi0.5 model configuration does not match the poker CPM")
    base_stats = json.loads((base_path / "assets/normalizer/norm_stats.json").read_text())["norm_stats"]
    cpm_stats = json.loads((root / "meta/action_norm_stats.json").read_text())["norm_stats"]
    if base_stats != cpm_stats:
        raise ValueError("pi0.5 and CPM must use exactly the same original normalization statistics")
    coordinates = ActionCoordinates(base_stats)
    cpm = load_cpm_model(root, model_variant)
    cpm.load_state_dict(payload["model_state"], strict=True)
    progress = {key: payload["progress"].get(key) for key in ("epoch", "global_step", "best_val", "best_step")}
    del payload
    cpm.eval().requires_grad_(requires_grad=False).to(device)

    import safetensors.torch  # noqa: PLC0415

    from openpi.models.pi0_config import Pi0Config  # noqa: PLC0415
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch  # noqa: PLC0415

    config = Pi0Config(
        pi05=True,
        action_dim=32,
        action_horizon=30,
        max_token_len=200,
        discrete_state_input=True,
        paligemma_variant=settings["paligemma_variant"],
        action_expert_variant=settings["action_expert_variant"],
        dtype=precision,
        pytorch_compile_mode=None,
    )
    base = PI0Pytorch(config)
    safetensors.torch.load_model(base, str(base_path / "model.safetensors"), strict=True)
    # Preserve OpenPI's selective BF16 policy; norms and action projections stay FP32.
    base.paligemma_with_expert.to_bfloat16_for_selected_params(precision)
    base.eval().requires_grad_(requires_grad=False).to(device)
    return LoadedTouchGuide(
        sampler=TouchGuideSampler(base, cpm, guidance, precision=precision),
        coordinates=coordinates,
        metadata={
            "base_checkpoint": str(base_path),
            "base_source": str(expected_source),
            "cpm_checkpoint": str(cpm_path),
            "cpm_variant": model_variant,
            "cpm_progress": progress,
            "data_root": str(root),
            "precision": precision,
            "device": str(device),
            "normalization_verified": True,
            "training_dependencies_verified": True,
            "verified_dependency_files": dependencies,
            "oa_action_coordinates": "original_pi05_to_scorer_affine"
            if model_variant == "recovery-state-cpm"
            else "original_pi05",
            "base_parameters": sum(parameter.numel() for parameter in base.parameters()),
            "cpm_parameters": sum(parameter.numel() for parameter in cpm.parameters()),
            "guidance_score": "raw_cosine_without_temperature_or_softmax",
            "cpm_action_scoring_precision": "float32",
        },
    )
