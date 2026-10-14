"""Late-step TouchGuide guidance for a frozen pi0.5 flow sampler.

Inputs and outputs are in the base policy's normalized action coordinates.
This module never differentiates through pi0.5 or updates either model.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import math
from typing import Any

import torch
from torch.nn import functional


@dataclass(frozen=True)
class GuidanceConfig:
    num_steps: int = 10
    guidance_steps: int = 3
    scale: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.num_steps, int) or isinstance(self.num_steps, bool) or self.num_steps < 1:
            raise ValueError("num_steps must be a positive integer")
        if (
            not isinstance(self.guidance_steps, int)
            or isinstance(self.guidance_steps, bool)
            or not 0 <= self.guidance_steps < self.num_steps
        ):
            raise ValueError("Guide only late steps; guidance_steps must be in [0, num_steps-1] to exclude t=1")
        if not math.isfinite(self.scale) or self.scale < 0:
            raise ValueError("Guidance scale must be finite and nonnegative")

    @property
    def enabled(self) -> bool:
        return self.scale > 0 and self.guidance_steps > 0


@dataclass(frozen=True)
class PrefixCache:
    state: torch.Tensor
    pad_masks: torch.Tensor
    key_values: Any


@dataclass(frozen=True)
class SamplingResult:
    actions: torch.Tensor
    diagnostics: tuple[dict[str, Any], ...]
    # Optional small traces for offline comparison, never computation graphs.
    trajectory: torch.Tensor | None = None
    guidance_updates: torch.Tensor | None = None


@torch.no_grad()
def prepare_pi05_prefix(base: Any, observation: Any) -> PrefixCache:
    """Same preprocessing, attention masks and KV cache as PI0Pytorch.sample_actions."""
    from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks  # noqa: PLC0415

    images, masks, tokens, token_masks, state = base._preprocess_observation(observation, train=False)  # noqa: SLF001
    prefix, pad, attention = base.embed_prefix(images, masks, tokens, token_masks)
    mask4d = base._prepare_attention_masks_4d(make_att_2d_masks(pad, attention))  # noqa: SLF001
    base.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
    _, cache = base.paligemma_with_expert.forward(
        attention_mask=mask4d,
        position_ids=torch.cumsum(pad, dim=1) - 1,
        past_key_values=None,
        inputs_embeds=[prefix, None],
        use_cache=True,
    )
    return PrefixCache(state, pad, cache)


class TouchGuideSampler:
    """Cache observations once, then guide the last Euler steps using raw cosine.

    ``sample`` accepts the already transformed PyTorch pi0.5 Observation (NCHW
    float RGB in [-1,1], quantile-normalized state and tokenized prompt), plus
    same-time right-hand maps [B,5,3,7,5] in newtons. Finger order is thumb,
    index, middle, ring, pinky; channels are normal, tangent0, tangent1.
    ``sample_cached`` is also exposed for fixed-cache numerical comparisons.
    """

    def __init__(self, base: Any, cpm: Any, config: GuidanceConfig, *, precision: str = "bfloat16"):
        if precision not in ("bfloat16", "float32"):
            raise ValueError("precision must be bfloat16 or float32")
        if (base.config.action_horizon, base.config.action_dim) != (30, 32):
            raise ValueError("This sampler requires the 30-step, 32-dimensional pi0.5 base")
        if (cpm.config.action_horizon, cpm.config.action_dim) != (30, 27):
            raise ValueError("CPM must score 30-step, 27-dimensional physical actions")
        for name, model in (("pi0.5", base), ("CPM", cpm)):
            if model.training or any(parameter.requires_grad for parameter in model.parameters()):
                raise ValueError(f"Load {name} in eval mode with all parameters frozen")
        if next(base.parameters()).device != next(cpm.parameters()).device:
            raise ValueError("pi0.5 and CPM must be on the same device")
        self.base = base
        self.cpm = cpm
        self.config = config
        self.precision = precision

    def _autocast(self, device: torch.device):
        if device.type == "cuda" and self.precision == "bfloat16":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return nullcontext()

    def sample(
        self,
        observation: Any,
        tactile_map_raw: torch.Tensor | None = None,
        *,
        noise: torch.Tensor | None = None,
        record_diagnostics: bool = True,
        record_trace: bool = False,
    ) -> SamplingResult:
        device = observation.state.device
        size = observation.state.shape[0]
        if noise is None:
            noise = self.base.sample_noise((size, 30, 32), device)
        with torch.no_grad(), self._autocast(device):
            prefix = prepare_pi05_prefix(self.base, observation)
            embedding = None
            if self.config.enabled:
                if tactile_map_raw is None or tuple(tactile_map_raw.shape) != (size, 5, 3, 7, 5):
                    raise ValueError("Guidance requires same-time right tactile maps [B,5,3,7,5]")
                state_kwargs = (
                    {"state_normalized": observation.state[:, :27]}
                    if getattr(self.cpm, "uses_state", False)
                    else {}
                )
                embedding = self.cpm.encode_observation(
                    observation.images["base_0_rgb"],
                    observation.images["right_wrist_0_rgb"],
                    tactile_map_raw,
                    **state_kwargs,
                ).detach()
        return self.sample_cached(
            prefix, embedding, noise, record_diagnostics=record_diagnostics, record_trace=record_trace
        )

    def score_gradient(
        self, observation_embedding: torch.Tensor, actions: torch.Tensor, flow_time: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-example cosine gradient, padded with five exact zeros (no batch averaging)."""
        if torch.is_inference_mode_enabled():
            raise RuntimeError("Guidance needs action gradients; use torch.no_grad(), not torch.inference_mode()")
        # Tiny guidance shifts can be smaller than BF16 input quantization.
        # Keep the differentiable action branch FP32 even when pi0.5 and the
        # cached observation encoder use BF16 autocast.
        with torch.enable_grad(), torch.autocast(device_type=actions.device.type, enabled=False):
            physical = actions[..., :27].detach().float().clone().requires_grad_(requires_grad=True)
            scores = self.cpm.paired_scores(observation_embedding.detach(), physical, flow_time)
            if tuple(scores.shape) != (actions.shape[0],):
                raise ValueError("CPM must return one raw cosine score per candidate action chunk")
            # Sum independent scores, so changing batch size does not rescale guidance.
            (gradient,) = torch.autograd.grad(scores.sum(), physical)
        if not torch.isfinite(scores).all() or not torch.isfinite(gradient).all():
            raise FloatingPointError("Nonfinite CPM score or action gradient")
        return functional.pad(gradient.detach(), (0, 5)), scores.detach()

    @torch.no_grad()
    def sample_cached(
        self,
        prefix: PrefixCache,
        observation_embedding: torch.Tensor | None,
        noise: torch.Tensor,
        *,
        record_diagnostics: bool = True,
        record_trace: bool = False,
    ) -> SamplingResult:
        if noise.ndim != 3 or tuple(noise.shape[1:]) != (30, 32) or noise.dtype != torch.float32:
            raise ValueError("Use float32 Gaussian noise [B,30,32] in normalized action coordinates")
        if not torch.isfinite(noise).all() or noise.device != prefix.state.device:
            raise ValueError("Noise must be finite and on the observation's device")
        if noise.shape[0] != prefix.state.shape[0]:
            raise ValueError("Observation and action batch sizes must match")
        if self.config.enabled and (observation_embedding is None or observation_embedding.shape[0] != noise.shape[0]):
            raise ValueError("Guidance requires one cached CPM observation per action chunk")
        x = noise.detach().clone()
        dt = torch.tensor(-1.0 / self.config.num_steps, dtype=torch.float32, device=x.device)
        time = torch.tensor(1.0, dtype=torch.float32, device=x.device)
        diagnostics = []
        trajectory = [x.clone()] if record_trace else []
        updates = []
        with self._autocast(x.device):
            for step in range(self.config.num_steps):
                # Preserve the stock FP32 time recurrence and Euler arithmetic.
                velocity = self.base.denoise_step(
                    prefix.state, prefix.pad_masks, prefix.key_values, x, time.expand(len(x))
                )
                active = self.config.enabled and step >= self.config.num_steps - self.config.guidance_steps
                row = {"step": step, "flow_time": float(time), "guided": active}
                shift = torch.zeros_like(x)
                if active:
                    gradient, scores = self.score_gradient(observation_embedding, x, time)
                    # Flow integrates from t=1 to t=0. Convert the velocity
                    # correction into a separate positive score-gradient action step.
                    correction = self.config.scale * time / (1.0 - time) * gradient
                    shift = -dt * correction
                    if record_diagnostics:
                        with torch.autocast(device_type=x.device.type, enabled=False):
                            local_scores = self.cpm.paired_scores(observation_embedding, (x + shift)[..., :27], time)
                        row.update(
                            score_before=scores.float().cpu().tolist(),
                            score_after_guidance_only=local_scores.float().cpu().tolist(),
                            gradient_rms=float(gradient[..., :27].square().mean().sqrt()),
                            velocity_correction_rms=float(correction[..., :27].square().mean().sqrt()),
                            latent_guidance_shift_rms=float(shift[..., :27].square().mean().sqrt()),
                            padded_guidance_max_abs=float(shift[..., 27:].abs().max()),
                        )
                # Match the stock base update's dtype and rounding before adding
                # FP32 guidance. Subtracting guidance from velocity first would
                # promote a BF16 velocity and change the base Euler product.
                x = x + dt * velocity
                if active:
                    x = x + shift
                if not torch.isfinite(x).all():
                    raise FloatingPointError(f"Nonfinite actions at denoising step {step}")
                if record_diagnostics:
                    diagnostics.append(row)
                if record_trace:
                    trajectory.append(x.clone())
                    updates.append(shift.clone())
                time += dt
        return SamplingResult(
            actions=x.detach(),
            diagnostics=tuple(diagnostics),
            trajectory=torch.stack(trajectory) if record_trace else None,
            guidance_updates=torch.stack(updates) if record_trace else None,
        )
