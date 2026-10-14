"""Single-step full-Jacobian forward guidance, not an unrolled sampling gradient.

Score the estimated clean action at CPM noise time zero. Backpropagate through
the current frozen action expert to x_t, mask padded coordinates, and apply a
late-step flow correction. The t/(1-t) schedule is a local flow adaptation of
Universal Guidance, not a claim of exact conditional posterior sampling.
"""

from dataclasses import dataclass

import torch

from guidance_common.scorer import CPMScorer
from openpi.touchguide.sampler import PrefixCache
from openpi.touchguide.sampler import SamplingResult
from openpi.touchguide.sampler import TouchGuideSampler
from tac_forward_guidance.config import ForwardGuidanceConfig


@dataclass(frozen=True)
class ForwardGradient:
    velocity: torch.Tensor
    gradient: torch.Tensor
    clean_gradient: torch.Tensor
    scores: torch.Tensor
    clean_actions: torch.Tensor
    raw_padded_gradient_max_abs: float


class ForwardGuidanceSampler(TouchGuideSampler):
    """Reuse prefix/observation preparation but replace the entire gradient path."""

    def __init__(self, base, cpm, config: ForwardGuidanceConfig, *, precision="bfloat16"):
        super().__init__(base, cpm, config, precision=precision)
        self.scorer = CPMScorer(cpm)

    def forward_gradient(self, prefix: PrefixCache, embedding, actions, flow_time) -> ForwardGradient:
        if torch.is_inference_mode_enabled():
            raise RuntimeError("Forward guidance needs input gradients; do not use torch.inference_mode()")
        time = torch.as_tensor(flow_time, dtype=torch.float32, device=actions.device)
        if time.ndim != 0 or not torch.isfinite(time) or not 0 < float(time) < 1:
            raise ValueError("Use a scalar flow time strictly between zero and one")
        with torch.enable_grad():
            # Detach only the previous sampling trajectory. Do NOT detach the
            # current velocity or clean estimate in the default full mode.
            leaf = actions.detach().float().clone().requires_grad_(requires_grad=True)
            with self._autocast(actions.device):
                if self.config.gradient_mode == "stop_velocity":
                    with torch.no_grad():
                        velocity = self.base.denoise_step(
                            prefix.state, prefix.pad_masks, prefix.key_values, leaf, time.expand(len(leaf))
                        )
                else:
                    velocity = self.base.denoise_step(
                        prefix.state, prefix.pad_masks, prefix.key_values, leaf, time.expand(len(leaf))
                    )
                    if not velocity.requires_grad:
                        raise RuntimeError(
                            "Action expert severed its input graph; full forward guidance cannot proceed"
                        )
            # CPM is evaluated in FP32 even inside the caller's BF16 autocast.
            with torch.autocast(device_type=actions.device.type, enabled=False):
                clean = leaf - time * velocity.float()
                scores = self.scorer.score(embedding, clean[..., :27], noise_time=0.0)
                gradient, clean_gradient = torch.autograd.grad(
                    scores.sum(), (leaf, clean), create_graph=False, retain_graph=False
                )
        if not torch.isfinite(gradient).all() or not torch.isfinite(velocity).all():
            raise FloatingPointError("Nonfinite action-expert velocity or forward gradient")
        raw_padded = float(gradient[..., 27:].abs().max())
        gradient = gradient.detach().clone()
        gradient[..., 27:] = 0
        return ForwardGradient(
            velocity.detach(), gradient, clean_gradient.detach(), scores.detach(), clean.detach(), raw_padded
        )

    @torch.no_grad()
    def clean_scores(self, prefix, embedding, actions, flow_time):
        with self._autocast(actions.device):
            velocity = self.base.denoise_step(
                prefix.state, prefix.pad_masks, prefix.key_values, actions, flow_time.expand(len(actions))
            )
        return self.scorer.score(embedding, (actions - flow_time * velocity.float())[..., :27], 0.0)

    def cap_update(self, shift):
        rms = shift[..., :27].square().mean(dim=(1, 2)).sqrt()
        factors = torch.ones_like(rms)
        if self.config.max_update_rms is not None:
            factors = (self.config.max_update_rms / rms.clamp_min(1e-12)).clamp(max=1.0)
        return shift * factors[:, None, None], factors

    @torch.no_grad()
    def sample_cached(self, prefix, observation_embedding, noise, *, record_diagnostics=True, record_trace=False):
        if not self.config.enabled:
            # Identical arithmetic, dtype recurrence and CPM-free baseline path.
            return super().sample_cached(
                prefix, None, noise, record_diagnostics=record_diagnostics, record_trace=record_trace
            )
        if torch.is_inference_mode_enabled():
            raise RuntimeError("Forward guidance needs input gradients; do not use torch.inference_mode()")
        if noise.ndim != 3 or tuple(noise.shape[1:]) != (30, 32) or noise.dtype != torch.float32:
            raise ValueError("Use float32 noise [B,30,32]")
        if not torch.isfinite(noise).all() or noise.device != prefix.state.device:
            raise ValueError("Noise must be finite and on the prefix device")
        if len(noise) != len(prefix.state):
            raise ValueError("Observation and action batch sizes must match")
        if observation_embedding is None or len(observation_embedding) != len(noise):
            raise ValueError("Forward guidance needs one cached observation per action chunk")
        x = noise.detach().clone()
        dt = torch.tensor(-1.0 / self.config.num_steps, dtype=torch.float32, device=x.device)
        time = torch.tensor(1.0, dtype=torch.float32, device=x.device)
        diagnostics, updates = [], []
        trajectory = [x.clone()] if record_trace else []
        for step in range(self.config.num_steps):
            active = step >= self.config.num_steps - self.config.guidance_steps
            row = {"step": step, "flow_time": float(time), "guided": active}
            shift = torch.zeros_like(x)
            if active:
                result = self.forward_gradient(prefix, observation_embedding, x, time)
                velocity = result.velocity
                shift, factors = self.cap_update(-dt * self.config.scale * time / (1 - time) * result.gradient)
                if not torch.isfinite(shift).all():
                    raise FloatingPointError("Nonfinite forward-guidance update")
                if record_diagnostics:
                    g = result.gradient[..., :27]
                    clean_g = result.clean_gradient[..., :27]
                    row.update(
                        cpm_score_time=0.0,
                        gradient_mode=self.config.gradient_mode,
                        score_before=result.scores.cpu().tolist(),
                        gradient_rms=float(g.square().mean().sqrt()),
                        clean_gradient_rms=float(clean_g.square().mean().sqrt()),
                        jacobian_contribution_rms=float((g - clean_g).square().mean().sqrt()),
                        raw_padded_gradient_max_abs=result.raw_padded_gradient_max_abs,
                        latent_guidance_shift_rms=float(shift[..., :27].square().mean().sqrt()),
                        per_sample_shift_rms=shift[..., :27].square().mean(dim=(1, 2)).sqrt().cpu().tolist(),
                        update_cap_factors=factors.cpu().tolist(),
                        capped_samples=int((factors < 1).sum()),
                        padded_guidance_max_abs=float(shift[..., 27:].abs().max()),
                    )
                    if self.config.local_score_probe:
                        # Recompute v(x+shift), rather than mislabelling a
                        # constant-velocity approximation as the forward score.
                        row["score_after_guidance_only"] = (
                            self.clean_scores(prefix, observation_embedding, x + shift, time).cpu().tolist()
                        )
            else:
                with self._autocast(x.device):
                    velocity = self.base.denoise_step(
                        prefix.state, prefix.pad_masks, prefix.key_values, x, time.expand(len(x))
                    )
            # Keep stock BF16 Euler multiplication before the separate FP32
            # correction. A zero score gradient must reproduce stock exactly.
            x = x + dt * velocity
            if active:
                x = x + shift
            x = x.detach()
            if not torch.isfinite(x).all():
                raise FloatingPointError(f"Nonfinite actions at flow step {step}")
            if record_diagnostics:
                diagnostics.append(row)
            if record_trace:
                trajectory.append(x.clone())
                updates.append(shift.clone())
            time += dt
        return SamplingResult(
            actions=x,
            diagnostics=tuple(diagnostics),
            trajectory=torch.stack(trajectory) if record_trace else None,
            guidance_updates=torch.stack(updates) if record_trace else None,
        )
