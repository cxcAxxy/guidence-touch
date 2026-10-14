"""Frozen pi0.5 plus a late-flow tactile velocity expert for poker draw.

The 22 slots are [left: history, five F6, five map] followed by the same
right-hand slots. The current dataset has only right-hand tactile signals, so
the left 11 slots are masked. The trained F6 VQ-VAE supplies the history code;
the trained map encoder supplies five spatial tokens. Current F6 is projected
per fingertip, as in T-Rex.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn

from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
from openpi.tactile_vqvae.models.tactile_map_encoder import TactileMapEncoder
from openpi.tactile_vqvae.models.tactile_vqvae import TactileVQVAE
from openpi.tactile_vqvae.models.tactile_vqvae import TactileVQVAEConfig


class TactileFlowCache(NamedTuple):
    """Frozen visual context and the action latent at flow time 0.4."""

    state: torch.Tensor
    prefix_pad_masks: torch.Tensor
    past_key_values: object
    x_split: torch.Tensor


class TactileBlock(nn.Module):
    def __init__(self, width: int = 1024, heads: int = 8) -> None:
        super().__init__()
        self.action_norm = nn.LayerNorm(width)
        self.tactile_norm = nn.LayerNorm(width)
        self.cross_attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.self_norm = nn.LayerNorm(width)
        self.self_attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.ff_norm = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width))

    def forward(self, action: torch.Tensor, tactile: torch.Tensor, invalid: torch.Tensor) -> torch.Tensor:
        query = self.action_norm(action)
        action = action + self.cross_attention(
            query, self.tactile_norm(tactile), self.tactile_norm(tactile),
            key_padding_mask=invalid, need_weights=False,
        )[0]
        query = self.self_norm(action)
        action = action + self.self_attention(query, query, query, need_weights=False)[0]
        return action + self.ff(self.ff_norm(action))


class TactileVelocityExpert(nn.Module):
    """A compact trainable tactile expert over frozen pi0.5 action features."""

    def __init__(self, f6_config: dict, codebook_size: int, width: int = 1024,
                 layers: int = 2, action_dim: int = 32) -> None:
        super().__init__()
        self.f6_vqvae = TactileVQVAE(TactileVQVAEConfig.from_dict(f6_config))
        self.map_encoder = TactileMapEncoder(token_dim=256)
        self.history_code = nn.Embedding(codebook_size, width)
        self.current_f6 = nn.Sequential(nn.Linear(6, width), nn.SiLU(), nn.Linear(width, width))
        self.map_projection = nn.Linear(256, width)
        self.slot_embedding = nn.Parameter(torch.randn(22, width) * 0.02)
        self.blocks = nn.ModuleList(TactileBlock(width) for _ in range(layers))
        self.output_norm = nn.LayerNorm(width)
        self.velocity_delta = nn.Linear(width, action_dim)
        nn.init.zeros_(self.velocity_delta.weight)
        nn.init.zeros_(self.velocity_delta.bias)
        self._freeze_encoders()

    def _freeze_encoders(self) -> None:
        for encoder in (self.f6_vqvae, self.map_encoder):
            encoder.eval()
            encoder.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self._freeze_encoders()
        return self

    def load_encoder_weights(self, f6_checkpoint: str, map_checkpoint: str) -> None:
        f6 = torch.load(f6_checkpoint, map_location="cpu", weights_only=False)
        maps = torch.load(map_checkpoint, map_location="cpu", weights_only=False)
        if f6["config"]["granularity"] != "hand" or f6["config"]["codebook_size"] != self.history_code.num_embeddings:
            raise ValueError("F6 checkpoint configuration does not match tactile expert")
        if maps["config"]["token_dim"] != 256 or maps["config"]["hand_mode"] != "right":
            raise ValueError("Expected a 256-D right-hand tactile map encoder")
        self.f6_vqvae.load_state_dict(f6["model_state"], strict=True)
        self.map_encoder.load_state_dict(maps["encoder_state"], strict=True)
        self._freeze_encoders()

    def tactile_tokens(self, history: torch.Tensor, current: torch.Tensor,
                       maps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if history.shape[1:] != (16, 5, 6) or current.shape[1:] != (5, 6) or maps.shape[1:] != (5, 3, 7, 5):
            raise ValueError("Expected right-hand F6 [B,16,5,6]/[B,5,6] and maps [B,5,3,7,5]")
        with torch.no_grad():
            indices = self.f6_vqvae.encode(history.float())
            map_features = self.map_encoder(maps.float())
        right = torch.cat([
            self.history_code(indices).unsqueeze(1),
            self.current_f6(current.float()),
            self.map_projection(map_features.float()),
        ], dim=1)
        left = torch.zeros_like(right)
        tokens = torch.cat([left, right], dim=1) + self.slot_embedding.unsqueeze(0)
        invalid = torch.zeros(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        invalid[:, :11] = True
        return tokens, invalid

    def forward(self, action_hidden: torch.Tensor, base_velocity: torch.Tensor,
                tactile_tokens: torch.Tensor, invalid: torch.Tensor) -> torch.Tensor:
        hidden = action_hidden.float()
        tactile_tokens = tactile_tokens.float()
        for block in self.blocks:
            hidden = block(hidden, tactile_tokens, invalid)
        # The branch predicts the tactile velocity itself. Zero-initializing
        # its correction starts the late flow at the pretrained pi0.5 velocity.
        return base_velocity.float() + self.velocity_delta(self.output_norm(hidden))


class TactilePI05(nn.Module):
    def __init__(self, base: nn.Module, tactile: TactileVelocityExpert) -> None:
        super().__init__()
        self.base = base
        self.tactile = tactile
        self.base.eval().requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.eval()
        self.tactile.train(mode)
        return self

    def late_flow_loss(self, observation, actions: torch.Tensor, history: torch.Tensor,
                       current: torch.Tensor, maps: torch.Tensor) -> tuple[torch.Tensor, dict]:
        batch = actions.shape[0]
        # Exactly the four times used by the last four steps of a ten-step
        # Euler flow: 0.4, 0.3, 0.2 and 0.1.
        time = torch.randint(1, 5, (batch,), device=actions.device).float() / 10.0
        noise = torch.randn_like(actions)
        captured: dict[str, torch.Tensor] = {}

        def capture(_module, inputs, output):
            captured["hidden"] = inputs[0].detach()
            captured["velocity"] = output.detach()

        hook = self.base.action_out_proj.register_forward_hook(capture)
        try:
            # The converted pi0.5 has bfloat16 weights, while flow inputs and
            # targets remain float32. Autocast handles its linear/vision ops;
            # the trainable tactile expert below runs in float32.
            with torch.no_grad(), torch.autocast(device_type=actions.device.type, dtype=torch.bfloat16):
                self.base(observation, actions, noise=noise, time=time)
        finally:
            hook.remove()
        if "hidden" not in captured:
            raise RuntimeError("pi0.5 action projection hook did not fire")
        tokens, invalid = self.tactile.tactile_tokens(history, current, maps)
        tactile_velocity = self.tactile(captured["hidden"], captured["velocity"], tokens, invalid)
        target = noise - actions
        base_mse = (captured["velocity"].float() - target).square().mean().detach()
        mse = (tactile_velocity - target).square().mean()
        correction = (tactile_velocity - captured["velocity"].float()).square().mean()
        loss = mse + 1e-4 * correction
        return loss, {"mse": mse.detach(), "base_mse": base_mse, "correction_mse": correction.detach()}

    @torch.no_grad()
    def prepare_late_flow(self, observation, *, noise: torch.Tensor) -> TactileFlowCache:
        """Run the first six pi0.5 flow steps once per action chunk."""
        device = observation.state.device
        if noise.shape != (observation.state.shape[0], self.base.config.action_horizon,
                           self.base.config.action_dim):
            raise ValueError("noise shape does not match the pi0.5 action chunk")
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            images, img_masks, lang_tokens, lang_masks, state = self.base._preprocess_observation(  # noqa: SLF001
                observation, train=False)
            prefix_embs, prefix_pad_masks, prefix_att_masks = self.base.embed_prefix(
                images, img_masks, lang_tokens, lang_masks)
            prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
            prefix_att_2d_masks_4d = self.base._prepare_attention_masks_4d(prefix_att_2d_masks)  # noqa: SLF001
            self.base.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
            _, past_key_values = self.base.paligemma_with_expert.forward(
                attention_mask=prefix_att_2d_masks_4d,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=True,
            )
            x_t = noise
            dt = torch.tensor(-0.1, dtype=torch.float32, device=device)
            time = torch.tensor(1.0, dtype=torch.float32, device=device)
            for _ in range(6):
                velocity = self.base.denoise_step(
                    state, prefix_pad_masks, past_key_values, x_t, time.expand(x_t.shape[0]))
                x_t = x_t + dt * velocity
                time = time + dt
        return TactileFlowCache(state, prefix_pad_masks, past_key_values, x_t)

    @torch.no_grad()
    def continue_late_flow(self, cache: TactileFlowCache, *, history: torch.Tensor,
                           current: torch.Tensor, maps: torch.Tensor) -> torch.Tensor:
        """Re-run the final four steps from the same slow snapshot with fresh touch."""
        tokens, invalid = self.tactile.tactile_tokens(history, current, maps)
        injected_steps = 0

        def inject(_module, inputs, output):
            nonlocal injected_steps
            injected_steps += 1
            with torch.autocast(device_type=cache.x_split.device.type, enabled=False):
                return self.tactile(inputs[0], output, tokens, invalid)

        hook = self.base.action_out_proj.register_forward_hook(inject)
        try:
            with torch.autocast(device_type=cache.x_split.device.type, dtype=torch.bfloat16):
                x_t = cache.x_split.clone()
                dt = torch.tensor(-0.1, dtype=torch.float32, device=x_t.device)
                time = torch.tensor(0.4, dtype=torch.float32, device=x_t.device)
                for _ in range(4):
                    velocity = self.base.denoise_step(
                        cache.state, cache.prefix_pad_masks, cache.past_key_values,
                        x_t, time.expand(x_t.shape[0]))
                    x_t = x_t + dt * velocity
                    time = time + dt
        finally:
            hook.remove()
        if injected_steps != 4:
            raise RuntimeError(f"Expected four tactile flow steps, observed {injected_steps}")
        return x_t

    @torch.no_grad()
    def sample_actions(self, observation, *, num_steps: int = 10, noise: torch.Tensor | None = None,
                       history: torch.Tensor, current: torch.Tensor, maps: torch.Tensor) -> torch.Tensor:
        if num_steps != 10:
            raise ValueError("This late-flow expert is calibrated for 10 denoising steps")
        tokens, invalid = self.tactile.tactile_tokens(history, current, maps)
        step = 0

        def inject(_module, inputs, output):
            nonlocal step
            step += 1
            if step <= 6:
                return output
            with torch.autocast(device_type=observation.state.device.type, enabled=False):
                return self.tactile(inputs[0], output, tokens, invalid)

        hook = self.base.action_out_proj.register_forward_hook(inject)
        try:
            with torch.autocast(device_type=observation.state.device.type, dtype=torch.bfloat16):
                result = self.base.sample_actions(observation.state.device, observation,
                                                  noise=noise, num_steps=num_steps)
        finally:
            hook.remove()
        if step != num_steps:
            raise RuntimeError(f"Expected {num_steps} flow steps, observed {step}")
        return result
