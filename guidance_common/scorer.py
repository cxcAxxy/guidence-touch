"""Reuse an existing frozen CPM without severing the caller's action graph."""

from typing import Any

import torch


class CPMScorer:
    """Raw cosine scorer; samplers own gradient computation and noise-time selection."""

    def __init__(self, model: Any):
        if model.training or any(p.requires_grad for p in model.parameters()):
            raise ValueError("CPMScorer requires an eval-mode CPM with frozen parameters")
        self.model = model

    @torch.no_grad()
    def encode_observation(self, head, wrist, tactile_map_raw, **state_kwargs):
        return self.model.encode_observation(head, wrist, tactile_map_raw, **state_kwargs).detach()

    def score(self, embedding: torch.Tensor, actions: torch.Tensor, noise_time=0.0) -> torch.Tensor:
        if actions.ndim != 3 or tuple(actions.shape[1:]) != (30, 27):
            raise ValueError("CPM scores normalized physical actions [B,30,27]")
        # float() preserves action gradients, including through an upstream VLA.
        # Do not detach actions here, apply softmax, or divide by train temperature.
        with torch.autocast(device_type=actions.device.type, enabled=False):
            scores = self.model.paired_scores(embedding.detach(), actions.float(), noise_time)
        if scores.shape != (len(actions),):
            raise ValueError("CPM must return one independent score per candidate")
        if not torch.isfinite(scores).all():
            raise FloatingPointError("Nonfinite CPM scores")
        return scores
