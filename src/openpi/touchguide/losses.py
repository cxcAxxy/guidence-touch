"""Symmetric contrastive matching losses for the TouchGuide CPM.

Labels express observation/action identity matching, not physical task success.
The caller supplies temperature-scaled logits; this module never applies a
softmax before cross entropy and does not construct or import a model.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as functional


def _finite_float_tensor(value: Any, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise ValueError(f"{name} must be a floating tensor")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")
    return value


def _identity_targets(
    targets: torch.Tensor | None, size: int, device: torch.device
) -> torch.Tensor:
    if targets is None:
        return torch.arange(size, dtype=torch.long, device=device)
    if (
        not isinstance(targets, torch.Tensor)
        or targets.dtype != torch.long
        or targets.shape != (size,)
    ):
        raise ValueError("targets must be a long tensor with shape [B]")
    expected = torch.arange(size, dtype=torch.long, device=targets.device)
    if not torch.equal(targets, expected):
        raise ValueError("Only diagonal identity targets arange(B) are supported")
    return targets.to(device=device)


def symmetric_infonce(
    logits: torch.Tensor, targets: torch.Tensor | None = None
) -> dict[str, torch.Tensor]:
    """Average row-wise and column-wise cross entropy on raw square logits."""
    logits = _finite_float_tensor(logits, "logits")
    if logits.ndim != 2 or logits.shape[0] < 2 or logits.shape[0] != logits.shape[1]:
        raise ValueError("logits must have square shape [B>=2,B]")
    targets = _identity_targets(targets, logits.shape[0], logits.device)
    loss_obs_to_action = functional.cross_entropy(logits, targets)
    loss_action_to_obs = functional.cross_entropy(logits.transpose(0, 1), targets)
    return {
        "loss": (loss_obs_to_action + loss_action_to_obs) / 2,
        "loss_obs_to_action": loss_obs_to_action,
        "loss_action_to_obs": loss_action_to_obs,
        "accuracy_obs_to_action": (logits.argmax(dim=1) == targets).float().mean(),
        "accuracy_action_to_obs": (logits.argmax(dim=0) == targets).float().mean(),
    }


def cpm_contrastive_loss(
    output: dict[str, Any], batch: dict[str, Any]
) -> dict[str, torch.Tensor]:
    """Validate matched identities and add detached cosine-score diagnostics.

    ``output['logits']`` must already include the model's temperature scaling.
    Clean-loader targets/masks may be on CPU when model output is on a GPU.
    """
    output_keys = {
        "logits",
        "cosine_scores",
        "observation_embedding",
        "action_embedding",
        "temperature",
    }
    batch_keys = {"targets", "positive_mask", "negative_mask"}
    if not isinstance(output, dict) or not output_keys.issubset(output):
        raise ValueError(f"output must contain {sorted(output_keys)}")
    if not isinstance(batch, dict) or not batch_keys.issubset(batch):
        raise ValueError(f"batch must contain {sorted(batch_keys)}")
    if batch["targets"] is None:
        raise ValueError("batch targets must explicitly identify diagonal matching")
    metrics = symmetric_infonce(output["logits"], batch["targets"])
    weights = batch.get("negative_weights", output.get("negative_weights"))
    if weights is not None:
        metrics.update(weighted_symmetric_infonce(output["logits"], weights))
    size = output["logits"].shape[0]
    positive = batch["positive_mask"]
    negative = batch["negative_mask"]
    for name, mask in (("positive_mask", positive), ("negative_mask", negative)):
        if (
            not isinstance(mask, torch.Tensor)
            or mask.dtype != torch.bool
            or mask.shape != (size, size)
        ):
            raise ValueError(f"{name} must be a bool tensor with shape [B,B]")
        identity = torch.eye(size, dtype=torch.bool, device=mask.device)
        expected = identity if name == "positive_mask" else ~identity
        if not torch.equal(mask, expected):
            raise ValueError(f"{name} must describe diagonal identity matching")
    scores = _finite_float_tensor(output["cosine_scores"], "cosine_scores")
    if scores.shape != (size, size):
        raise ValueError("cosine_scores must have shape [B,B]")
    observation = _finite_float_tensor(
        output["observation_embedding"], "observation_embedding"
    )
    action = _finite_float_tensor(output["action_embedding"], "action_embedding")
    if (
        observation.ndim != 2
        or observation.shape[0] != size
        or observation.shape[1] < 1
    ):
        raise ValueError("observation_embedding must have shape [B,D>=1]")
    if action.shape != observation.shape:
        raise ValueError(
            "action_embedding must have the same [B,D] shape as observation_embedding"
        )
    temperature = _finite_float_tensor(output["temperature"], "temperature")
    if temperature.numel() != 1 or not bool((temperature > 0).all()):
        raise ValueError("temperature must contain one positive scalar")
    positive_score = scores[positive.to(device=scores.device)].mean().detach()
    negative_score = scores[negative.to(device=scores.device)].mean().detach()
    metrics.update(
        positive_score=positive_score,
        negative_score=negative_score,
        score_gap=positive_score - negative_score,
        temperature=temperature.reshape(()).detach(),
    )
    return metrics


def weighted_symmetric_infonce(logits, weights):
    """Weighted partition function; diagonal remains the only positive target."""
    logits = _finite_float_tensor(logits, "logits")
    weights = (
        _finite_float_tensor(weights, "negative weights").detach().to(logits.device)
    )
    if (
        logits.ndim != 2
        or logits.shape[0] < 2
        or logits.shape[0] != logits.shape[1]
        or weights.shape != logits.shape
    ):
        raise ValueError("Expected square logits and weights")
    if (
        torch.any(weights < 0)
        or torch.any(weights > 1)
        or not torch.all(weights.diagonal() == 1)
    ):
        raise ValueError("Weights must lie in [0,1], with diagonal 1")
    # clamp only inside log to keep the derivative finite; zero weights are
    # explicitly masked out, and never become extra positive labels.
    adjusted = logits + weights.clamp_min(1e-30).log()
    adjusted = adjusted.masked_fill(weights == 0, -torch.inf)
    row = (torch.logsumexp(adjusted, 1) - logits.diagonal()).mean()
    column = (torch.logsumexp(adjusted, 0) - logits.diagonal()).mean()
    offdiag = ~torch.eye(len(logits), dtype=torch.bool, device=logits.device)
    w = weights[offdiag]
    return {
        "loss": (row + column) / 2,
        "loss_obs_to_action": row,
        "loss_action_to_obs": column,
        "unweighted_loss": symmetric_infonce(logits)["loss"].detach(),
        "negative_weight_mean": w.mean(),
        "ignored_negative_fraction": (w == 0).float().mean(),
        "downweighted_negative_fraction": ((w > 0) & (w < 1)).float().mean(),
        "effective_negatives_per_observation": (weights.sum(1) - 1).mean(),
    }
