"""Distance weights for ambiguous action negatives; never success labels."""

from __future__ import annotations

import math

import torch

GROUPS = (
    slice(0, 7),
    slice(7, 11),
    slice(11, 15),
    slice(15, 19),
    slice(19, 23),
    slice(23, 27),
)


@torch.no_grad()
def action_distances(actions: torch.Tensor) -> torch.Tensor:
    """Max group RMS over both the first five and all 30 clean CPM steps.

    Arm delta is its own group. Each finger is a separate group so differences
    in one finger are not diluted by four stationary fingers. The two windows
    retain differences at the start of trajectories sharing the same endpoint.
    """
    if actions.ndim != 3 or tuple(actions.shape[1:]) != (30, 27):
        raise ValueError("Expected clean CPM actions [B,30,27]")
    if not actions.is_floating_point() or not torch.isfinite(actions).all():
        raise ValueError("Action distances require finite floating tensors")
    actions = actions.detach().float()
    size = len(actions)
    result = torch.zeros(size, size, device=actions.device)
    for window in (5, 30):
        for group in GROUPS:
            x = actions[:, :window, group].reshape(size, -1)
            # Direct cdist avoids cancellation in x*x+y*y-2*x*y for near twins.
            distance = torch.cdist(
                x, x, compute_mode="donot_use_mm_for_euclid_dist"
            ) / math.sqrt(x.shape[1])
            result = torch.maximum(result, distance)
    return result


@torch.no_grad()
def negative_weights(actions: torch.Tensor, policy: dict) -> torch.Tensor:
    if policy.get("schema") != "touchguide_negative_weights_v1":
        raise ValueError("Unsupported negative-pair policy")
    ignore, full = (
        float(policy[k]) for k in ("ignore_distance", "full_weight_distance")
    )
    if not math.isfinite(ignore) or not math.isfinite(full) or not 0 <= ignore < full:
        raise ValueError("Use finite 0 <= ignore_distance < full_weight_distance")
    distances = action_distances(actions)
    weights = ((distances - ignore) / (full - ignore)).clamp(0, 1)
    weights.fill_diagonal_(1)
    return weights


def attach_pairing(output: dict, model, batch: dict) -> dict:
    """Use clean actions for weighting, independent of flow noise and scores."""
    if not getattr(model, "uses_recovery_normalization", False):
        return output
    clean = model.action_encoder.coordinates.pi05_actions(batch["actions"]).detach()
    return {
        **output,
        "negative_weights": negative_weights(clean, model.negative_policy),
    }
