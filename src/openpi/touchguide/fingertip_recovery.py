"""Pure numerical recovery targets and explicitly synthetic contact-map proposals.

Normal force uses a configurable local compliance surrogate, not a measured
stiffness. Spatial changes reuse the recorded Gaussian taxel kernel and mesh.
No-contact pads stay at zero: new object contact is not inferred from pad axes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TactileSynthesisConfig:
    release_distance_m: float = 0.001
    max_force_gain: float = 3.0
    max_rotation_deg: float = 15.0
    contact_threshold_n: float = 0.001

    def __post_init__(self):
        values = (self.release_distance_m, self.max_force_gain, self.max_rotation_deg, self.contact_threshold_n)
        if not np.isfinite(values).all() or min(values) <= 0 or self.max_force_gain < 1:
            raise ValueError("Tactile synthesis parameters must be positive; max_force_gain must be >=1")


def recovery_targets(start, goal, horizon=30):
    """Thirty future targets; implicit t=0 is start, t=30 is goal.

    Minimum-jerk quintic interpolation has zero endpoint velocity/acceleration.
    A fixed arm and inactive fingers are preserved exactly, not re-rounded by
    mixing two large absolute values. Joint paths stay between the endpoints.
    """
    start, goal = np.asarray(start, dtype=np.float64), np.asarray(goal, dtype=np.float64)
    if start.shape != (27,) or goal.shape != (27,) or not np.isfinite(np.r_[start, goal]).all():
        raise ValueError("Recovery endpoints must be finite 27D joint states")
    if horizon < 2 or not np.array_equal(start[:7], goal[:7]):
        raise ValueError("Recovery requires >=2 steps and an unchanged arm")
    t = np.arange(1, horizon + 1, dtype=np.float64) / horizon
    alpha = t**3 * (10 - 15 * t + 6 * t**2)
    alpha[-1] = 1.0
    result = start[None] + alpha[:, None] * (goal - start)[None]
    result[-1] = goal
    return result.astype(np.float32), alpha


def normalize_coordinates(values, statistics):
    """Original OpenPI quantile formula and epsilon; do not clip values."""
    values = np.asarray(values, dtype=np.float32)
    low, high = (np.asarray(statistics[key], dtype=np.float32) for key in ("q01", "q99"))
    if low.shape != (27,) or high.shape != (27,) or not np.isfinite(np.r_[low, high]).all():
        raise ValueError("Expected original finite 27D quantile statistics")
    if np.any(high < low):
        raise ValueError("Invalid quantile ordering")
    return ((values - low) / (high - low + 1e-6) * 2 - 1).astype(np.float32)


def synthesize_tactile(
    original,
    mask,
    eligible,
    displacement,
    rotation_deg,
    contact_basis,
    centroid,
    original_taxels_world,
    perturbed_taxels_world,
    original_tangents_world,
    perturbed_tangents_world,
    sigma_m,
    config,
):
    """Deform existing contact support; retain exact unselected/zero proposals.

    Fn gain = clip(1 - actual_tip_displacement.dot(away_normal)/release_distance,
    0, max_force_gain). Gaussian distance *ratios* deform the recorded profile,
    so an identity transform exactly recovers even multi-contact source maps.
    The original signed shear vector is projected into the perturbed pad axes;
    this preserves its world direction approximately, without inventing a
    velocity-dependent friction law from a static position perturbation.
    """
    original = np.asarray(original, dtype=np.float32)
    if original.shape != (5, 3, 7, 5) or not np.isfinite(original).all() or np.any(original[:, 0] < 0):
        raise ValueError("Expected finite [5,3,7,5] force maps with nonnegative normals")
    if not np.isfinite(sigma_m) or sigma_m <= 0:
        raise ValueError("Gaussian sigma must be positive")
    mask, eligible = np.asarray(mask, dtype=bool), np.asarray(eligible, dtype=bool)
    result = original.copy()
    gain = np.ones(5)
    normal_displacement = np.zeros(5)
    spatial_profile_l1 = np.zeros(5)
    sources = np.full(5, "unchanged", dtype=object)
    supported = ~mask | eligible
    for finger in np.flatnonzero(mask):
        if not eligible[finger]:
            sources[finger] = "unmodeled_contact_change_keep_source"
            continue
        sources[finger] = "synthetic_existing_contact_local_surrogate"
        delta = displacement[finger]
        # Exact identity protects original maps, including signed shear.
        if np.array_equal(original_taxels_world[finger], perturbed_taxels_world[finger]) and np.all(delta == 0):
            continue
        n = contact_basis[finger, 0]
        normal_displacement[finger] = np.dot(delta, n)
        gain[finger] = np.clip(1 - normal_displacement[finger] / config.release_distance_m, 0, config.max_force_gain)
        base = original[finger].astype(np.float64).reshape(3, 35)
        total = base[0].sum()
        if total <= 0 or not np.isfinite(centroid[finger]).all():
            raise ValueError("Modeled contact requires positive source force and a measured centroid")
        before = original_taxels_world[finger] - centroid[finger]
        after = perturbed_taxels_world[finger] - centroid[finger]
        log_ratio = (np.sum(before**2, axis=1) - np.sum(after**2, axis=1)) / (2 * sigma_m**2)
        weights = np.exp(log_ratio - log_ratio.max())
        denominator = np.dot(base[0], weights)
        if denominator <= 0 or not np.isfinite(denominator):
            raise ValueError("Invalid deformed taxel distribution")
        relative_weights = weights * total / denominator
        profile = base[0] * relative_weights / total
        spatial_profile_l1[finger] = np.abs(profile - base[0] / total).sum()
        tangent_projection = perturbed_tangents_world[finger] @ original_tangents_world[finger].T
        output = np.vstack([base[0], tangent_projection @ base[1:]])
        output *= relative_weights[None] * gain[finger]
        result[finger] = output.reshape(3, 7, 5).astype(np.float32)
    recommended = bool(np.all(supported) and np.all(np.asarray(rotation_deg)[mask] <= config.max_rotation_deg))
    return {
        "tactile_map_raw": result,
        "normal_displacement_m": normal_displacement,
        "normal_force_gain": gain,
        "spatial_profile_l1": spatial_profile_l1,
        "finger_synthesis_source": sources,
        "tactile_supported_fingers": supported,
        "recommended_training": recommended,
    }
