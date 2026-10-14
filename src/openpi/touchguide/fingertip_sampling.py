"""Enumerate finger subsets and sample mixed directions for static IK pilots."""

from __future__ import annotations

from functools import lru_cache
from itertools import product

import numpy as np

from openpi.touchguide.fingertip_kinematics import FINGERS


@lru_cache(maxsize=5)
def _mixed_choices(count):
    choices = np.array(list(product(range(6), repeat=count)), dtype=np.int8)
    return choices[np.any(choices != choices[:, :1], axis=1)]


def subset_patterns(rng, mixed_per_subset=6):
    """All 31 nonempty masks; six uniform directions plus sampled mixtures.

    Codes: 0 away, 1 approach, 2/3 tangent0 +/-, 4/5 tangent1 +/-;
    -1 marks an untouched finger. Mixtures are unique within each subset.
    """
    if not 0 <= mixed_per_subset <= 30:
        raise ValueError("mixed_per_subset must be between 0 and 30")
    for subset in range(1, 32):
        fingers = np.flatnonzero([(subset >> f) & 1 for f in range(5)])
        mask = np.isin(np.arange(5), fingers)
        for direction in range(6):
            codes = np.full(5, -1, dtype=np.int8)
            codes[fingers] = direction
            yield subset, mask, "uniform", codes
        if len(fingers) == 1:
            continue
        choices = _mixed_choices(len(fingers))
        away = np.any(choices == 0, axis=1)
        approach = np.any(choices == 1, axis=1)
        normal = away | approach
        tangent0 = np.any((choices == 2) | (choices == 3), axis=1)
        tangent1 = np.any((choices == 4) | (choices == 5), axis=1)
        # Guarantee opposite normal directions, normal/tangent mixtures and
        # both tangent axes; subsets of >=3 also include all three axis families.
        strata = [
            away & approach,
            away & approach,
            normal & tangent0,
            normal & tangent1,
            tangent0 & tangent1,
            normal & tangent0 & tangent1 if len(fingers) >= 3 else normal & tangent1,
        ]
        available = np.ones(len(choices), dtype=bool)
        for index in range(mixed_per_subset):
            eligible = available & (strata[index] if index < len(strata) else True)
            candidates = np.flatnonzero(eligible)
            if not len(candidates):
                raise ValueError("No distinct mixed pattern remains in the requested stratum")
            chosen = int(rng.choice(candidates))
            available[chosen] = False
            codes = np.full(5, -1, dtype=np.int8)
            codes[fingers] = choices[chosen]
            yield subset, mask, "mixed", codes


def direction_basis(anchor, raw, kinematics, full_qpos):
    """Measured contact chart when reliable; current pad axes otherwise.

    Fallback outward/inward axes describe the finger pad, not a measured
    object/contact normal. Moving inward alone cannot establish new contact.
    """
    kinematics.fk(full_qpos)
    contact = np.asarray(anchor["contact_fingers"], dtype=bool)
    eligible = np.asarray(anchor["eligible_fingers"], dtype=bool)
    if np.any(eligible & ~contact):
        raise ValueError("Eligible contact finger without contact")
    group = raw["tactile_contact_force"]
    names = group["link_names"].asstr()[:].tolist()
    normals = group["normal_axis_local"][:]
    tangents = group["tangent_basis_local"][:]
    basis = np.zeros((5, 3, 3))
    sources = []
    for finger, name in enumerate(FINGERS):
        if eligible[finger]:
            basis[finger] = [
                anchor["contact_normal_away_world"][finger],
                anchor["contact_tangent0_world"][finger],
                anchor["contact_tangent1_world"][finger],
            ]
            sources.append("measured_contact_chart")
        else:
            link = f"hand_r_{name}_link{6 if name == 'thumb' else 4}"
            index = names.index(link)
            rotation = kinematics.data.xmat[kinematics.model.body(link).id].reshape(3, 3)
            local = np.vstack([-normals[index], tangents[index]])
            basis[finger] = local @ rotation.T
            sources.append("current_pad_axes_unreliable_contact" if contact[finger] else "current_pad_axes_no_contact")
        if not np.isfinite(basis[finger]).all() or not np.allclose(
            basis[finger] @ basis[finger].T, np.eye(3), atol=1e-6
        ):
            raise ValueError(f"Invalid direction basis for {name}")
    return basis, sources


def compose_finger_solutions(kinematics, state, full_qpos, delta, solutions, config):
    """Combine separately solved fingers with a fixed arm, then recompute FK.

    Each finger has a disjoint actuated chain. Reusing its solution avoids
    resolving the same local IK for every subset. Independent validation also
    compares composed results with joint calls to the reference solver.
    """
    original_pos, original_rot = kinematics.fk(full_qpos)
    result = np.array(state, copy=True)
    full = np.array(full_qpos, copy=True)
    mask = np.any(delta != 0, axis=1)
    if set(solutions) != set(np.flatnonzero(mask)):
        raise ValueError("Cached solutions do not cover the selected fingers exactly")
    evaluations = np.zeros(5, dtype=np.int64)
    statuses = ["not_perturbed"] * 5
    bounded = True
    for finger, solution in solutions.items():
        selected = np.arange(5) == finger
        if not np.array_equal(solution["perturbed_fingers"], selected):
            raise ValueError("Cache must contain a single-finger solution")
        if not np.array_equal(solution["requested_displacement_world_m"][finger], delta[finger]):
            raise ValueError("Cached direction/amplitude differs from target")
        s = slice(7 + 4 * finger, 11 + 4 * finger)
        result[s] = solution["state"][s]
        full[kinematics.state_qpos[s]] = result[s]
        if finger == 0:
            full[kinematics.thumb6_qpos] = solution["full_qpos"][kinematics.thumb6_qpos]
        bounded &= bool(solution["joint_bounds_ok"])
        evaluations[finger] = solution["evaluations"][finger]
        statuses[finger] = solution["status"][finger]
    positions, rotations = kinematics.fk(full)
    targets = original_pos + delta
    errors = np.linalg.norm(positions - targets, axis=1)
    success = bool(
        np.all(errors[mask] <= config.position_tolerance_m) and bounded and "empty_joint_bounds" not in statuses
    )
    angles = np.arccos(np.clip((np.einsum("fij,fij->f", original_rot, rotations) - 1) / 2, -1, 1))
    return {
        "state": result,
        "full_qpos": full,
        "perturbed_fingers": mask,
        "target_position_world_m": targets,
        "actual_position_world_m": positions,
        "actual_rotation_world": rotations,
        "requested_displacement_world_m": delta.copy(),
        "actual_displacement_world_m": positions - original_pos,
        "error_m": errors,
        "rotation_change_rad": angles,
        "evaluations": evaluations,
        "status": statuses,
        "joint_bounds_ok": bounded,
        "success": success,
    }
