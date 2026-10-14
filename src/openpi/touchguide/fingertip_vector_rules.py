"""Initial, explicitly synthetic rules for three-component finger perturbations."""

from collections import Counter
from itertools import product

import numpy as np

from openpi.touchguide.fingertip_kinematics import IKConfig
from openpi.touchguide.fingertip_recovery import TactileSynthesisConfig

RULES = {
    "version": "augmentation_rules_v1",
    "units": "signed millimetres in measured contact-local [away_normal, tangent1, tangent2]",
    "perturb_probability": 0.65,
    "component_zero_probability_before_constraints": 0.35,
    "nonzero_component_levels_mm": [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
    "release_goal_probability_if_both_goals_feasible": 0.25,
    "keep_contact_positive_normal_levels_mm": [0.25, 0.5, 0.75],
    "release_positive_normal_levels_mm": [1.25, 1.5, 2.0],
    "max_displacement_norm_mm": 2.0,
    "random_options_per_finger": 128,
    "pool_bootstrap": "26 signed 0.25mm component combinations and pure-axis amplitude levels",
    "contact_threshold_n": 0.001,
    "synthetic_release_distance_mm": 1.0,
    "synthetic_force_gain_cap": 3.0,
    "max_ik_error_um": 50.0,
    "max_joint_delta_rad": 0.15,
    "max_pad_rotation_deg": 15.0,
    "action_horizon": 30,
    "variants_per_origin": 10,
    "seed": 42,
    "requirements": [
        "Each supported finger draws its own flag, signed normal/tangent1/tangent2 components and contact goal.",
        "Unselected or unsupported fingers have zero displacement and exactly unchanged state/tactile.",
        "Selected fingers require at least one nonzero component; entirely unchanged groups are skipped.",
        "Values may coincide between fingers; no forced equality or inequality.",
        "Component draws are conditioned on displacement norm, contact goal and IK feasibility; not an unconstrained product distribution.",
        "Normal contact goal is sampled first; displacement must produce that goal under the local force surrogate.",
        "No new contact is fabricated on an originally unsupported/no-contact finger.",
        "Arm state is fixed; finger IK changes only selected finger joints.",
        "Thirty-step minimum-jerk recovery returns to the original current state, with no appended task continuation.",
        "Vision remains the original frame; synthetic touch is not physics ground truth.",
        "Retain train/val episode split and deterministic source identity; no training or full generation in a preview.",
    ],
}


def component_code(values):
    active = np.flatnonzero(np.asarray(values) != 0)
    if len(active) == 1:
        axis = int(active[0])
        return 2 * axis + int(values[axis] < 0)
    return {(0, 1): 6, (0, 2): 7, (1, 2): 8, (0, 1, 2): 9}[tuple(active)]


def _draw_components(rng, release):
    values = np.zeros(3)
    for axis in range(3):
        if axis == 0 and release:
            values[axis] = rng.choice(RULES["release_positive_normal_levels_mm"])
        elif rng.random() >= RULES["component_zero_probability_before_constraints"]:
            sign = int(rng.choice([-1, 1]))
            levels = (
                RULES["keep_contact_positive_normal_levels_mm"]
                if axis == 0 and sign > 0
                else RULES["nonzero_component_levels_mm"]
            )
            values[axis] = sign * rng.choice(levels)
    return values


def feasible_vector_options(k, state, full, basis, eligible, source_force, cache, rng):
    banks, audits, total = [[] for _ in range(5)], [], Counter()
    config = TactileSynthesisConfig()
    for finger in range(5):
        audit = Counter()
        if eligible[finger]:
            proposals = {}
            bootstrap = [np.asarray(v) * 0.25 for v in product((-1, 0, 1), repeat=3) if any(v)]
            for axis in range(3):
                for magnitude in RULES["nonzero_component_levels_mm"]:
                    for sign in (-1, 1):
                        values = np.zeros(3)
                        values[axis] = magnitude * sign
                        bootstrap.append(values)
            for values in bootstrap:
                gain = np.clip(1 - values[0] / RULES["synthetic_release_distance_mm"], 0, config.max_force_gain)
                goal = int(source_force[finger] * gain <= config.contact_threshold_n)
                proposals[tuple(values)] = goal
            random_values = set()
            for _ in range(2048):
                release = rng.random() < RULES["release_goal_probability_if_both_goals_feasible"]
                values = _draw_components(rng, release)
                norm = np.linalg.norm(values)
                if not 0 < norm <= RULES["max_displacement_norm_mm"]:
                    audit["rejected_zero_or_norm_draw"] += 1
                    continue
                key = tuple(values)
                if key in proposals:
                    continue
                proposals[key] = int(release)
                random_values.add(key)
                if len(random_values) >= RULES["random_options_per_finger"]:
                    break
            for component_tuple, goal in proposals.items():
                values = np.asarray(component_tuple)
                pattern = int(sum(1 << int(axis) for axis in np.flatnonzero(values)))
                audit["probed_options"] += 1
                audit[f"probed_component_pattern_{pattern}"] += 1
                delta = np.zeros((5, 3))
                delta[finger] = values @ basis[finger] * 0.001
                key = (finger, tuple(values))
                result = k.solve(state, full, delta, IKConfig())
                cache[key] = result
                if not result["success"]:
                    audit["ik_or_joint_bounds"] += 1
                    continue
                if np.rad2deg(result["rotation_change_rad"][finger]) > config.max_rotation_deg:
                    audit["pad_rotation"] += 1
                    continue
                actual = result["actual_displacement_world_m"][finger]
                if np.linalg.norm(actual) * 1000 > RULES["max_displacement_norm_mm"] + 1e-6:
                    audit["actual_norm_limit"] += 1
                    continue
                normal = float(actual @ basis[finger, 0])
                gain = np.clip(1 - normal / config.release_distance_m, 0, config.max_force_gain)
                if bool(source_force[finger] * gain > config.contact_threshold_n) == bool(goal):
                    audit["contact_goal_not_met"] += 1
                    continue
                banks[finger].append((tuple(values), goal, pattern))
                audit["feasible_options"] += 1
                audit[f"feasible_component_pattern_{pattern}"] += 1
                audit["feasible_release" if goal else "feasible_keep_contact"] += 1
        audits.append(dict(audit))
        total.update(audit)
    return banks, {"probe_counts": dict(total), "by_finger": audits}


def random_vector_combinations(rng, banks, audit, attempts=600):
    available = np.asarray([bool(bank) for bank in banks])
    if not available.any():
        return
    for _ in range(attempts):
        mask = (rng.random(5) < RULES["perturb_probability"]) & available
        audit["flag_draws"] += 1
        if not mask.any():
            audit["all_unchanged_draws"] += 1
            continue
        components = np.zeros((5, 3))
        goals = np.full(5, -1, np.int8)
        codes = np.full(5, -1, np.int8)
        for finger in np.flatnonzero(mask):
            bank = banks[finger]
            possible_goals = sorted({choice[1] for choice in bank})
            probability = np.asarray(
                [
                    RULES["release_goal_probability_if_both_goals_feasible"]
                    if goal
                    else 1 - RULES["release_goal_probability_if_both_goals_feasible"]
                    for goal in possible_goals
                ]
            )
            goal = int(rng.choice(possible_goals, p=probability / probability.sum()))
            choices = [choice for choice in bank if choice[1] == goal]
            patterns = sorted({choice[2] for choice in choices})
            zero = RULES["component_zero_probability_before_constraints"]
            weights = np.asarray([(1 - zero) ** p.bit_count() * zero ** (3 - p.bit_count()) for p in patterns])
            pattern = int(rng.choice(patterns, p=weights / weights.sum()))
            choices = [choice for choice in choices if choice[2] == pattern]
            choice = choices[int(rng.integers(len(choices)))]
            components[finger], goals[finger] = choice[:2]
            codes[finger] = component_code(components[finger])
            audit[f"drawn_finger_{finger}"] += 1
            audit[f"drawn_component_pattern_{pattern}"] += 1
            audit["drawn_release_goal" if goal else "drawn_keep_contact_goal"] += 1
        audit["combined_proposals"] += 1
        audit[f"selected_finger_count_{int(mask.sum())}"] += 1
        yield {
            "codes": codes,
            "magnitudes": np.linalg.norm(components, axis=1),
            "components_mm": components,
            "contact_goals": goals,
        }
