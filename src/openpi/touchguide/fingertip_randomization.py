"""Finger-wise random proposals conditioned on IK and synthetic contact support.

Contact changes are goals coupled to normal displacement, never independent
labels. This module does not create new contact on originally unsupported pads.
"""

from collections import Counter

import numpy as np

from openpi.touchguide.fingertip_kinematics import IKConfig
from openpi.touchguide.fingertip_recovery import TactileSynthesisConfig


def feasible_finger_options(kinematics, state, full_qpos, basis, eligible, source_force, cache):
    """Probe each finger before combining, so invalid fingers cannot bias masks."""
    banks = [[] for _ in range(5)]
    counts = Counter()
    by_finger = []
    config = TactileSynthesisConfig()
    for finger in range(5):
        audit = Counter()
        if eligible[finger]:
            for code in range(6):
                # Below release distance: retain contact. Above it: release.
                amplitudes = (0.25, 0.5, 0.75, 1.1, 1.5, 2.0) if code == 0 else (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
                for magnitude in amplitudes:
                    audit[f"probed_direction_{code}"] += 1
                    delta = np.zeros((5, 3))
                    delta[finger] = basis[finger, code // 2] * (1 if code % 2 == 0 else -1) * magnitude * 0.001
                    key = (finger, code, magnitude)
                    result = kinematics.solve(state, full_qpos, delta, IKConfig())
                    cache[key] = result
                    if not result["success"]:
                        audit["ik_or_joint_bounds"] += 1
                        continue
                    if np.rad2deg(result["rotation_change_rad"][finger]) > config.max_rotation_deg:
                        audit["pad_rotation"] += 1
                        continue
                    actual_normal = float(result["actual_displacement_world_m"][finger] @ basis[finger, 0])
                    gain = np.clip(1 - actual_normal / config.release_distance_m, 0, config.max_force_gain)
                    contact_after = source_force[finger] * gain > config.contact_threshold_n
                    release_goal = code == 0 and magnitude > config.release_distance_m * 1000
                    if contact_after == release_goal:
                        audit["contact_goal_not_met"] += 1
                        continue
                    banks[finger].append((code, magnitude, int(release_goal)))
                    audit[f"feasible_direction_{code}"] += 1
                    audit["feasible_release" if release_goal else "feasible_keep_contact"] += 1
        by_finger.append(dict(audit))
        counts.update(audit)
    return banks, {"probe_counts": dict(counts), "by_finger": by_finger}


def random_finger_combinations(rng, banks, audit, perturb_probability=0.65, attempts=600):
    """Bernoulli mask per finger, then random feasible direction/contact/amplitude.

    Equal choices between different fingers are allowed. No single-finger
    fallback is applied. Direction and amplitude distributions are conditional
    on physical reachability and on the synthetic contact goal.
    """
    if not 0 < perturb_probability < 1:
        raise ValueError("perturb_probability must be between zero and one")
    available = np.asarray([bool(bank) for bank in banks])
    if not available.any():
        return
    for _ in range(attempts):
        selected = (rng.random(5) < perturb_probability) & available
        audit["flag_draws"] += 1
        if not selected.any():
            audit["all_unchanged_draws"] += 1
            continue
        codes = np.full(5, -1, np.int8)
        magnitudes = np.zeros(5, np.float64)
        for finger in np.flatnonzero(selected):
            bank = banks[finger]
            directions = sorted({choice[0] for choice in bank})
            direction = int(rng.choice(directions))
            choices = [choice for choice in bank if choice[0] == direction]
            goals = sorted({choice[2] for choice in choices})
            goal = int(rng.choice(goals))
            choices = [choice for choice in choices if choice[2] == goal]
            choice = choices[int(rng.integers(len(choices)))]
            codes[finger], magnitudes[finger] = choice[:2]
            audit[f"drawn_finger_{finger}"] += 1
            audit[f"drawn_direction_{direction}"] += 1
            audit["drawn_release_goal" if goal else "drawn_keep_contact_goal"] += 1
        audit["combined_proposals"] += 1
        audit[f"selected_finger_count_{int(selected.sum())}"] += 1
        if selected.sum() > 1:
            chosen = codes[selected]
            audit["multi_finger_proposals"] += 1
            audit["normal_tangent_mixed_proposals"] += int(np.any(chosen < 2) and np.any(chosen >= 2))
        yield codes, magnitudes
