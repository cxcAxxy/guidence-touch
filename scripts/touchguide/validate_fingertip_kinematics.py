"""Independent saved-proposal FK checks and targeted IK/Jacobian validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from openpi.touchguide.fingertip_kinematics import FingertipKinematics
from openpi.touchguide.fingertip_kinematics import IKConfig


def validate(root):
    config = json.loads((root / "config.json").read_text())
    k = FingertipKinematics(config["model_path"])
    ik = IKConfig(
        config["position_tolerance_mm"] * 1e-3,
        config["max_joint_delta_rad"],
        config["posture_scale_m_per_rad"],
        config["max_evaluations"],
    )
    with h5py.File(root / "perturbations.h5", "r") as f:
        states, baseline_full = f["anchors/state"][:], f["anchors/full_qpos"][:]
        original_pos = f["anchors/position"][:]
        a = {name: d[:] for name, d in f["attempts"].items() if name != "status"}
        anchor_splits = f["anchors/split"].asstr()[:]
        episodes = f["anchors/episode_index"][:]
        mixed_coverage = None
        all_fk_error = []
        all_rotation_error = []
        for i, q in enumerate(a["full_qpos"]):
            anchor = int(a["anchor_index"][i])
            state, base = states[anchor], baseline_full[anchor]
            mask = a["perturbed_fingers"][i]
            changeable = np.zeros(27, dtype=bool)
            for finger in np.flatnonzero(mask):
                changeable[7 + 4 * finger : 11 + 4 * finger] = True
            assert np.array_equal(a["state"][i, ~changeable], state[~changeable]), "Arm/inactive state changed"
            assert np.array_equal(q[k.state_qpos], a["state"][i]), "Saved state/full qpos disagree"
            full_changeable = np.zeros(k.model.nq, dtype=bool)
            full_changeable[k.state_qpos[changeable]] = True
            if mask[0]:
                full_changeable[k.thumb6_qpos] = True
                assert abs((q[k.thumb6_qpos] - q[k.state_qpos[10]]) - (base[k.thumb6_qpos] - state[10])) < 1e-12
            assert np.array_equal(q[~full_changeable], base[~full_changeable]), "Inactive model joint changed"
            positions, rotations = k.fk(q)
            all_fk_error.append(float(np.max(np.abs(positions - a["actual_position_world_m"][i]))))
            all_rotation_error.append(float(np.max(np.abs(rotations - a["actual_rotation_world"][i]))))
            delta = positions - original_pos[anchor]
            np.testing.assert_allclose(delta, a["actual_displacement_world_m"][i], atol=1e-12, rtol=0)
            np.testing.assert_allclose(delta[~mask], 0, atol=1e-12, rtol=0)
            errors = np.linalg.norm(delta - a["requested_displacement_world_m"][i], axis=1)
            np.testing.assert_allclose(errors, a["error_m"][i], atol=1e-12, rtol=0)
            assert np.max(np.abs(a["state"][i] - state)) <= ik.max_joint_delta_rad + 1e-10
            if a["success"][i]:
                assert np.all(errors[mask] <= ik.position_tolerance_m)
                assert np.all(a["state"][i, changeable] >= k.lower[changeable] - 1e-10)
                assert np.all(a["state"][i, changeable] <= k.upper[changeable] + 1e-10)
                assert a["joint_bounds_ok"][i]
        assert max(all_fk_error) < 1e-12
        assert max(all_rotation_error) < 1e-12
        split_of_episode = {}
        for episode, split in zip(episodes, anchor_splits, strict=True):
            assert split_of_episode.get(int(episode), split) == split
            split_of_episode[int(episode)] = split
        with np.load(root / "valid_indices.npz") as indices:
            for split in ("train", "val"):
                rows = indices[split]
                assert np.all(a["success"][rows])
                assert np.all(anchor_splits[a["anchor_index"][rows]] == split)
                np.testing.assert_array_equal(
                    rows, np.flatnonzero(a["success"] & (anchor_splits[a["anchor_index"]] == split))
                )
            np.testing.assert_array_equal(indices["rejected"], np.flatnonzero(~a["success"]))
        if config.get("combination_mode") == "all_subsets_mixed":
            codes = a["direction_codes"]
            masks = a["perturbed_fingers"]
            np.testing.assert_array_equal(masks, codes >= 0)
            assert np.all((codes >= -1) & (codes <= 5))
            expected_coefficients = np.zeros_like(a["coefficients"])
            rows, fingers = np.nonzero(masks)
            expected_coefficients[rows, fingers, codes[rows, fingers] // 2] = (
                np.where(codes[rows, fingers] % 2 == 0, 1, -1) * a["magnitude_mm"][rows] * 1e-3
            )
            np.testing.assert_array_equal(expected_coefficients, a["coefficients"])
            bases = f["anchors/direction_basis_world"][:]
            np.testing.assert_allclose(
                bases @ bases.swapaxes(-1, -2), np.broadcast_to(np.eye(3), bases.shape), atol=1e-6
            )
            expected_delta = np.einsum("nfa,nfai->nfi", expected_coefficients, bases[a["anchor_index"]])
            np.testing.assert_allclose(expected_delta, a["requested_displacement_world_m"], atol=1e-12, rtol=0)
            subset_bits = np.sum(masks * (1 << np.arange(5)), axis=1)
            np.testing.assert_array_equal(subset_bits, a["subset_code"])
            basis_sources = f["anchors/direction_basis_source"].asstr()[:]
            eligible = f["anchors/eligible_fingers"][:]
            contact = f["anchors/contact_fingers"][:]
            np.testing.assert_array_equal(basis_sources == "measured_contact_chart", eligible)
            np.testing.assert_array_equal(basis_sources == "current_pad_axes_no_contact", ~contact)
            np.testing.assert_array_equal(basis_sources == "current_pad_axes_unreliable_contact", contact & ~eligible)
            for anchor in range(len(states)):
                in_anchor = a["anchor_index"] == anchor
                assert set(a["subset_code"][in_anchor]) == set(range(1, 32))
                for subset in range(1, 32):
                    in_group = np.flatnonzero(in_anchor & (a["subset_code"] == subset))
                    finger_count = int(subset).bit_count()
                    expected_patterns = 6 + (config["mixed_patterns_per_subset"] if finger_count > 1 else 0)
                    assert len(in_group) == expected_patterns * len(config["magnitudes_mm"])
                    local_patterns = set()
                    mixed_patterns = set()
                    for index in in_group:
                        directions = tuple(codes[index, masks[index]].tolist())
                        local_patterns.add((directions, float(a["magnitude_mm"][index])))
                        if a["pattern_kind"][index] == b"mixed":
                            assert len(set(directions)) > 1
                            mixed_patterns.add(directions)
                        else:
                            assert a["pattern_kind"][index] == b"uniform"
                            assert len(set(directions)) == 1
                    assert len(local_patterns) == len(in_group), "Duplicate subset/direction/amplitude proposal"
                    assert len(mixed_patterns) == (config["mixed_patterns_per_subset"] if finger_count > 1 else 0)
                    if finger_count > 1 and config["mixed_patterns_per_subset"] >= 6:
                        assert any(0 in d and 1 in d for d in mixed_patterns)
                        assert any(any(v < 2 for v in d) and any(v in (2, 3) for v in d) for d in mixed_patterns)
                        assert any(any(v < 2 for v in d) and any(v in (4, 5) for v in d) for d in mixed_patterns)
                        assert any(any(v in (2, 3) for v in d) and any(v in (4, 5) for v in d) for d in mixed_patterns)
                        if finger_count >= 3:
                            assert any({v // 2 for v in d} == {0, 1, 2} for d in mixed_patterns)
            mixed_coverage = {
                "all_31_subsets_per_anchor": True,
                "independent_finger_direction_codes": True,
                "mixed_direction_families_checked": True,
                "no_duplicate_proposals": True,
                "basis_source_flags_checked": True,
            }
    replay_rows = np.unique(np.linspace(0, len(a["state"]) - 1, 48, dtype=int))
    if config.get("combination_mode") == "all_subsets_mixed":
        candidates = np.flatnonzero((a["pattern_kind"] == b"mixed") & (a["perturbed_fingers"].sum(axis=1) >= 2))
        replay_rows = np.unique(np.r_[replay_rows, candidates[np.linspace(0, len(candidates) - 1, 32, dtype=int)]])
    for row in replay_rows:
        anchor = int(a["anchor_index"][row])
        replay = k.solve(states[anchor], baseline_full[anchor], a["requested_displacement_world_m"][row], ik)
        np.testing.assert_allclose(replay["state"], a["state"][row], atol=1e-12, rtol=0)
        assert replay["success"] == a["success"][row]
    # Finite differences independently check mj_jacSite reduction, including
    # joint5 moving the passive thumb joint6 (the two columns must be summed).
    jac_errors = []
    for anchor in (0, len(states) // 2, len(states) - 1):
        q = baseline_full[anchor]
        zero = k.solve(states[anchor], q, np.zeros((5, 3)), ik)
        assert zero["success"]
        assert np.array_equal(zero["state"], states[anchor])
        assert np.array_equal(zero["full_qpos"], q)
        for finger in range(5):
            analytic = k.finger_jacobian(q, finger)
            numerical = np.empty((3, 4))
            for joint in range(4):
                column = 7 + 4 * finger + joint
                plus, minus = q.copy(), q.copy()
                plus[k.state_qpos[column]] += 1e-6
                minus[k.state_qpos[column]] -= 1e-6
                if finger == 0 and joint == 3:
                    plus[k.thumb6_qpos] += 1e-6
                    minus[k.thumb6_qpos] -= 1e-6
                numerical[:, joint] = (k.fk(plus)[0][finger] - k.fk(minus)[0][finger]) / 2e-6
            jac_errors.append(float(np.max(np.abs(analytic - numerical))))
    assert max(jac_errors) < 1e-7
    impossible = np.zeros((5, 3))
    impossible[1, 2] = 1.0
    result = k.solve(states[0], baseline_full[0], impossible, ik)
    assert not result["success"], "Unreachable target incorrectly accepted"
    assert np.all(np.isfinite(result["state"]))
    # Reachable target derived from an actual local index-finger posture.
    usable = next(
        i
        for i, state in enumerate(states)
        if np.all(state[11:15] > k.lower[11:15]) and np.all(state[11:15] < k.upper[11:15])
    )
    state, q = states[usable], baseline_full[usable]
    moved = q.copy()
    column = 12
    step = 0.005 if state[column] + 0.005 < k.upper[column] else -0.005
    moved[k.state_qpos[column]] += step
    target_delta = np.zeros((5, 3))
    target_delta[1] = k.fk(moved)[0][1] - k.fk(q)[0][1]
    reachable = k.solve(state, q, target_delta, ik)
    assert reachable["success"], "Known reachable local target rejected"
    report = {
        "valid": True,
        "mixed_coverage": mixed_coverage,
        "solver_replay_attempts_checked": len(replay_rows),
        "attempts_independently_forward_checked": len(a["state"]),
        "max_saved_position_error_m": max(all_fk_error),
        "max_saved_rotation_element_error": max(all_rotation_error),
        "finite_difference_jacobians_checked": len(jac_errors),
        "max_jacobian_difference_m_per_rad": max(jac_errors),
        "zero_displacement_identity_checks": 3,
        "reachable_target_accepted": True,
        "unreachable_target_rejected": True,
        "inactive_joints_exactly_preserved": True,
        "selected_joint_bounds_and_trust_region_checked": True,
        "split_and_valid_indices_checked": True,
    }
    (root / "validation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-root",
        type=Path,
        default=Path(
            "/path/to/datasets/touchguidence/augmentation/poker_fingertip_recovery_v1/kinematics/pilot_v1"
        ),
    )
    validate(p.parse_args().data_root)
