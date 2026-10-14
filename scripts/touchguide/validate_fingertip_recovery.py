"""Audit saved recovery pairs against original recordings and geometric contracts."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from openpi.touchguide.fingertip_kinematics import FingertipKinematics  # noqa: E402
from openpi.touchguide.fingertip_recovery import TactileSynthesisConfig  # noqa: E402
from openpi.touchguide.fingertip_recovery import synthesize_tactile  # noqa: E402

DEFAULT = Path(
    "/path/to/datasets/touchguidence/augmentation/poker_fingertip_recovery_v1/recovery/pilot_v1_30steps"
)


def run(root):
    config = json.loads((root / "config.json").read_text())
    kin_config = json.loads((Path(config["kinematic_root"]) / "config.json").read_text())
    stats = json.loads((root / "action_norm_stats.json").read_text())["norm_stats"]
    k = FingertipKinematics(kin_config["model_path"])
    with ExitStack() as stack:
        f = stack.enter_context(h5py.File(root / "recovery.h5", "r"))
        kin = stack.enter_context(h5py.File(Path(config["kinematic_root"]) / "perturbations.h5", "r"))
        a, s = f["anchors"], f["samples"]
        idx = s["anchor_index"][:]
        count = len(idx)
        state, goal = s["state_raw"][:], a["state"][:][idx]
        absolute, actions = s["actions_absolute"][:], s["actions"][:]
        touch = s["tactile_map_raw"][:]
        base_touch = a["tactile_map_raw"][:][idx]
        mask = s["perturbed_fingers"][:]
        selected_joints = np.concatenate([np.zeros((count, 7), dtype=bool), np.repeat(mask, 4, axis=1)], axis=1)
        for value in (state, absolute, actions, touch, s["state_normalized"][:]):
            assert np.isfinite(value).all(), "Nonfinite model input/target"
        assert (absolute.shape, actions.shape, touch.shape) == ((count, 30, 27), (count, 30, 27), (count, 5, 3, 7, 5))
        assert np.array_equal(absolute[:, -1], goal.astype(np.float32)), "Recovery endpoint mismatch"
        assert np.array_equal(state[:, :7], goal[:, :7]), "Arm state changed"
        unchanged = np.broadcast_to(~selected_joints[:, None], absolute.shape)
        fixed = np.broadcast_to(goal.astype(np.float32)[:, None], absolute.shape)
        assert np.array_equal(absolute[unchanged], fixed[unchanged]), "Inactive recovery target changed"
        path = np.concatenate([state.astype(np.float32)[:, None], absolute], axis=1)
        steps = np.diff(path, axis=1)
        direction = np.sign(goal - state)
        assert np.min(steps * direction[:, None]) >= -1e-7, "Joint recovery reversed direction"
        assert np.all(absolute >= np.minimum(state, goal)[:, None] - 2e-7)
        assert np.all(absolute <= np.maximum(state, goal)[:, None] + 2e-7)
        # Midpoint and time indexing are independent known facts of the
        # symmetric minimum-jerk curve: t=15/30 is halfway, t=30/30 is goal.
        midpoint_error = float(np.max(np.abs(absolute[:, 14] - ((state + goal) / 2).astype(np.float32))))
        assert midpoint_error < 2e-7
        assert np.allclose(f["interpolation_alpha"][14], 0.5, atol=1e-14)
        assert f["interpolation_alpha"][-1] == 1
        physical = absolute.astype(np.float64)
        physical[:, :, :7] -= state[:, None, :7]
        low, high = (np.asarray(stats["actions"][key]) for key in ("q01", "q99"))
        restored = (actions.astype(np.float64) + 1) / 2 * (high - low + 1e-6) + low
        norm_error = float(np.max(np.abs(restored - physical)))
        assert norm_error < 1e-6, "Action normalization round trip failed"
        assert np.array_equal(touch[~mask], base_touch[~mask]), "Inactive tactile map changed"
        eligible = a["eligible_fingers"][:][idx]
        assert np.array_equal(touch[mask & ~eligible], base_touch[mask & ~eligible]), "Unmodeled contact was invented"
        assert np.all(touch[:, :, 0] >= 0), "Negative normal force"
        old_fn = base_touch[:, :, 0].sum(axis=(-2, -1), dtype=np.float64)
        new_fn = touch[:, :, 0].sum(axis=(-2, -1), dtype=np.float64)
        gains = s["normal_force_gain"][:]
        conservation_error = float(np.max(np.abs(new_fn - old_fn * gains)))
        assert conservation_error < 1e-6, "Normal force accounting failed"
        codes = s["direction_codes"][:]
        away, toward = (mask & eligible & (codes == c) for c in (0, 1))
        assert np.all(new_fn[away] <= old_fn[away] + 1e-6), "Away motion increased normal force"
        assert np.all(new_fn[toward] >= old_fn[toward] - 1e-6), "Approach motion reduced normal force"
        assert np.array_equal(s["contact_fingers"][:], new_fn > config["contact_threshold_n"])
        attempts = s["source_attempt_index"][:]
        assert np.array_equal(attempts, np.flatnonzero(kin["attempts/success"][:]))
        assert np.array_equal(state, kin["attempts/state"][:][attempts])
        source_files = {}
        for i in range(len(a["state"])):
            source_path = a["cpm_h5_path"].asstr()[i]
            if source_path not in source_files:
                source_files[source_path] = stack.enter_context(h5py.File(source_path, "r"))
            cpm = source_files[source_path]
            row = int(a["sample_row"][i])
            assert np.array_equal(a["state"][i].astype(np.float32), cpm["state_raw"][row])
            for key in ("tactile_map_raw", "actions", "actions_absolute"):
                assert np.array_equal(a[key][i], cpm[key][row]), "Clean pair changed"
            assert a["frame_index"][i] == cpm["frame_index"][row]
            for key in ("head_video", "right_wrist_video"):
                assert a[key].asstr()[i] == cpm.attrs[key]
                assert Path(cpm.attrs[key]).is_file()
        splits = a["split"].asstr()[:]
        train_ep = set(a["episode_index"][:][splits == "train"].tolist())
        val_ep = set(a["episode_index"][:][splits == "val"].tolist())
        assert not train_ep & val_ep
        recommended = np.all(~mask | eligible, axis=1) & np.all(
            ~mask | (np.rad2deg(s["rotation_change_rad"][:]) <= config["max_rotation_deg"]), axis=1
        )
        assert np.array_equal(recommended, s["recommended_training"][:])
        for split in ("train", "val"):
            for suffix, expected in (("", splits[idx] == split), ("_supported", (splits[idx] == split) & recommended)):
                with np.load(root / "index" / f"{split}{suffix}.npz", allow_pickle=False) as saved:
                    assert np.array_equal(saved["sample_index"], np.flatnonzero(expected))
                    assert np.array_equal(saved["anchor_index"], idx[expected])
        # Independent FK checks of saved start/endpoint and full interior paths.
        # Preserve the source passive thumb residual throughout recovery.
        replay_rows = np.unique(np.linspace(0, count - 1, 64).astype(int))
        max_tip_start_error, max_tip_end_error = 0.0, 0.0
        maximum_joint_speed, maximum_joint_acceleration = 0.0, 0.0
        for j in replay_rows:
            i = int(idx[j])
            full = s["full_qpos"][j]
            start_position, _ = k.fk(full)
            reference = kin["attempts/actual_position_world_m"][attempts[j]]
            max_tip_start_error = max(max_tip_start_error, float(np.max(np.abs(start_position - reference))))
            for target in absolute[j]:
                q = full.copy()
                q[k.state_qpos] = target
                q[k.thumb6_qpos] += float(target[10] - state[j, 10])
                positions, _ = k.fk(q)
                assert np.isfinite(positions).all()
                active = selected_joints[j]
                # The original goal may contain tiny recorded soft-limit
                # excursions, so compare with its explicit endpoint envelope.
                lower = np.minimum(k.lower, goal[j])
                upper = np.maximum(k.upper, goal[j])
                assert np.all(target[active] >= lower[active] - 2e-7)
                assert np.all(target[active] <= upper[active] + 2e-7)
            max_tip_end_error = max(max_tip_end_error, float(np.max(np.abs(positions - a["position"][i]))))
            velocity = np.diff(path[j], axis=0) * config["fps"]
            acceleration = np.diff(velocity, axis=0) * config["fps"]
            maximum_joint_speed = max(maximum_joint_speed, float(np.max(np.abs(velocity))))
            maximum_joint_acceleration = max(maximum_joint_acceleration, float(np.max(np.abs(acceleration))))
        assert max_tip_start_error < 1e-12
        assert max_tip_end_error < 1e-7
        # Zero-motion invariance is essential: synthesis must not alter a clean
        # map merely because the source has more than one physical contact.
        synthesis_config = TactileSynthesisConfig(
            release_distance_m=config["release_distance_mm"] / 1000,
            max_force_gain=config["max_force_gain"],
        )
        for i in range(len(a["state"])):
            identity = synthesize_tactile(
                a["tactile_map_raw"][i],
                a["eligible_fingers"][i],
                a["eligible_fingers"][i],
                np.zeros((5, 3)),
                np.zeros(5),
                a["direction_basis_world"][i],
                a["contact_centroid_world_m"][i],
                a["taxel_positions_world_m"][i],
                a["taxel_positions_world_m"][i],
                a["tangents_world"][i],
                a["tangents_world"][i],
                float(a["gaussian_sigma_m"][i]),
                synthesis_config,
            )
            assert np.array_equal(identity["tactile_map_raw"], a["tactile_map_raw"][i])
        # Analytic flat-grid translation checks independently verify the sign
        # of spatial warping and zero-force behavior at the release distance.
        column, row = np.meshgrid(np.linspace(-0.006, 0.006, 5), np.linspace(-0.009, 0.009, 7))
        mesh = np.stack([column, row, np.zeros_like(column)], axis=-1).reshape(35, 3)
        mesh = np.tile(mesh[None], (5, 1, 1))
        synthetic_base = np.zeros((5, 3, 7, 5), dtype=np.float32)
        profile = np.exp(-(column**2 + row**2) / (2 * 0.003**2))
        synthetic_base[:, 0] = profile / profile.sum()
        chart = np.tile(np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])[None], (5, 1, 1))
        tangent = np.tile(np.array([[1, 0, 0], [0, 1, 0]])[None], (5, 1, 1))
        moved = mesh + np.array([0.001, 0, 0])
        shifted = synthesize_tactile(
            synthetic_base,
            np.ones(5, dtype=bool),
            np.ones(5, dtype=bool),
            np.tile([0.001, 0, 0], (5, 1)),
            np.zeros(5),
            chart,
            np.zeros((5, 3)),
            mesh,
            moved,
            tangent,
            tangent,
            0.003,
            synthesis_config,
        )["tactile_map_raw"]
        assert np.all(np.sum(shifted[:, 0] * column, axis=(-2, -1)) < -0.0007)
        released = synthesize_tactile(
            synthetic_base,
            np.ones(5, dtype=bool),
            np.ones(5, dtype=bool),
            np.tile([0, 0, synthesis_config.release_distance_m], (5, 1)),
            np.zeros(5),
            chart,
            np.zeros((5, 3)),
            mesh,
            mesh + np.array([0, 0, synthesis_config.release_distance_m]),
            tangent,
            tangent,
            0.003,
            synthesis_config,
        )["tactile_map_raw"]
        assert not released.any()
        report = {
            "passed": True,
            "pairs_checked": count,
            "clean_pairs_checked": len(a["state"]),
            "fk_recovery_paths_checked": len(replay_rows),
            "fk_interior_targets_checked": len(replay_rows) * 30,
            "source_state_and_clean_actions_preserved": True,
            "episode_split_disjoint": True,
            "recovery_midpoint_max_error_rad": midpoint_error,
            "action_normalization_roundtrip_max_error_rad": norm_error,
            "normal_force_accounting_max_error_n": conservation_error,
            "start_fk_max_error_m": max_tip_start_error,
            "endpoint_fk_max_error_m": max_tip_end_error,
            "sampled_max_joint_speed_rad_per_s": maximum_joint_speed,
            "sampled_max_joint_acceleration_rad_per_s2": maximum_joint_acceleration,
            "identity_tactile_checks": len(a["state"]),
            "analytic_spatial_and_release_checks": True,
            "does_not_validate_real_contact_or_task_success": True,
        }
    (root / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT)
    run(parser.parse_args().data_root.resolve())
