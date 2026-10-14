# ruff: noqa: RUF001
"""Stage 2 pilot: contact-frame translations -> finger-local IK -> 27D states.

This generates kinematic proposals, not synthetic tactile maps, recovery chunks,
physical rollouts or score labels. Source data and the contact index stay intact.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections import defaultdict
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import sys
import time

import h5py
import mujoco
import numpy as np
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from openpi.touchguide.fingertip_kinematics import FINGERS  # noqa: E402
from openpi.touchguide.fingertip_kinematics import JOINTS  # noqa: E402
from openpi.touchguide.fingertip_kinematics import FingertipKinematics  # noqa: E402
from openpi.touchguide.fingertip_kinematics import IKConfig  # noqa: E402
from openpi.touchguide.fingertip_sampling import compose_finger_solutions  # noqa: E402
from openpi.touchguide.fingertip_sampling import direction_basis  # noqa: E402
from openpi.touchguide.fingertip_sampling import subset_patterns  # noqa: E402

DEFAULT_ROOT = Path("/path/to/datasets/touchguidence/augmentation/poker_fingertip_recovery_v1")
DEFAULT_SIM = Path("/path/to/kaitactilesim")
VERSION = "poker_fingertip_kinematic_pilot_v2_mixed"
AXES = (
    ("away", 0, 1),
    ("approach", 0, -1),
    ("tangent0_plus", 1, 1),
    ("tangent0_minus", 1, -1),
    ("tangent1_plus", 2, 1),
    ("tangent1_minus", 2, -1),
)


def write_json(path, content):
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def quantiles(values):
    x = np.asarray(values, dtype=np.float64).ravel()
    if not len(x):
        return None
    return dict(zip(("min", "p50", "p95", "max"), np.quantile(x, [0, 0.5, 0.95, 1]).tolist(), strict=True))


def select_anchors(table, count, seed):
    small = table.select(["split", "episode_index", "demonstration_phase", "eligible_fingers"]).to_pylist()
    groups = defaultdict(list)
    for index, row in enumerate(small):
        if any(row["eligible_fingers"]):
            groups[(row["demonstration_phase"], row["split"])].append(index)
    rng = np.random.default_rng(seed)
    selected = []
    for phase in sorted({r["demonstration_phase"] for r in small}):
        for split in ("train", "val"):
            candidates = groups[(phase, split)].copy()
            rng.shuffle(candidates)
            distinct = set()
            group_selected = []
            for index in candidates:
                episode = small[index]["episode_index"]
                if episode not in distinct:
                    group_selected.append(index)
                    distinct.add(episode)
                    if len(group_selected) == count:
                        break
            if len(group_selected) < count:
                raise ValueError(f"Insufficient distinct episodes in {phase}/{split}: {len(group_selected)} < {count}")
            selected.extend(group_selected)
    return table.take(selected)


def interp_qpos(raw, columns, timestamp):
    times = np.asarray(raw["state/timestamp"])
    if timestamp < times[0] - 1e-9 or timestamp > times[-1] + 1e-9:
        raise ValueError("Requested FK time outside recorded state range")
    right = int(np.searchsorted(times, timestamp, side="left"))
    right = min(right, len(times) - 1)
    left = max(0, right - 1)
    weight = 0.0 if right == left else float((timestamp - times[left]) / (times[right] - times[left]))
    return (1 - weight) * raw["state/qpos"][left][columns] + weight * raw["state/qpos"][right][columns]
    # Robot joints are scalar angles. Normalize interpolated free/ball quaternions
    # separately in the caller, rather than changing the arm/hand interpolation.


def normalize_free_quaternions(kinematics, qpos):
    for j in range(kinematics.model.njnt):
        kind = int(kinematics.model.jnt_type[j])
        start = int(kinematics.model.jnt_qposadr[j])
        if kind == int(mujoco.mjtJoint.mjJNT_FREE):
            part = qpos[start + 3 : start + 7]
        elif kind == int(mujoco.mjtJoint.mjJNT_BALL):
            part = qpos[start : start + 4]
        else:
            continue
        norm = np.linalg.norm(part)
        if norm < 1e-12:
            raise ValueError("Invalid source quaternion")
        part /= norm
    return qpos


def save_array(group, name, values):
    a = np.asarray(values)
    if a.dtype.kind in "OUS":
        group.create_dataset(name, data=a.astype(object), dtype=h5py.string_dtype("utf-8"))
    else:
        group.create_dataset(name, data=a, compression="gzip", shuffle=True)


def build(args):
    started = time.time()
    root = args.anchor_root.resolve()
    default_name = "pilot_v2_mixed" if args.combination_mode == "all_subsets_mixed" else "pilot_legacy"
    out = (args.output_dir or root / "kinematics" / default_name).resolve()
    # Every run has a unique destination; never overwrite a previous pilot.
    if out.exists():
        raise FileExistsError(f"Output already exists; choose a new --output-dir: {out}")
    configuration = json.loads((root / "meta/selection_config.json").read_text())
    if tuple(configuration["joint_order"]) != JOINTS:
        raise ValueError("Stage 1 joint ordering differs from the 27D IK contract")
    if args.selected_anchors:
        selected = pq.read_table(args.selected_anchors.resolve())
    else:
        table = pq.read_table(root / "index/contact_anchors.parquet")
        selected = select_anchors(table, args.anchors_per_phase_split, args.seed)
    if not len(selected) or len(set(selected["anchor_uid"].to_pylist())) != len(selected):
        raise ValueError("Selected anchors must be nonempty and unique")
    anchors = selected.to_pylist()
    sys.path.insert(0, str(args.sim_root.resolve() / "src"))
    from kaihand_tactile_env.shared.config import default_model_path  # noqa: PLC0415
    from kaihand_tactile_env.shared.config import model_fingerprint  # noqa: PLC0415

    model_path = (args.model_path or default_model_path("poker-draw")).resolve()
    fingerprint = model_fingerprint(model_path)
    model_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
    k = FingertipKinematics(model_path)
    ik_config = IKConfig(
        args.position_tolerance_mm * 1e-3, args.max_joint_delta_rad, args.posture_scale, args.max_evaluations
    )
    seed_rows = []
    attempts = []
    audit = []
    seen_split = {}
    local_ik_solves = 0
    print(
        f"Selected {len(anchors)} contact anchors across {len({a['demonstration_phase'] for a in anchors})} phases",
        flush=True,
    )
    with ExitStack() as stack:
        files = {}

        def source(path):
            if path not in files:
                files[path] = stack.enter_context(h5py.File(path, "r"))
            return files[path]

        for anchor_index, a in enumerate(anchors):
            if a["episode_index"] in seen_split and seen_split[a["episode_index"]] != a["split"]:
                raise ValueError("Episode leaked across train/val")
            seen_split[a["episode_index"]] = a["split"]
            raw = source(a["source_raw_h5"])
            metadata = json.loads(raw.attrs["metadata_json"])
            if metadata.get("base_model_fingerprint") != fingerprint or metadata.get("base_model_sha256") != model_sha:
                raise ValueError(f"Recorded base model does not match: {a['anchor_uid']}")
            columns = k.source_columns(raw["state/full_qpos_names"].asstr()[:].tolist())
            cpm = source(a["cpm_h5_path"])
            state = np.asarray(cpm["state_raw"][a["sample_row"]], dtype=np.float64)
            if state.shape != (27,) or not np.isfinite(state).all():
                raise ValueError("Invalid CPM original state")
            camera_time = float(a["observation_state_timestamp_s"])
            full_qpos = normalize_free_quaternions(k, interp_qpos(raw, columns, camera_time))
            state_error = float(np.max(np.abs(full_qpos[k.state_qpos] - state)))
            if state_error > 2e-7:
                raise ValueError(f"Camera/state alignment mismatch: {a['anchor_uid']} {state_error}")
            full_qpos[k.state_qpos] = state
            positions, rotations = k.fk(full_qpos)
            ci = int(a["source_camera_frame_index"])
            pose_time = float(raw["cameras/head/pose_timestamp"][ci])
            archived = np.asarray(raw["cameras/head/world_from_fingertip"][ci, 1])
            pose_qpos = normalize_free_quaternions(k, interp_qpos(raw, columns, pose_time))
            reconstructed, reconstructed_rot = k.fk(pose_qpos)
            archived_error = np.linalg.norm(reconstructed - archived[:, :3, 3], axis=1)
            if np.max(archived_error) > args.cached_fk_tolerance_mm * 1e-3:
                raise ValueError(
                    f"Cached FK reconstruction exceeds tolerance: {a['anchor_uid']} {archived_error * 1000} mm"
                )
            source_bounds = np.maximum(0, np.maximum(k.lower - state, state - k.upper))
            passive_offset = float(full_qpos[k.thumb6_qpos] - state[10])
            seed_rows.append(
                {
                    "state": state,
                    "full_qpos": full_qpos,
                    "position": positions,
                    "rotation": rotations,
                    "source_limit_violation_rad": source_bounds,
                }
            )
            audit.append(
                {
                    "anchor_uid": a["anchor_uid"],
                    "demonstration_phase": a["demonstration_phase"],
                    "state_interp_error_rad": state_error,
                    "cached_pose_reconstruction_error_mm": (archived_error * 1000).tolist(),
                    "cached_rotation_max_element_error": float(np.max(np.abs(reconstructed_rot - archived[:, :3, :3]))),
                    "cached_to_current_tip_motion_mm": (
                        np.linalg.norm(positions - archived[:, :3, 3], axis=1) * 1000
                    ).tolist(),
                    "observation_state_timestamp_s": camera_time,
                    "cached_pose_timestamp_s": pose_time,
                    "state_recording_period_s": float(np.median(np.diff(raw["state/timestamp"][:]))),
                    "passive_thumb6_minus_thumb5_rad": passive_offset,
                    "source_limit_violation_max_rad": float(source_bounds.max()),
                }
            )
            eligible = np.asarray(a["eligible_fingers"], dtype=bool)
            if args.combination_mode == "all_subsets_mixed":
                basis, basis_sources = direction_basis(a, raw, k, full_qpos)
                rng = np.random.default_rng(np.random.SeedSequence([args.seed, anchor_index]))
                patterns = list(subset_patterns(rng, args.mixed_patterns_per_subset))
            else:
                basis = np.zeros((5, 3, 3))
                basis_sources = ["not_used"] * 5
                for finger in np.flatnonzero(eligible):
                    basis[finger] = [
                        a["contact_normal_away_world"][finger],
                        a["contact_tangent0_world"][finger],
                        a["contact_tangent1_world"][finger],
                    ]
                    basis_sources[finger] = "measured_contact_chart"
                masks = [np.arange(5) == finger for finger in np.flatnonzero(eligible)]
                if eligible.sum() >= 2:
                    masks.append(eligible)
                patterns = []
                for mask in masks:
                    subset = int(sum((1 << int(finger)) for finger in np.flatnonzero(mask)))
                    for direction in range(6):
                        codes = np.full(5, -1, dtype=np.int8)
                        codes[mask] = direction
                        patterns.append((subset, mask, "uniform", codes))
            seed_rows[-1].update(
                direction_basis_world=basis,
                direction_basis_source=basis_sources,
                contact_fingers=np.asarray(a["contact_fingers"], dtype=bool),
                eligible_fingers=eligible,
            )
            cache = {}
            for subset, mask, pattern_kind, codes in patterns:
                names = "_".join(FINGERS[finger] for finger in np.flatnonzero(mask))
                mode = ("single_" if mask.sum() == 1 else "multi_subset_") + names
                axis_name = AXES[int(codes[mask][0])][0] if pattern_kind == "uniform" else "mixed"
                for amplitude_mm in args.magnitudes_mm:
                    coefficients = np.zeros((5, 3))
                    for finger in np.flatnonzero(mask):
                        _, axis_index, sign = AXES[int(codes[finger])]
                        coefficients[finger, axis_index] = sign * amplitude_mm * 1e-3
                    delta = np.einsum("fa,fai->fi", coefficients, basis)
                    solutions = {}
                    for finger in np.flatnonzero(mask):
                        key = (int(finger), int(codes[finger]), float(amplitude_mm))
                        if key not in cache:
                            single_delta = np.zeros((5, 3))
                            single_delta[finger] = delta[finger]
                            cache[key] = k.solve(state, full_qpos, single_delta, ik_config)
                            local_ik_solves += 1
                        solutions[int(finger)] = cache[key]
                    result = compose_finger_solutions(k, state, full_qpos, delta, solutions, ik_config)
                    unchanged = np.ones(27, dtype=bool)
                    for finger in np.flatnonzero(mask):
                        unchanged[7 + 4 * finger : 11 + 4 * finger] = False
                    if not np.array_equal(result["state"][unchanged], state[unchanged]):
                        raise AssertionError("IK changed the arm or an unselected finger")
                    if np.max(np.abs(result["state"] - state)) > args.max_joint_delta_rad + 1e-10:
                        raise AssertionError("IK exceeded its local joint trust region")
                    if np.max(np.linalg.norm(result["actual_displacement_world_m"][~mask], axis=1), initial=0) > 1e-12:
                        raise AssertionError("Unselected fingertip moved")
                    attempts.append(
                        dict(
                            anchor_index=anchor_index,
                            mode=mode,
                            axis=axis_name,
                            magnitude_mm=amplitude_mm,
                            coefficients=coefficients,
                            subset_code=subset,
                            direction_codes=codes.copy(),
                            pattern_kind=pattern_kind,
                            **result,
                        )
                    )
            print(
                f"Anchor {anchor_index + 1}/{len(anchors)}: {a['demonstration_phase']}/{a['split']} -> {len(attempts)} attempts",
                flush=True,
            )
    out.mkdir(parents=True)
    pq.write_table(selected, out / "selected_anchors.parquet", compression="zstd")
    with h5py.File(out / "perturbations.h5", "w") as f:
        f.attrs["schema"] = VERSION
        f.attrs["semantic"] = "kinematic proposal only; no tactile/action or success labels generated"
        f.attrs["joint_order_json"] = json.dumps(JOINTS)
        f.attrs["finger_order_json"] = json.dumps(FINGERS)
        f.attrs["full_qpos_names_json"] = json.dumps(k.labels)
        original = f.create_group("anchors")
        for key in seed_rows[0]:
            save_array(original, key, [r[key] for r in seed_rows])
        for key in (
            "anchor_uid",
            "split",
            "episode_index",
            "sample_row",
            "demonstration_phase",
            "cpm_h5_path",
            "source_raw_h5",
        ):
            save_array(original, key, [a[key] for a in anchors])
        proposals = f.create_group("attempts")
        for key in attempts[0]:
            save_array(proposals, key, [r[key] for r in attempts])
    successful = np.array([r["success"] for r in attempts])
    indices = np.array([r["anchor_index"] for r in attempts])
    split = np.array([anchors[i]["split"] for i in indices])
    np.savez_compressed(
        out / "valid_indices.npz",
        train=np.flatnonzero(successful & (split == "train")),
        val=np.flatnonzero(successful & (split == "val")),
        rejected=np.flatnonzero(~successful),
    )
    groups = defaultdict(lambda: [0, 0])
    for r in attempts:
        keys = (
            "phase:" + anchors[r["anchor_index"]]["demonstration_phase"],
            "axis:" + r["axis"],
            "mode:" + ("multi" if r["mode"].startswith("multi") else "single"),
            f"magnitude_mm:{r['magnitude_mm']:g}",
            f"subset:{r['subset_code']:02d}",
            "pattern_kind:" + r["pattern_kind"],
            f"finger_count:{int(r['perturbed_fingers'].sum())}",
        )
        for key in keys:
            groups[key][0] += 1
            groups[key][1] += int(r["success"])
    stats = {
        "schema": VERSION,
        "selected_anchors": len(anchors),
        "phases": dict(Counter(a["demonstration_phase"] for a in anchors)),
        "anchor_splits": dict(Counter(a["split"] for a in anchors)),
        "attempts": len(attempts),
        "attempts_per_anchor": dict(Counter(r["anchor_index"] for r in attempts)),
        "local_ik_solves": local_ik_solves,
        "valid_splits": {
            split_name: int(np.sum(successful & (split == split_name))) for split_name in ("train", "val")
        },
        "direction_basis_sources": dict(Counter(v for seed in seed_rows for v in seed["direction_basis_source"])),
        "valid": int(successful.sum()),
        "rejected": int((~successful).sum()),
        "valid_ratio": float(successful.mean()),
        "valid_error_mm": quantiles(
            [e * 1000 for r in attempts if r["success"] for e in r["error_m"][r["perturbed_fingers"]]]
        ),
        "all_requested_error_mm": quantiles([e * 1000 for r in attempts for e in r["error_m"][r["perturbed_fingers"]]]),
        "actual_displacement_mm": quantiles(
            [
                np.linalg.norm(d) * 1000
                for r in attempts
                if r["success"]
                for d in r["actual_displacement_world_m"][r["perturbed_fingers"]]
            ]
        ),
        "max_joint_delta_rad": max(
            float(np.max(np.abs(r["state"] - seed_rows[r["anchor_index"]]["state"]))) for r in attempts
        ),
        "phase_axis_mode_magnitude": {
            key: {"attempts": n, "valid": v, "valid_ratio": v / n} for key, (n, v) in sorted(groups.items())
        },
        "invariants": {
            "arm_and_unselected_joints_exactly_unchanged": True,
            "unselected_tips_unchanged": True,
            "valid_selected_joints_bounded": all(r["joint_bounds_ok"] for r in attempts if r["success"]),
            "original_episode_split_preserved": True,
        },
        "elapsed_seconds": time.time() - started,
    }
    failure_limits = Counter()
    for r in attempts:
        if r["success"]:
            continue
        state = seed_rows[r["anchor_index"]]["state"]
        selected_joints = np.repeat(r["perturbed_fingers"], 4)
        q = r["state"][7:][selected_joints]
        base = state[7:][selected_joints]
        physical = bool(np.any(np.minimum(q - k.lower[7:][selected_joints], k.upper[7:][selected_joints] - q) < 1e-6))
        local = bool(np.any(np.abs(q - base) >= args.max_joint_delta_rad - 1e-6))
        key = (
            "model_and_local_limit"
            if physical and local
            else "model_limit"
            if physical
            else "local_limit"
            if local
            else "residual_without_limit_hit"
        )
        failure_limits[key] += 1
    stats["rejected_limit_flags"] = dict(failure_limits)
    settings = {
        "schema": VERSION,
        "anchor_root": str(root),
        "output_dir": str(out),
        "model_path": str(model_path),
        "base_model_sha256": model_sha,
        "base_model_fingerprint": fingerprint,
        "combination_mode": args.combination_mode,
        "mixed_patterns_per_subset": args.mixed_patterns_per_subset,
        "selected_anchor_file": str(args.selected_anchors.resolve()) if args.selected_anchors else None,
        "direction_codes": {"-1": "unperturbed", **{str(i): axis[0] for i, axis in enumerate(AXES)}},
        "mixture_sampling": "six uniform patterns plus stratified unique mixtures per subset; common magnitude tier, independently selected finger directions",
        "direction_fallback": "recorded pad local axes transformed by current distal body FK; +normal is outward from pad, not an observed object contact normal",
        "ik_reuse": "fixed arm and disjoint finger chains: solve each finger/direction/magnitude once, compose and recompute FK",
        "implementation_sha256": {
            str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__),
                REPO / "src/openpi/touchguide/fingertip_kinematics.py",
                REPO / "src/openpi/touchguide/fingertip_sampling.py",
            )
        },
        "seed": args.seed,
        "anchors_per_phase_split": args.anchors_per_phase_split,
        "magnitudes_mm": args.magnitudes_mm,
        "position_tolerance_mm": args.position_tolerance_mm,
        "max_joint_delta_rad": args.max_joint_delta_rad,
        "posture_scale_m_per_rad": args.posture_scale,
        "max_evaluations": args.max_evaluations,
        "cached_fk_tolerance_mm": args.cached_fk_tolerance_mm,
        "target": "site position only; orientation is measured, not held constant",
        "thumb_coupling": "joint6 moves with joint5, preserving recorded source equality residual",
        "source_limits": "Unselected source joints retain recorded values, including tiny soft-limit excursions; selected joints obey model ranges",
        "cached_pose_audit": "Interpolate recorded qpos at pose_timestamp; 100Hz state cannot reconstruct every 500Hz physics pose exactly",
        "no_tactile_synthesis": True,
        "no_action_generation": True,
        "no_physics_rollout": True,
    }
    write_json(out / "config.json", settings)
    write_json(out / "summary.json", stats)
    write_json(
        out / "source_fk_audit.json",
        {
            "anchors": audit,
            "maximum_cached_reconstruction_error_mm": max(max(a["cached_pose_reconstruction_error_mm"]) for a in audit),
            "maximum_state_interp_error_rad": max(a["state_interp_error_rad"] for a in audit),
        },
    )
    (out / "README.md").write_text(
        "# 指尖扰动与逆解试制数据\n\n"
        "本目录只完成第二阶段运动学：接触坐标系位移 → 手指局部 IK → 27 维 state。\n"
        "手臂 7 维和未选中手指不变。原始数据、第一阶段索引、训练/验证拆分保持原样。\n\n"
        "- `selected_anchors.parquet`：原始接触样本及方向、时间戳、来源路径。\n"
        "- `perturbations.h5/anchors`：原 state、完整 qpos、重新计算的指尖位姿。\n"
        "- `perturbations.h5/attempts`：全部尝试；目标/实际位移、新 state、IK 误差、通过标志。\n"
        "- `valid_indices.npz`：只通过运动学检查的 train/val 行号；另保留 rejected。\n"
        "- `source_fk_audit.json`：模型与录制对应关系、时间插值和被动拇指残差。\n\n"
        "通过 IK 仅表示局部位置目标可实现；未生成扰动触觉、恢复动作或物理执行标签。\n"
        "失败尝试不能直接当作任务失败负样本。接触图表可能左手系，不能作为旋转矩阵。\n"
        "\n新版覆盖全部 31 个非空五指组合，混合方向分层采样，并使用共同的幅度档位。\n"
        "`anchors/direction_basis_source` 区分真实接触方向与无接触/不可靠接触的指尖阵列方向。\n"
        "后者的向内移动不表示一定发生新接触；未合成触觉。\n"
        "`attempts/subset_code` 使用五位 bitmask，低位到高位为 thumb/index/middle/ring/pinky。\n"
        "`attempts/direction_codes` 每指独立方向，-1 为不动；`pattern_kind` 区分 uniform/mixed。\n"
    )
    print(
        json.dumps(
            {
                k: stats[k]
                for k in ("selected_anchors", "attempts", "valid", "rejected", "valid_error_mm", "elapsed_seconds")
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    print("Saved:", out, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--anchor-root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--selected-anchors", type=Path, help="Reuse an existing selected_anchors.parquet")
    p.add_argument("--combination-mode", choices=("legacy", "all_subsets_mixed"), default="all_subsets_mixed")
    p.add_argument("--mixed-patterns-per-subset", type=int, default=6)
    p.add_argument("--sim-root", type=Path, default=DEFAULT_SIM)
    p.add_argument("--model-path", type=Path)
    p.add_argument("--anchors-per-phase-split", type=int, default=4)
    p.add_argument("--magnitudes-mm", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    p.add_argument("--position-tolerance-mm", type=float, default=0.05)
    p.add_argument("--max-joint-delta-rad", type=float, default=0.15)
    p.add_argument("--posture-scale", type=float, default=1e-4)
    p.add_argument("--max-evaluations", type=int, default=80)
    p.add_argument("--cached-fk-tolerance-mm", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if (
        args.anchors_per_phase_split < 1
        or not args.magnitudes_mm
        or not all(np.isfinite(x) and x > 0 for x in args.magnitudes_mm)
    ):
        p.error("anchor count and finite perturbation magnitudes must be positive")
    if not np.isfinite(args.cached_fk_tolerance_mm) or args.cached_fk_tolerance_mm <= 0:
        p.error("cached FK tolerance must be positive")
    if not 1 <= args.mixed_patterns_per_subset <= 30:
        p.error("mixed-patterns-per-subset must be between 1 and 30")
    if len(set(args.magnitudes_mm)) != len(args.magnitudes_mm):
        p.error("perturbation magnitude tiers must be unique")
    build(args)


if __name__ == "__main__":
    main()
