"""Build the small fingertip recovery pilot; never run training or simulation."""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import h5py
import numpy as np
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from openpi.touchguide.fingertip_kinematics import FINGERS  # noqa: E402
from openpi.touchguide.fingertip_kinematics import FingertipKinematics  # noqa: E402
from openpi.touchguide.fingertip_recovery import TactileSynthesisConfig  # noqa: E402
from openpi.touchguide.fingertip_recovery import normalize_coordinates  # noqa: E402
from openpi.touchguide.fingertip_recovery import recovery_targets  # noqa: E402
from openpi.touchguide.fingertip_recovery import synthesize_tactile  # noqa: E402

BASE = Path("/path/to/datasets/touchguidence/augmentation/poker_fingertip_recovery_v1")
SCHEMA = "poker_fingertip_recovery_pilot_v1_30steps"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def quantiles(values):
    x = np.asarray(values, dtype=np.float64).ravel()
    return dict(zip(("min", "p50", "p95", "max"), np.quantile(x, [0, 0.5, 0.95, 1]).tolist(), strict=True))


def store(group, name, value):
    array = np.asarray(value)
    if array.dtype.kind in "OUS":
        return group.create_dataset(name, data=array.astype(object), dtype=h5py.string_dtype("utf-8"))
    return group.create_dataset(name, data=array, compression="gzip", compression_opts=4, shuffle=True)


def csv_rows(path, rows):
    with Path(path).open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    started = time.monotonic()
    root = args.output_dir.resolve()
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite: {root}")
    kin_root = args.kinematic_root.resolve()
    kin_config = json.loads((kin_root / "config.json").read_text())
    source_config = json.loads((Path(kin_config["anchor_root"]) / "meta/selection_config.json").read_text())
    norm_path = Path(source_config["action_norm_stats_path"])
    if digest(norm_path) != source_config["action_norm_stats_sha256"]:
        raise ValueError("Original checkpoint statistics changed")
    stats = json.loads(norm_path.read_text())["norm_stats"]
    k = FingertipKinematics(kin_config["model_path"])
    if digest(k.model_path) != kin_config["base_model_sha256"]:
        raise ValueError("Original kinematic model changed")
    config = TactileSynthesisConfig(
        release_distance_m=args.release_distance_mm / 1000,
        max_force_gain=args.max_force_gain,
        max_rotation_deg=args.max_rotation_deg,
        contact_threshold_n=source_config["contact_threshold_n"],
    )
    selected = pq.read_table(kin_root / "selected_anchors.parquet").to_pylist()
    root.mkdir(parents=True)
    (root / "index").mkdir()
    (root / "plots").mkdir()
    shutil.copy2(norm_path, root / "action_norm_stats.json")
    shutil.copy2(kin_root / "selected_anchors.parquet", root / "selected_anchors.parquet")
    configuration = {
        "schema": SCHEMA,
        "kinematic_root": str(kin_root),
        "kinematic_h5_sha256": digest(kin_root / "perturbations.h5"),
        "output_dir": str(root),
        "action_horizon": 30,
        "fps": source_config["fps"],
        "recovery_duration_s": 30 / source_config["fps"],
        "recovery_interpolation": "minimum-jerk alpha(s)=10s^3-15s^4+6s^5; s=1/30..1; implicit start s=0",
        "action_convention": "fixed arm physical delta 7 + hand absolute targets 20; original quantile formula; no clipping",
        "normalized_state_convention": "original checkpoint state quantiles; raw state is also retained",
        "action_norm_stats_source": str(norm_path),
        "action_norm_stats_sha256": digest(norm_path),
        "normal_force_rule": "Fn_new/Fn_old=clip(1-actual_tip_displacement_dot_away/release_distance,0,max_force_gain)",
        "release_distance_mm": args.release_distance_mm,
        "release_distance_semantics": "user-adjustable synthetic compliance parameter; not measured or fitted stiffness",
        "max_force_gain": args.max_force_gain,
        "spatial_rule": "normalized ratios of recorded Gaussian 3D mesh distances to the measured fixed contact centroid",
        "shear_rule": "retain source world shear direction approximately; project into perturbed pad tangent axes; apply spatial weights and normal gain",
        "unmodeled_contact_rule": "keep source tactile when a perturbed finger lacks a measured contact chart; flag unsupported",
        "recommended_training_rule": f"all perturbed fingers have measured contact charts; rotation <= {args.max_rotation_deg} deg",
        "contact_threshold_n": config.contact_threshold_n,
        "max_rotation_deg": config.max_rotation_deg,
        "image_rule": "reference unchanged source head/wrist video frames; static small-displacement approximation",
        "clean_pair_rule": "retain original source state, tactile and original task action chunk",
        "proposed_clean_fraction": 0.5,
        "sampling_note": "training sampling is not implemented here; sample branch then anchor then one variant; enumerate pool independently of training weight",
        "synthetic_tactile_is_solver_ground_truth": False,
        "new_contact_predicted": False,
        "recovery_object_success_labels": False,
        "no_physics_rollout": True,
        "no_training": True,
        "implementation_sha256": {
            str(p.relative_to(REPO)): digest(p)
            for p in (REPO / "src/openpi/touchguide/fingertip_recovery.py", Path(__file__).resolve())
        },
    }
    write_json(root / "config.json", configuration)
    records = []
    with ExitStack() as stack:
        source = stack.enter_context(h5py.File(kin_root / "perturbations.h5", "r"))
        out = stack.enter_context(h5py.File(root / "recovery.h5", "w"))
        out.attrs["schema"] = SCHEMA
        out.attrs["tactile_kind"] = "synthetic_local_contact_surrogate_not_solver_ground_truth"
        original = source["anchors"]
        attempts = source["attempts"]
        indices = np.flatnonzero(attempts["success"][:])
        if not len(indices) or len(selected) != len(original["state"]):
            raise ValueError("Empty or mismatched source pilot")
        anchor_idx = attempts["anchor_index"][:][indices]
        anchors = out.create_group("anchors")
        for name in original:
            original.copy(name, anchors)
        n = len(indices)
        samples = out.create_group("samples")
        for name in (
            "anchor_index",
            "state",
            "full_qpos",
            "perturbed_fingers",
            "direction_codes",
            "subset_code",
            "magnitude_mm",
            "pattern_kind",
            "actual_displacement_world_m",
            "rotation_change_rad",
        ):
            dataset = attempts[name]
            store(
                samples,
                "state_raw" if name == "state" else name,
                dataset.asstr()[:][indices] if dataset.dtype.kind == "O" else dataset[:][indices],
            )
        store(samples, "source_attempt_index", indices)
        shapes = {
            "state_normalized": (n, 27),
            "actions_absolute": (n, 30, 27),
            "actions": (n, 30, 27),
            "tactile_map_raw": (n, 5, 3, 7, 5),
            "normal_displacement_m": (n, 5),
            "normal_force_gain": (n, 5),
            "spatial_profile_l1": (n, 5),
            "contact_fingers": (n, 5),
            "tactile_supported_fingers": (n, 5),
            "recommended_training": (n,),
        }
        for name, shape in shapes.items():
            dtype = (
                np.bool_
                if name in ("contact_fingers", "tactile_supported_fingers", "recommended_training")
                else np.float32
            )
            samples.create_dataset(name, shape=shape, dtype=dtype, compression="gzip", compression_opts=4, shuffle=True)
        samples.create_dataset("finger_synthesis_source", (n, 5), dtype=h5py.string_dtype("utf-8"))
        clean_maps, clean_actions, clean_absolute = [], [], []
        frames, head_videos, wrist_videos = [], [], []
        taxel_world, taxel_site, tangent_world, tangent_site, centroids, sigmas = [], [], [], [], [], []
        episode_split = {}
        for i, a in enumerate(selected):
            if original["anchor_uid"].asstr()[i] != a["anchor_uid"]:
                raise ValueError("Source anchor identity mismatch")
            if a["episode_index"] in episode_split and episode_split[a["episode_index"]] != a["split"]:
                raise ValueError("Source episode leaked across train/val")
            episode_split[a["episode_index"]] = a["split"]
            with h5py.File(a["cpm_h5_path"], "r") as cpm:
                row = a["sample_row"]
                if not np.array_equal(cpm["state_raw"][row], original["state"][i].astype(np.float32)):
                    raise ValueError("Clean source state changed")
                clean_maps.append(cpm["tactile_map_raw"][row])
                clean_actions.append(cpm["actions"][row])
                clean_absolute.append(cpm["actions_absolute"][row])
                frames.append(int(cpm["frame_index"][row]))
                head_videos.append(cpm.attrs["head_video"])
                wrist_videos.append(cpm.attrs["right_wrist_video"])
            position, rotation = k.fk(original["full_qpos"][i])
            with h5py.File(a["source_raw_h5"], "r") as raw:
                g = raw["tactile_contact_force"]
                meta = json.loads(g.attrs["metadata_json"])
                if meta["grid_shape"] != [7, 5] or meta["algorithm"] != "normalized_gaussian_3d_taxel_distance":
                    raise ValueError("Unsupported source taxel algorithm")
                sigmas.append(float(meta["kernel_sigma_m"]))
                ids = [
                    g["link_names"].asstr()[:].tolist().index(f"hand_r_{f}_link{6 if f == 'thumb' else 4}")
                    for f in FINGERS
                ]
                mesh = g["taxel_positions_local_m"][:][ids]
                tangents = g["tangent_basis_local"][:][ids]
                body_ids = [k.model.body(f"hand_r_{f}_link{6 if f == 'thumb' else 4}").id for f in FINGERS]
                body_r = k.data.xmat[body_ids].reshape(5, 3, 3).copy()
                body_p = k.data.xpos[body_ids].copy()
                world = np.einsum("fij,ftj->fti", body_r, mesh) + body_p[:, None]
                axes = np.einsum("fij,ftj->fti", body_r, tangents)
                taxel_world.append(world)
                taxel_site.append(np.einsum("fji,ftj->fti", rotation, world - position[:, None]))
                tangent_world.append(axes)
                tangent_site.append(np.einsum("fji,ftj->fti", rotation, axes))
                centroids.append([[np.nan] * 3 if v is None else v for v in a["contact_centroid_world_m"]])
        for name, value in (
            ("tactile_map_raw", clean_maps),
            ("actions", clean_actions),
            ("actions_absolute", clean_absolute),
            ("frame_index", frames),
            ("head_video", head_videos),
            ("right_wrist_video", wrist_videos),
            ("taxel_positions_site_m", taxel_site),
            ("tangents_site", tangent_site),
            ("taxel_positions_world_m", taxel_world),
            ("tangents_world", tangent_world),
            ("contact_centroid_world_m", centroids),
            ("gaussian_sigma_m", sigmas),
        ):
            store(anchors, name, value)
        clean_maps = np.asarray(clean_maps)
        taxel_world, taxel_site = np.asarray(taxel_world), np.asarray(taxel_site)
        tangent_world, tangent_site = np.asarray(tangent_world), np.asarray(tangent_site)
        centroids = np.asarray(centroids)
        basis, eligible = original["direction_basis_world"][:], original["eligible_fingers"][:]
        states = samples["state_raw"][:]
        all_goal = original["state"][:]
        positions = attempts["actual_position_world_m"][:][indices]
        rotations = attempts["actual_rotation_world"][:][indices]
        displacements = samples["actual_displacement_world_m"][:]
        rotation_deg = np.rad2deg(samples["rotation_change_rad"][:])
        masks = samples["perturbed_fingers"][:]
        for begin in range(0, n, 256):
            end = min(begin + 256, n)
            blocks = {name: [] for name in shapes}
            sources = []
            for j in range(begin, end):
                i = int(anchor_idx[j])
                absolute, alpha = recovery_targets(states[j], all_goal[i])
                physical = absolute.copy()
                physical[:, :7] -= states[j, :7].astype(np.float32)
                blocks["actions_absolute"].append(absolute)
                blocks["actions"].append(normalize_coordinates(physical, stats["actions"]))
                blocks["state_normalized"].append(normalize_coordinates(states[j], stats["state"]))
                new_mesh = np.einsum("fij,ftj->fti", rotations[j], taxel_site[i]) + positions[j, :, None]
                new_tangents = np.einsum("fij,ftj->fti", rotations[j], tangent_site[i])
                synthetic = synthesize_tactile(
                    clean_maps[i],
                    masks[j],
                    eligible[i],
                    displacements[j],
                    rotation_deg[j],
                    basis[i],
                    centroids[i],
                    taxel_world[i],
                    new_mesh,
                    tangent_world[i],
                    new_tangents,
                    sigmas[i],
                    config,
                )
                for name in (
                    "tactile_map_raw",
                    "normal_displacement_m",
                    "normal_force_gain",
                    "spatial_profile_l1",
                    "tactile_supported_fingers",
                    "recommended_training",
                ):
                    blocks[name].append(synthetic[name])
                sources.append(synthetic["finger_synthesis_source"])
                blocks["contact_fingers"].append(
                    synthetic["tactile_map_raw"][:, 0].sum(axis=(-2, -1), dtype=np.float64) > config.contact_threshold_n
                )
            for name, value in blocks.items():
                samples[name][begin:end] = np.asarray(value)
            samples["finger_synthesis_source"][begin:end] = np.asarray(sources, dtype=object)
            if begin % 4096 == 0 or end == n:
                print(f"Built {end}/{n} state/tactile/recovery pairs", flush=True)
        store(out, "interpolation_alpha", alpha)
        splits = original["split"].asstr()[:][anchor_idx]
        supported = samples["recommended_training"][:]
        for split in ("train", "val"):
            for suffix, chosen in (("", splits == split), ("_supported", (splits == split) & supported)):
                rows = np.flatnonzero(chosen)
                np.savez_compressed(
                    root / "index" / f"{split}{suffix}.npz",
                    sample_index=rows,
                    anchor_index=anchor_idx[rows],
                    source_attempt_index=indices[rows],
                )
        before = clean_maps[anchor_idx, :, 0].sum(axis=(-2, -1), dtype=np.float64)
        after = samples["tactile_map_raw"][:, :, 0].sum(axis=(-2, -1), dtype=np.float64)
        codes = samples["direction_codes"][:]
        supported_fingers = samples["tactile_supported_fingers"][:]
        magnitude = samples["magnitude_mm"][:]
        profile = samples["spatial_profile_l1"][:]
        direction_names = ("away", "approach", "tangent0_plus", "tangent0_minus", "tangent1_plus", "tangent1_minus")
        for code, name in enumerate(direction_names):
            for amplitude in np.unique(magnitude):
                selected_fingers = (codes == code) & masks & supported_fingers & (magnitude[:, None] == amplitude)
                if not selected_fingers.any():
                    continue
                old, new = before[selected_fingers], after[selected_fingers]
                records.append(
                    {
                        "direction": name,
                        "magnitude_mm": float(amplitude),
                        "finger_cases": int(selected_fingers.sum()),
                        "original_fn_mean_n": float(old.mean()),
                        "synthetic_fn_mean_n": float(new.mean()),
                        "force_ratio_mean": float((new / old).mean()),
                        "lost_contact_fraction": float((new <= config.contact_threshold_n).mean()),
                        "spatial_profile_l1_mean": float(profile[selected_fingers].mean()),
                    }
                )
        csv_rows(root / "force_change_by_direction.csv", records)
        transition = np.zeros((6, 6), dtype=np.int64)
        np.add.at(
            transition,
            ((before > config.contact_threshold_n).sum(axis=1), (after > config.contact_threshold_n).sum(axis=1)),
            1,
        )
        clean_future = np.asarray(clean_absolute)
        endpoint_future_distance = np.max(np.abs(clean_future[:, -1] - all_goal.astype(np.float32)), axis=1)
        summary = {
            "schema": SCHEMA,
            "clean_anchors": len(selected),
            "original_episodes": len(episode_split),
            "augmented_pairs": n,
            "splits": {
                s: {
                    "pairs": int(np.sum(splits == s)),
                    "supported_pairs": int(np.sum((splits == s) & supported)),
                    "clean_anchors": int(np.sum(original["split"].asstr()[:] == s)),
                }
                for s in ("train", "val")
            },
            "supported_pairs": int(supported.sum()),
            "unsupported_pairs": int((~supported).sum()),
            "unsupported_meaning": "a selected finger lacks a measured contact chart, or exceeds the configured pad rotation range; not a task failure label",
            "supported_is_physical_ground_truth": False,
            "by_phase": dict(Counter(original["demonstration_phase"].asstr()[:][anchor_idx])),
            "by_finger_count": {str(c): int(np.sum(masks.sum(axis=1) == c)) for c in range(1, 6)},
            "by_pattern": dict(Counter(samples["pattern_kind"].asstr()[:])),
            "normal_force_gain": quantiles(samples["normal_force_gain"][:][masks & supported_fingers]),
            "joint_recovery_delta_rad": quantiles(np.abs(states - all_goal[anchor_idx])),
            "original_future_endpoint_vs_recovery_goal_max_delta_rad": quantiles(endpoint_future_distance),
            "contact_count_transition_original_row_synthetic_column": transition.tolist(),
            "force_changes": records,
            "normalized_action_outside_unit_fraction": float(np.mean(np.abs(samples["actions"][:]) > 1)),
            "normalized_state_outside_unit_fraction": float(np.mean(np.abs(samples["state_normalized"][:]) > 1)),
            "source_videos_copied": False,
            "training_started": False,
            "runtime_s": time.monotonic() - started,
        }
        write_json(root / "summary.json", summary)
    (root / "README.md").write_text(
        "# Fingertip recovery pilot\n\n"
        "Synthetic state/tactile proposals with 30 future joint targets returning to the source current state.\n"
        "No task continuation is appended. Clean anchors retain original task chunks.\n\n"
        "- `recovery.h5`: anchors (clean pairs and video references), samples (noisy state, synthetic tactile, recovery actions).\n"
        "- `index/train.npz`, `val.npz`: all kinematically accepted variants, original episode split.\n"
        "- `index/*_supported.npz`: only variants whose selected fingers have measured contact charts and small pad rotation.\n"
        "- Supported synthetic samples are still approximations, not solver-verified contact or successful recovery labels.\n"
        "- Missing contact charts keep the original map and are explicitly flagged unsupported. New contacts are not invented.\n"
        "- Vision uses the original head/wrist video frame; no videos are copied.\n"
        "- Original checkpoint quantiles and unclipped normalization are saved in `action_norm_stats.json`.\n"
        "- The existing CPM training loader is unchanged; this pilot is for inspecting augmentation before training integration.\n"
        "- A proposed clean:augmented sampling ratio is 1:1; this builder does not set training weights.\n"
        "- Force compliance is configurable (default release distance 1 mm), not an identified physical model.\n"
    )
    print(
        json.dumps(
            {k: summary[k] for k in ("clean_anchors", "augmented_pairs", "supported_pairs", "splits", "runtime_s")},
            ensure_ascii=False,
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kinematic-root", type=Path, default=BASE / "kinematics/pilot_v2_mixed")
    parser.add_argument("--output-dir", type=Path, default=BASE / "recovery/pilot_v1_30steps")
    parser.add_argument("--release-distance-mm", type=float, default=1.0)
    parser.add_argument("--max-force-gain", type=float, default=3.0)
    parser.add_argument("--max-rotation-deg", type=float, default=15.0)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
