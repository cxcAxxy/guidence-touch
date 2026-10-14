"""Full-frame contact pairs and ten seeded, supported recovery variants per origin.

The three dataset entries are nested HDF5 virtual views. No training is run.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
from datetime import UTC
from datetime import datetime
import fcntl
from functools import cache
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

import h5py
import numpy as np
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from openpi.touchguide.fingertip_kinematics import FINGERS  # noqa: E402
from openpi.touchguide.fingertip_kinematics import FingertipKinematics  # noqa: E402
from openpi.touchguide.fingertip_kinematics import IKConfig  # noqa: E402
from openpi.touchguide.fingertip_randomization import feasible_finger_options  # noqa: E402
from openpi.touchguide.fingertip_randomization import random_finger_combinations  # noqa: E402
from openpi.touchguide.fingertip_recovery import TactileSynthesisConfig  # noqa: E402
from openpi.touchguide.fingertip_recovery import normalize_coordinates  # noqa: E402
from openpi.touchguide.fingertip_recovery import recovery_targets  # noqa: E402
from openpi.touchguide.fingertip_recovery import synthesize_tactile  # noqa: E402
from openpi.touchguide.fingertip_sampling import compose_finger_solutions  # noqa: E402
from openpi.touchguide.fingertip_sampling import direction_basis  # noqa: E402
from openpi.touchguide.fingertip_vector_rules import RULES  # noqa: E402
from openpi.touchguide.fingertip_vector_rules import feasible_vector_options  # noqa: E402
from openpi.touchguide.fingertip_vector_rules import random_vector_combinations  # noqa: E402

BASE = Path("/path/to/datasets/sim/poker-draw")
OLD = BASE / "pi05_trex/touchguide/cpm_datasets/poker_0920_200_touchguide_v1"
SIDE = BASE / "pi05_trex/0920_200"
DEFAULT = Path("/path/to/datasets/touchguidence/augmentation/poker_fingertip_recovery_v3_independent10")
MODEL = Path("/path/to/kaitactilesim/src/kaihand_tactile_env/tasks/poker_draw/scene.xml")
FIELDS = (
    "frame_index",
    "timestamp_s",
    "state_raw",
    "actions_absolute",
    "actions",
    "tactile_map_raw",
    "contact_fingers",
    "source_state_index",
    "source_camera_timestamp_s",
    "pair_kind",
    "variant_index",
)


def configure_sources(source_root, tactile_root, model_path):
    """Apply explicit input paths in parent and worker processes."""
    global OLD, SIDE, MODEL  # noqa: PLW0603
    OLD = Path(source_root).expanduser().resolve()
    SIDE = Path(tactile_root).expanduser().resolve()
    MODEL = Path(model_path).expanduser().resolve()
    kinematics.cache_clear()


@cache
def kinematics():
    return FingertipKinematics(MODEL)


def load_geometry_module():
    path = REPO / "scripts/touchguide/prepare_contact_anchors.py"
    spec = importlib.util.spec_from_file_location("contact_geometry_builder", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def publish(source, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copyfile(source, temp)
    temp.replace(destination)


def store(file, arrays):
    for name, value in arrays.items():
        array = np.asarray(value)
        if array.dtype.kind in "OUS":
            raise TypeError(f"String arrays are not part of the numeric episode schema: {name}")
        file.create_dataset(name, data=array, compression="gzip", compression_opts=1, shuffle=True)


def make_clean(task):
    root, meta, split, preview_limit = task
    eid = int(meta["episode_index"])
    marker = root / "clean/completed" / f"episode_{eid:06d}.json"
    if marker.exists():
        return read(marker)
    table = pq.read_table(
        meta["pi05_parquet"],
        columns=[
            "frame_index",
            "episode_index",
            "observation.state",
            "action",
            "timestamp",
            "provenance.source_state_index",
            "provenance.source_camera_timestamp_s",
            "provenance.source_camera_frame_index",
            "provenance.action_source_camera_frame_index",
        ],
    )

    def col(name, dtype):
        return np.asarray(table[name].to_pylist(), dtype=dtype)

    n = int(meta["num_frames"])
    frame = col("frame_index", np.int32)
    state, action = col("observation.state", np.float32), col("action", np.float32)
    assert np.array_equal(frame, np.arange(n))
    assert np.all(col("episode_index", np.int64) == eid)
    assert state.shape == action.shape == (n, 27)
    source_frames = col("provenance.source_camera_frame_index", np.int64)
    assert np.array_equal(col("provenance.action_source_camera_frame_index", np.int64), source_frames + 1)
    source_index = col("provenance.source_state_index", np.int64)
    camera_time = col("provenance.source_camera_timestamp_s", np.float64)
    with h5py.File(SIDE / meta["file"], "r") as side:
        maps = side["right_hand_tactile_map"][:]
        assert np.array_equal(side["frame_index"][:], frame)
        assert np.array_equal(side["source_state_index"][:], source_index)
        assert np.allclose(side["source_camera_timestamp_s"][:], camera_time, rtol=0, atol=1e-4)
    assert maps.shape == (n, 5, 3, 7, 5)
    assert np.all(maps[:, :, 0] >= 0)
    finite = np.isfinite(state).all(axis=1) & np.isfinite(maps).all(axis=(1, 2, 3, 4))
    valid_chunk = np.convolve(np.isfinite(action).all(axis=1).astype(int), np.ones(30, int), "valid") == 30
    contact = maps[:, :, 0].sum(axis=(-2, -1), dtype=np.float64) > 0.001
    chosen = np.flatnonzero(finite[: n - 29] & valid_chunk & contact[: n - 29].any(axis=1))
    all_contact_count = len(chosen)
    if preview_limit:
        chosen = chosen[np.unique(np.linspace(0, len(chosen) - 1, min(preview_limit, len(chosen))).astype(int))]
    norm = read(OLD / "meta/action_norm_stats.json")["norm_stats"]
    absolute = action[chosen[:, None] + np.arange(30)]
    physical = absolute.copy()
    physical[:, :, :7] -= state[chosen, None, :7]
    arrays = {
        "frame_index": chosen.astype(np.int32),
        "timestamp_s": col("timestamp", np.float64)[chosen],
        "state_raw": state[chosen],
        "actions_absolute": absolute,
        "actions": normalize_coordinates(physical, norm["actions"]),
        "tactile_map_raw": maps[chosen],
        "contact_fingers": contact[chosen],
        "source_state_index": source_index[chosen],
        "source_camera_timestamp_s": camera_time[chosen],
        "pair_kind": np.zeros(len(chosen), np.uint8),
        "variant_index": np.full(len(chosen), -1, np.int16),
    }
    old_manifest = read(OLD / "meta/manifest.json")
    pi05_root = Path(old_manifest["sources"]["pi05_root"]).expanduser().resolve()
    info = read(pi05_root / "meta/info.json")
    videos = {
        key: str(
            (
                pi05_root
                / info["video_path"].format(
                    episode_chunk=eid // info["chunks_size"], episode_index=eid, video_key=camera
                )
            ).resolve()
        )
        for key, camera in (
            ("head_video", "observation.images.head"),
            ("right_wrist_video", "observation.images.right_wrist"),
        )
    }
    assert all(Path(p).is_file() for p in videos.values())
    destination = root / "clean/episodes" / f"episode_{eid:06d}.h5"
    with tempfile.TemporaryDirectory(prefix="contact_clean_", dir="/tmp") as scratch:
        path = Path(scratch) / "episode.h5"
        with h5py.File(path, "w") as target:
            store(target, arrays)
            target.attrs.update(videos)
            target.attrs.update(
                episode_index=eid,
                split=split,
                source_parquet=meta["pi05_parquet"],
                source_tactile=str(SIDE / meta["file"]),
                source_num_frames=n,
                fps=30,
                action_horizon=30,
                action_dim=27,
                normalizer_sha256=old_manifest["sources"]["normalizer_sha256"],
            )
        with h5py.File(path, "r") as check:
            assert np.array_equal(check["state_raw"][:], state[chosen])
            assert np.array_equal(check["tactile_map_raw"][:], maps[chosen])
            assert np.array_equal(check["actions_absolute"][:], absolute)
            assert np.all(check["contact_fingers"][:].any(axis=1))
        publish(path, destination)
    result = {
        "episode_index": eid,
        "split": split,
        "pairs": len(chosen),
        "full_contact_pairs": all_contact_count,
        "bytes": destination.stat().st_size,
        "passed": True,
    }
    write(marker, result)
    return result


def clean_metadata(root, results):
    out = root / "clean"
    old = read(OLD / "meta/manifest.json")
    sources = read(OLD / "meta/sources.json")
    splits = {s: [r["episode_index"] for r in results if r["split"] == s] for s in ("train", "val")}
    write(out / "meta/splits.json", {"episode_indices": splits, "seed": 42})
    for name in ("action_norm_stats.json", "tactile_scale.json"):
        publish(OLD / "meta" / name, out / "meta" / name)
    write(
        out / "meta/manifest.json",
        {
            "schema": "touchguide_cpm_positive_entry_v1",
            "configuration": {
                "fps": 30,
                "stride": 1,
                "action_horizon": 30,
                "action_dim": 27,
                "seed": 42,
                "contact_normal_sum_threshold_n": 0.001,
            },
            "sources": old["sources"],
            "normalized_action_checkpoint_step": 20000,
            "splits": {
                s: {
                    "count": sum(r["pairs"] for r in results if r["split"] == s),
                    "episode_indices": ids,
                    "episodes": len(ids),
                }
                for s, ids in splits.items()
            },
            "image_storage": "original video references",
            "tactile_scale_policy": "reuse original train-only scale for controlled comparisons",
        },
    )
    write(
        out / "meta/sources.json", [m for m in sources if m["episode_index"] in {r["episode_index"] for r in results}]
    )
    for split, ids in splits.items():
        parts = []
        for eid in ids:
            with h5py.File(out / "episodes" / f"episode_{eid:06d}.h5", "r") as f:
                frames = f["frame_index"][:]
                parts.append(
                    {
                        "episode_index": np.full(len(frames), eid, np.int32),
                        "sample_row": np.arange(len(frames), dtype=np.int32),
                        "frame_index": frames,
                        "timestamp_s": f["timestamp_s"][:],
                    }
                )
        (out / "index").mkdir(exist_ok=True)
        np.savez_compressed(
            out / "index" / f"{split}.npz",
            **{
                key: np.concatenate([p[key] for p in parts])
                if parts
                else np.empty(0, dtype=np.float64 if key == "timestamp_s" else np.int32)
                for key in ("episode_index", "sample_row", "frame_index", "timestamp_s")
            },
        )


def geometry(task):
    root, meta, split = task
    eid = int(meta["episode_index"])
    marker = root / "geometry/completed" / f"episode_{eid:06d}.json"
    if marker.exists():
        return read(marker)
    module = load_geometry_module()
    with h5py.File(root / "clean/episodes" / f"episode_{eid:06d}.h5", "r") as f:
        force = f["tactile_map_raw"][:, :, 0].sum(axis=(-2, -1), dtype=np.float64)
        expected = len(force)
    args = SimpleNamespace(stride=1, max_clock_skew_s=1 / 30, minimum_normal_coherence=0.9, max_episodes=0)
    part = module.index_episode((root / "clean", SIDE, eid, meta, split, force, 0.001, args))
    destination = root / "geometry/episodes" / f"episode_{eid:06d}.parquet"
    table = module.pa.Table.from_pylist(part["records"], schema=module.schema())
    with tempfile.TemporaryDirectory(prefix="contact_geometry_", dir="/tmp") as scratch:
        path = Path(scratch) / "episode.parquet"
        pq.write_table(table, path, compression="zstd")
        assert pq.read_table(path).equals(table)
        publish(path, destination)
    result = {
        "episode_index": eid,
        "split": split,
        "clean_origins": expected,
        "geometry_origins": len(table),
        "excluded": dict(part["exclusions"]),
        "source_audits": part["audits"],
        "passed": True,
    }
    write(marker, result)
    return result


def sample_candidates(rng, eligible, policy="independent"):
    """Sample fingers with replacement in direction and amplitude space.

    Selected fingers may share either property; equality is not rejected.
    Unlike the legacy policy there is no shared magnitude or 50% shared-code
    branch. The finite-budget fallback stays within supported contact fingers.
    """
    fingers = np.flatnonzero(eligible)
    if not len(fingers):
        return
    if policy not in ("independent", "legacy"):
        raise ValueError(f"Unknown sampling policy: {policy}")
    for attempt in range(600):
        count = int(rng.integers(1, len(fingers) + 1)) if attempt < 300 else 1
        selected = rng.choice(fingers, count, replace=False)
        codes = np.full(5, -1, np.int8)
        magnitudes = np.zeros(5, np.float64)
        if policy == "independent":
            codes[selected] = rng.integers(0, 6, count)
            magnitudes[selected] = (
                rng.choice([0.5, 1.0, 2.0], count, replace=True) if attempt < 300 else rng.uniform(0.5, 2.0, count)
            )
        else:
            if count == 1 or rng.random() < 0.5:
                codes[selected] = rng.integers(0, 6)
            else:
                while True:
                    directions = rng.integers(0, 6, count)
                    if len(set(directions.tolist())) > 1:
                        break
                codes[selected] = directions
            magnitude = float(rng.choice([0.5, 1.0, 2.0])) if attempt < 300 else float(rng.uniform(0.5, 2.0))
            magnitudes[selected] = magnitude
        yield codes, magnitudes


def augment(task):
    root, meta, split, seed, sampling_policy = task
    eid = int(meta["episode_index"])
    marker = root / "augmented/completed" / f"episode_{eid:06d}.json"
    if marker.exists():
        return read(marker)
    k = kinematics()
    norm = read(root / "clean/meta/action_norm_stats.json")["norm_stats"]
    records = pq.read_table(root / "geometry/episodes" / f"episode_{eid:06d}.parquet").to_pylist()
    with h5py.File(root / "clean/episodes" / f"episode_{eid:06d}.h5", "r") as clean:
        clean_data = {name: clean[name][:] for name in FIELDS}
        attrs = dict(clean.attrs)
    n = len(clean_data["state_raw"])
    per_origin = np.zeros(n, np.int16)
    output, rejection, shortfalls = [], Counter(), []
    proposal_counts, feasibility_audits = Counter(), []
    max_ik_error = 0.0
    normal_error, normalized_error = 0.0, 0.0
    with h5py.File(meta["source_raw_h5"], "r") as raw:
        native_meta = json.loads(raw.attrs["metadata_json"])
        if native_meta["base_model_sha256"] != digest(MODEL):
            raise ValueError("Recorded base model differs")
        columns = k.source_columns(raw["state/full_qpos_names"].asstr()[:].tolist())
        times = raw["state/timestamp"][:]
        full_states = raw["state/qpos"][:][:, columns]
        group = raw["tactile_contact_force"]
        names = group["link_names"].asstr()[:].tolist()
        link_names = [f"hand_r_{f}_link{6 if f == 'thumb' else 4}" for f in FINGERS]
        ids = [names.index(name) for name in link_names]
        meshes = group["taxel_positions_local_m"][:][ids]
        local_tangents = group["tangent_basis_local"][:][ids]
        bodies = [k.model.body(name).id for name in link_names]
        sigma = float(json.loads(group.attrs["metadata_json"])["kernel_sigma_m"])
        for anchor in records:
            row = int(anchor["sample_row"])
            state = clean_data["state_raw"][row].astype(np.float64)
            time_s = float(anchor["observation_state_timestamp_s"])
            right = min(int(np.searchsorted(times, time_s, side="left")), len(times) - 1)
            left = max(0, right - 1)
            weight = 0.0 if right == left else (time_s - times[left]) / (times[right] - times[left])
            full = (1 - weight) * full_states[left] + weight * full_states[right]
            assert np.max(np.abs(full[k.state_qpos] - state)) <= 2e-7
            for j in range(k.model.njnt):
                if int(k.model.jnt_type[j]) == 0:
                    address = int(k.model.jnt_qposadr[j]) + 3
                    full[address : address + 4] /= np.linalg.norm(full[address : address + 4])
            full[k.state_qpos] = state
            position, rotation = k.fk(full)
            basis, _ = direction_basis(anchor, raw, k, full)
            k.fk(full)
            body_r = k.data.xmat[bodies].reshape(5, 3, 3).copy()
            body_p = k.data.xpos[bodies].copy()
            mesh_world = np.einsum("fij,ftj->fti", body_r, meshes) + body_p[:, None]
            mesh_site = np.einsum("fji,ftj->fti", rotation, mesh_world - position[:, None])
            tangent_world = np.einsum("fij,ftj->fti", body_r, local_tangents)
            tangent_site = np.einsum("fji,ftj->fti", rotation, tangent_world)
            centroid = np.array([[np.nan] * 3 if v is None else v for v in anchor["contact_centroid_world_m"]])
            eligible = np.asarray(anchor["eligible_fingers"], bool)
            rng = np.random.default_rng(np.random.SeedSequence([seed, eid, int(anchor["frame_index"])]))
            cache, attempted, signatures, variants = {}, set(), set(), []
            if sampling_policy in ("per_finger", "vector_components"):
                source_force = clean_data["tactile_map_raw"][row, :, 0].sum(axis=(-2, -1), dtype=np.float64)
                if sampling_policy == "vector_components":
                    banks, bank_audit = feasible_vector_options(
                        k, state, full, basis, eligible, source_force, cache, rng
                    )
                else:
                    banks, bank_audit = feasible_finger_options(k, state, full, basis, eligible, source_force, cache)
                bank_audit.update(
                    clean_row=row, frame_index=int(anchor["frame_index"]), eligible_fingers=eligible.tolist()
                )
                feasibility_audits.append(bank_audit)
                candidates = (
                    random_vector_combinations(rng, banks, proposal_counts)
                    if sampling_policy == "vector_components"
                    else random_finger_combinations(rng, banks, proposal_counts)
                )
            else:
                candidates = sample_candidates(rng, eligible, sampling_policy)
            for proposal in candidates:
                if isinstance(proposal, dict):
                    codes, magnitudes = proposal["codes"], proposal["magnitudes"]
                    components = proposal["components_mm"]
                    key = tuple(components.ravel())
                else:
                    codes, magnitudes = proposal
                    components = np.zeros((5, 3))
                    for finger in np.flatnonzero(codes >= 0):
                        code = int(codes[finger])
                        components[finger, code // 2] = magnitudes[finger] * (1 if code % 2 == 0 else -1)
                    key = (tuple(codes.tolist()), tuple(magnitudes.tolist()))
                if key in attempted:
                    continue
                attempted.add(key)
                mask = codes >= 0
                delta = np.zeros((5, 3))
                for finger in np.flatnonzero(mask):
                    if sampling_policy == "vector_components":
                        delta[finger] = components[finger] @ basis[finger] * 0.001
                    else:
                        code = int(codes[finger])
                        delta[finger] = (
                            basis[finger, code // 2] * (1 if code % 2 == 0 else -1) * magnitudes[finger] * 0.001
                        )
                solutions = {}
                for finger in np.flatnonzero(mask):
                    cache_key = (
                        (int(finger), tuple(components[finger]))
                        if sampling_policy == "vector_components"
                        else (int(finger), int(codes[finger]), float(magnitudes[finger]))
                    )
                    if cache_key not in cache:
                        single = np.zeros((5, 3))
                        single[finger] = delta[finger]
                        cache[cache_key] = k.solve(state, full, single, IKConfig())
                    solutions[int(finger)] = cache[cache_key]
                result = compose_finger_solutions(k, state, full, delta, solutions, IKConfig())
                if not result["success"]:
                    rejection["ik_or_joint_bounds"] += 1
                    continue
                if np.any(np.rad2deg(result["rotation_change_rad"])[mask] > 15):
                    rejection["pad_rotation"] += 1
                    continue
                target, _ = recovery_targets(result["state"], state)
                signature = target.tobytes()
                if signature in signatures:
                    rejection["duplicate_recovery_chunk"] += 1
                    continue
                signatures.add(signature)
                moved_mesh = (
                    np.einsum("fij,ftj->fti", result["actual_rotation_world"], mesh_site)
                    + result["actual_position_world_m"][:, None]
                )
                moved_tangent = np.einsum("fij,ftj->fti", result["actual_rotation_world"], tangent_site)
                synthesized = synthesize_tactile(
                    clean_data["tactile_map_raw"][row],
                    mask,
                    eligible,
                    result["actual_displacement_world_m"],
                    np.rad2deg(result["rotation_change_rad"]),
                    basis,
                    centroid,
                    mesh_world,
                    moved_mesh,
                    tangent_world,
                    moved_tangent,
                    sigma,
                    TactileSynthesisConfig(),
                )
                assert synthesized["recommended_training"]
                requested_contact_changes = (
                    proposal["contact_goals"]
                    if sampling_policy == "vector_components"
                    else np.where(mask, ((codes == 0) & (magnitudes > 1)).astype(np.int8), -1)
                )
                if sampling_policy in ("per_finger", "vector_components"):
                    actual_contact = synthesized["tactile_map_raw"][:, 0].sum(axis=(-2, -1), dtype=np.float64) > 0.001
                    assert np.all(actual_contact[mask] == (requested_contact_changes[mask] == 0))
                if sampling_policy == "vector_components":
                    assert (
                        np.max(np.linalg.norm(result["actual_displacement_world_m"][mask], axis=1)) * 1000
                        <= RULES["max_displacement_norm_mm"] + 1e-6
                    )
                active_joints = np.r_[np.zeros(7, bool), np.repeat(mask, 4)]
                assert np.array_equal(result["state"][~active_joints], state[~active_joints])
                assert np.array_equal(target[-1], state.astype(np.float32))
                assert np.array_equal(synthesized["tactile_map_raw"][~mask], clean_data["tactile_map_raw"][row][~mask])
                physical = target.copy()
                physical[:, :7] -= result["state"][:7].astype(np.float32)
                actions = normalize_coordinates(physical, norm["actions"])
                low, high = (np.asarray(norm["actions"][name]) for name in ("q01", "q99"))
                restored = (actions.astype(np.float64) + 1) / 2 * (high - low + 1e-6) + low
                norm_error = float(np.max(np.abs(restored - physical)))
                assert norm_error < 1e-6
                old_fn = clean_data["tactile_map_raw"][row, :, 0].sum(axis=(-2, -1), dtype=np.float64)
                new_map = synthesized["tactile_map_raw"]
                new_fn = new_map[:, 0].sum(axis=(-2, -1), dtype=np.float64)
                force_error = float(np.max(np.abs(new_fn - old_fn * synthesized["normal_force_gain"])))
                assert force_error < 1e-6
                if not variants:
                    replay = result["full_qpos"].copy()
                    replay[k.state_qpos] = target[-1]
                    replay[k.thumb6_qpos] += float(target[-1, 10] - result["state"][10])
                    end_position, _ = k.fk(replay)
                    assert np.max(np.abs(end_position - position)) < 1e-7
                values = {
                    "state_raw": result["state"].astype(np.float32),
                    "ik_state_raw": result["state"],
                    "actions_absolute": target,
                    "actions": actions,
                    "tactile_map_raw": new_map,
                    "contact_fingers": new_fn > 0.001,
                    "source_clean_row": row,
                    "eligible_fingers": eligible,
                    "perturbed_fingers": mask,
                    "direction_codes": codes,
                    "magnitude_mm": float(magnitudes.max()),
                    "finger_magnitudes_mm": magnitudes.copy(),
                    "requested_local_components_mm": components.copy(),
                    "requested_displacement_world_m": delta,
                    "actual_displacement_world_m": result["actual_displacement_world_m"],
                    "ik_error_m": result["error_m"],
                    "rotation_change_rad": result["rotation_change_rad"],
                    "normal_force_gain": synthesized["normal_force_gain"],
                    "recommended_training": True,
                }
                if sampling_policy in ("per_finger", "vector_components"):
                    values["requested_contact_changes"] = requested_contact_changes
                for name in ("frame_index", "timestamp_s", "source_state_index", "source_camera_timestamp_s"):
                    values[name] = clean_data[name][row]
                variants.append(values)
                max_ik_error = max(max_ik_error, float(np.max(result["error_m"][mask])))
                normal_error = max(normal_error, force_error)
                normalized_error = max(normalized_error, norm_error)
                if len(variants) == 10:
                    break
            for slot, index in enumerate(rng.permutation(len(variants))):
                values = variants[int(index)]
                values["pair_kind"], values["variant_index"] = np.uint8(1), np.int16(slot)
                output.append(values)
            per_origin[row] = len(variants)
            if len(variants) != 10:
                shortfalls.append(
                    {"clean_row": row, "frame_index": int(anchor["frame_index"]), "variants": len(variants)}
                )
    missing = np.flatnonzero(per_origin == 0)
    for row in missing:
        if not any(x["clean_row"] == row for x in shortfalls):
            shortfalls.append(
                {"clean_row": int(row), "frame_index": int(clean_data["frame_index"][row]), "variants": 0}
            )
    destination = root / "augmented/episodes" / f"episode_{eid:06d}.h5"
    with tempfile.TemporaryDirectory(prefix="contact_aug_", dir="/tmp") as scratch:
        path = Path(scratch) / "episode.h5"
        with h5py.File(path, "w") as f:
            if not output:
                raise ValueError(f"No valid augmentation in episode {eid}")
            arrays = {name: np.asarray([v[name] for v in output]) for name in output[0]}
            for value in arrays.values():
                assert np.isfinite(value).all()
            store(f, arrays)
            f.create_dataset("variants_per_clean_origin", data=per_origin)
            f.attrs.update(attrs)
            f.attrs["sampling_policy"] = sampling_policy
            f.attrs["magnitude_mm_semantics"] = (
                "maximum requested selected-finger magnitude; per-finger values are in finger_magnitudes_mm"
            )
            f.attrs["tactile_kind"] = "synthetic_local_contact_surrogate_not_physics_ground_truth"
            f.attrs["recovery_kind"] = "30_step_minimum_jerk_return_to_source_current_state_no_task_continuation"
        with h5py.File(path, "r") as f:
            assert len(f["state_raw"]) == int(per_origin.sum())
            assert np.all(f["recommended_training"][:])
            for name in FIELDS:
                assert len(f[name]) == len(output)
        publish(path, destination)
    mixing = Counter()
    transitions = Counter()
    for value in output:
        selected = value["perturbed_fingers"]
        components = value["requested_local_components_mm"][selected]
        unit = components / np.linalg.norm(components, axis=1, keepdims=True)
        directions = {tuple(np.round(v, 8)) for v in unit}
        has_normal = np.any(components[:, 0] != 0)
        has_tangent = np.any(components[:, 1:] != 0)
        mixing["variants_with_single_finger_normal_tangent"] += int(
            np.any((components[:, 0] != 0) & np.any(components[:, 1:] != 0, axis=1))
        )
        mixing["variants_with_single_finger_three_components"] += int(np.any(np.all(components != 0, axis=1)))
        amplitudes = value["finger_magnitudes_mm"][selected]
        if selected.sum() > 1:
            mixing["multi_finger"] += 1
            mixing["same_direction" if len(directions) == 1 else "different_directions"] += 1
            mixing["same_amplitude" if len(set(amplitudes.tolist())) == 1 else "different_amplitudes"] += 1
            mixing["mixed_normal_tangent"] += int(has_normal and has_tangent)
            mixing["directions_and_amplitudes_both_different"] += int(
                len(directions) > 1 and len(set(amplitudes.tolist())) > 1
            )
        else:
            mixing["single_finger"] += 1
        original = clean_data["contact_fingers"][value["source_clean_row"]]
        synthetic = value["contact_fingers"]
        lost, gained = original & ~synthetic, ~original & synthetic
        transitions["variants_with_contact_loss"] += int(lost.any())
        transitions["variants_with_contact_gain"] += int(gained.any())
        transitions["lost_finger_contacts"] += int(lost.sum())
        transitions["gained_finger_contacts"] += int(gained.sum())
    result = {
        "sampling_policy": sampling_policy,
        "mixing_counts": dict(mixing),
        "proposal_counts": dict(proposal_counts),
        "per_origin_feasibility": feasibility_audits,
        "synthetic_contact_transition_counts": dict(transitions),
        "episode_index": eid,
        "split": split,
        "clean_origins": n,
        "augmented_pairs": len(output),
        "shortfalls": shortfalls,
        "ten_variants_every_origin": bool(np.all(per_origin == 10)),
        "rejected_candidates": dict(rejection),
        "max_ik_error_m": max_ik_error,
        "normal_force_accounting_max_error_n": normal_error,
        "normalization_roundtrip_max_error_rad": normalized_error,
        "direction_histogram": dict(Counter(str(int(c)) for v in output for c in v["direction_codes"] if c >= 0)),
        "finger_count_histogram": dict(Counter(str(int(v["perturbed_fingers"].sum())) for v in output)),
        "bytes": destination.stat().st_size,
        "passed": True,
    }
    write(marker, result)
    print(f"Episode {eid}: {len(output)} supported pairs; shortfalls={len(shortfalls)}", flush=True)
    return result


def create_views(root, clean_results, aug_results):
    if any(not r["ten_variants_every_origin"] for r in aug_results):
        raise ValueError("Some origins have fewer than ten supported variants; ratios cannot be advertised as exact")
    counts = {}
    for ratio in (1, 5, 10):
        entry = root / f"ratio_1_{ratio}"
        indices = {s: [] for s in ("train", "val")}
        for record in clean_results:
            eid, split = record["episode_index"], record["split"]
            clean_path = root / "clean/episodes" / f"episode_{eid:06d}.h5"
            aug_path = root / "augmented/episodes" / f"episode_{eid:06d}.h5"
            path = entry / "episodes" / f"episode_{eid:06d}.h5"
            path.parent.mkdir(parents=True, exist_ok=True)
            n = record["pairs"]
            with h5py.File(clean_path, "r") as clean, h5py.File(aug_path, "r") as aug:
                assert len(aug["state_raw"]) == 10 * n
                assert np.array_equal(aug["source_clean_row"][:], np.repeat(np.arange(n), 10))
                assert np.array_equal(aug["variant_index"][:], np.tile(np.arange(10), n))
                with h5py.File(path.with_suffix(".h5.tmp"), "w", libver="latest") as view:
                    for name in FIELDS:
                        shape = (n * (1 + ratio), *clean[name].shape[1:])
                        layout = h5py.VirtualLayout(shape=shape, dtype=clean[name].dtype)
                        source = h5py.VirtualSource(str(clean_path), name, shape=clean[name].shape)
                        layout[:n] = source[:]
                        other = h5py.VirtualSource(str(aug_path), name, shape=aug[name].shape)
                        if ratio == 10:
                            layout[n:] = other[:]
                        else:
                            for slot in range(ratio):
                                layout[n + slot :: ratio] = other[slot::10]
                        view.create_virtual_dataset(name, layout)
                    view.attrs.update(dict(clean.attrs))
                    view.attrs.update(clean_augmented_ratio=f"1:{ratio}", shared_storage=True)
                path.with_suffix(".h5.tmp").replace(path)
                frames = np.r_[clean["frame_index"][:], np.repeat(clean["frame_index"][:], ratio)]
                timestamps = np.r_[clean["timestamp_s"][:], np.repeat(clean["timestamp_s"][:], ratio)]
                indices[split].append(
                    {
                        "episode_index": np.full(len(frames), eid, np.int32),
                        "sample_row": np.arange(len(frames), dtype=np.int32),
                        "frame_index": frames,
                        "timestamp_s": timestamps,
                    }
                )
                with h5py.File(path, "r") as view:
                    for row in (0, n, n + ratio - 1, len(frames) - 1):
                        source_row = row if row < n else ((row - n) // ratio) * 10 + (row - n) % ratio
                        expected = clean if row < n else aug
                        for name in FIELDS:
                            assert np.array_equal(view[name][row], expected[name][source_row])
        for split, parts in indices.items():
            (entry / "index").mkdir(exist_ok=True)
            data = {
                k: np.concatenate([p[k] for p in parts])
                if parts
                else np.empty(0, dtype=np.float64 if k == "timestamp_s" else np.int32)
                for k in ("episode_index", "sample_row", "frame_index", "timestamp_s")
            }
            np.savez_compressed(entry / "index" / f"{split}.npz", **data)
        for name in ("action_norm_stats.json", "tactile_scale.json", "splits.json", "sources.json"):
            publish(root / "clean/meta" / name, entry / "meta" / name)
        manifest = read(root / "clean/meta/manifest.json")
        for split in ("train", "val"):
            manifest["splits"][split]["clean_count"] = manifest["splits"][split]["count"]
            manifest["splits"][split]["augmented_count"] = manifest["splits"][split]["count"] * ratio
            manifest["splits"][split]["count"] *= 1 + ratio
        manifest.update(
            clean_augmented_ratio=f"1:{ratio}",
            storage="shared_clean_and_augmented_HDF5_virtual_views",
            nested_variant_slots=list(range(ratio)),
            training_started=False,
        )
        write(entry / "meta/manifest.json", manifest)
        write(
            entry / "dataset.json",
            {
                "ratio": f"1:{ratio}",
                "reader": "openpi.touchguide.data.CPMDataset",
                "root": str(entry),
                "split_counts": manifest["splits"],
                "requires_origin_unique_global_batch_sampler": True,
                "source_identity": "episode_index+frame_index; clean and all variants share this identity",
                "candidate_action_noise_applied_on_read": False,
            },
        )
        counts[str(ratio)] = {
            s: {k: manifest["splits"][s][k] for k in ("clean_count", "augmented_count", "count")}
            for s in ("train", "val")
        }
    return counts


def run_stage(root, name, function, tasks, workers):
    results = []
    write(
        root / "progress.json",
        {"status": "running", "stage": name, "completed_episodes": 0, "total_episodes": len(tasks)},
    )
    with ProcessPoolExecutor(
        max_workers=workers, initializer=configure_sources, initargs=(str(OLD), str(SIDE), str(MODEL))
    ) as pool:
        for future in as_completed([pool.submit(function, task) for task in tasks]):
            results.append(future.result())
            write(
                root / "progress.json",
                {
                    "status": "running",
                    "stage": name,
                    "completed_episodes": len(results),
                    "total_episodes": len(tasks),
                    "updated_at_utc": datetime.now(UTC).isoformat(),
                },
            )
            print(f"{name}: {len(results)}/{len(tasks)} episodes", flush=True)
    return sorted(results, key=lambda r: r["episode_index"])


def run(args):
    configure_sources(
        getattr(args, "source_root", OLD), getattr(args, "tactile_root", SIDE), getattr(args, "model_path", MODEL)
    )
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / "run.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        time.monotonic()
        episodes = [json.loads(s) for s in (SIDE / "meta/episodes.jsonl").read_text().splitlines() if s]
        old = read(OLD / "meta/manifest.json")
        splits = {i: s for s in ("train", "val") for i in old["splits"][s]["episode_indices"]}
        if args.episode_ids:
            wanted = set(args.episode_ids)
            episodes = [m for m in episodes if m["episode_index"] in wanted]
        config = {
            "schema": "full_contact_random10_nested_scales_v2",
            "sampling_policy": args.sampling_policy,
            "augmentation_rules": RULES if args.sampling_policy == "vector_components" else None,
            "reuse_base_root": str(args.reuse_base_root.resolve()) if args.reuse_base_root else None,
            "skip_ratio_views": args.skip_ratio_views,
            "episode_ids": [m["episode_index"] for m in episodes],
            "source_root": str(OLD),
            "source_manifest_sha256": digest(OLD / "meta/manifest.json"),
            "model_sha256": digest(MODEL),
            "seed": args.seed,
            "max_origins_per_episode": args.max_origins_per_episode,
            "contact_threshold_n": 0.001,
            "stride": 1,
            "action_horizon": 30,
            "variants_per_origin": 10,
            "magnitudes_mm": None
            if args.sampling_policy == "vector_components"
            else ([0.25, 0.5, 0.75, 1.0, 1.1, 1.5, 2.0] if args.sampling_policy == "per_finger" else [0.5, 1.0, 2.0]),
            "per_finger_perturb_probability": 0.65
            if args.sampling_policy in ("per_finger", "vector_components")
            else None,
            "per_finger_contact_goal": "conditional keep/release; verified after IK; no new contact"
            if args.sampling_policy in ("per_finger", "vector_components")
            else None,
            "fallback_magnitude_range_mm": None if args.sampling_policy == "vector_components" else [0.5, 2.0],
            "selected_fingers": "random nonempty subsets of reliable existing contact fingers",
            "directions": "signed normal/tangent1/tangent2 components including zeros, conditioned on rules"
            if args.sampling_policy == "vector_components"
            else "each selected finger independently samples away/approach/tangent0+/-/tangent1+/- with replacement; legacy optional",
            "amplitude_sampling": "per-finger with replacement; equality allowed",
            "magnitude_mm_semantics": "maximum selected-finger amplitude; finger_magnitudes_mm stores all five",
            "contact_transitions": "existing contact may be synthetically released; new contact is unsupported",
            "force_release_distance_mm": 1.0,
            "max_force_gain": 3.0,
            "max_rotation_deg": 15.0,
            "ratios": [1, 5, 10],
            "no_training": True,
            "synthetic_tactile_is_physics_ground_truth": False,
            "code_sha256": {
                str(p.relative_to(REPO)): digest(p)
                for p in (
                    Path(__file__).resolve(),
                    REPO / "scripts/touchguide/prepare_contact_anchors.py",
                    REPO / "src/openpi/touchguide/fingertip_kinematics.py",
                    REPO / "src/openpi/touchguide/fingertip_sampling.py",
                    REPO / "src/openpi/touchguide/fingertip_recovery.py",
                    REPO / "src/openpi/touchguide/fingertip_randomization.py",
                    REPO / "src/openpi/touchguide/fingertip_vector_rules.py",
                )
            },
        }
        if args.sampling_policy == "vector_components":
            write(root / "augmentation_rules_v1.json", RULES)
        if (root / "config.json").exists():
            if read(root / "config.json") != config:
                raise ValueError("Resume source/configuration/code mismatch; use a new destination")
        else:
            write(root / "config.json", config)
        if (root / "summary.json").exists() and read(root / "summary.json").get("status") == "complete":
            print("Already complete:", root, flush=True)
            return None
        try:
            if args.reuse_base_root:
                base = args.reuse_base_root.resolve()
                if base == root:
                    raise ValueError("Base-pool reuse requires a distinct destination")
                base_summary = read(base / "summary.json")
                base_config = read(base / "config.json")
                if (
                    base_summary["status"] != "complete"
                    or base_config["source_manifest_sha256"] != config["source_manifest_sha256"]
                    or base_config["model_sha256"] != config["model_sha256"]
                    or base_config["episode_ids"] != config["episode_ids"]
                    or base_config["max_origins_per_episode"] != config["max_origins_per_episode"]
                ):
                    raise ValueError("Base pool must be complete with the same sources and model")
                for part in ("clean", "geometry"):
                    target = root / part
                    if target.is_symlink():
                        if target.resolve() != (base / part).resolve():
                            raise ValueError("Base directory symlink mismatch")
                    elif target.exists():
                        raise ValueError("Refuse to replace an existing non-shared source directory")
                    else:
                        target.symlink_to(base / part, target_is_directory=True)
                clean_count = read(root / "clean/summary.json")["pairs"]
                if not args.episode_ids and not args.max_origins_per_episode and clean_count != 129196:
                    raise ValueError("Unexpected full-frame clean source count")
                print(f"Reusing {clean_count} immutable clean origins and contact geometry from {base}", flush=True)
            else:
                clean = run_stage(
                    root,
                    "clean",
                    make_clean,
                    [(root, m, splits[m["episode_index"]], args.max_origins_per_episode) for m in episodes],
                    args.workers,
                )
                clean_metadata(root, clean)
                clean_count = sum(r["pairs"] for r in clean)
                if not args.episode_ids and not args.max_origins_per_episode:
                    assert clean_count == 129196
                write(
                    root / "clean/summary.json",
                    {
                        "pairs": clean_count,
                        "bytes": sum(r["bytes"] for r in clean),
                        "splits": {s: sum(r["pairs"] for r in clean if r["split"] == s) for s in ("train", "val")},
                        "passed": True,
                    },
                )
                geom = run_stage(
                    root, "geometry", geometry, [(root, m, splits[m["episode_index"]]) for m in episodes], args.workers
                )
                write(
                    root / "geometry/summary.json",
                    {
                        "clean_origins": clean_count,
                        "geometry_origins": sum(r["geometry_origins"] for r in geom),
                        "episodes": geom,
                    },
                )
            aug = run_stage(
                root,
                "augmentation",
                augment,
                [(root, m, splits[m["episode_index"]], args.seed, args.sampling_policy) for m in episodes],
                args.workers,
            )
            write(
                root / "augmented/summary.json",
                {
                    "pairs": sum(r["augmented_pairs"] for r in aug),
                    "bytes": sum(r["bytes"] for r in aug),
                    "episodes": aug,
                },
            )
            if args.skip_ratio_views:
                report = {
                    "status": "complete",
                    "scope": "small_preview_no_ratio_entries",
                    "clean_pairs": clean_count,
                    "augmented_pairs": sum(r["augmented_pairs"] for r in aug),
                    "sampling_policy": args.sampling_policy,
                    "training_started": False,
                    "completed_at_utc": datetime.now(UTC).isoformat(),
                }
                write(root / "summary.json", report)
                write(root / "progress.json", report)
                print(json.dumps(report), flush=True)
                return None
            write(root / "progress.json", {"status": "running", "stage": "creating_ratio_views"})
            # Release run.lock on return before starting the separate, locally
            # staged parallel finalizer. No sequential NAS row validation here.
            return True
        except Exception as error:
            previous = read(root / "progress.json")
            previous.update(status="failed", error=repr(error), updated_at_utc=datetime.now(UTC).isoformat())
            write(root / "progress.json", previous)
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=OLD)
    parser.add_argument("--tactile-root", type=Path, default=SIDE)
    parser.add_argument("--model-path", type=Path, default=MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT)
    parser.add_argument(
        "--sampling-policy", choices=("independent", "legacy", "per_finger", "vector_components"), default="independent"
    )
    parser.add_argument("--reuse-base-root", type=Path)
    parser.add_argument("--skip-ratio-views", action="store_true", help="Small preview only; no ratio dataset entries")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episode-ids", type=int, nargs="*")
    parser.add_argument("--max-origins-per-episode", type=int, default=0)
    args = parser.parse_args()
    if args.workers < 1 or args.max_origins_per_episode < 0:
        parser.error("invalid worker or preview origin count")
    if args.skip_ratio_views and not (args.episode_ids or args.max_origins_per_episode):
        parser.error("skip-ratio-views is limited to small previews")
    if run(args):
        subprocess.run(
            [
                sys.executable,
                "-u",
                str(REPO / "scripts/touchguide/finalize_contact_scales.py"),
                "--output-dir",
                str(args.output_dir.resolve()),
                "--workers",
                str(args.workers),
            ],
            check=True,
        )
