"""Animate saved absolute recovery targets using FK and the existing tactile surrogate."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import sys
import tempfile

import h5py
import numpy as np
import pyarrow.parquet as pq
from scipy.spatial import ConvexHull

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from openpi.touchguide.fingertip_kinematics import FINGERS  # noqa: E402
from openpi.touchguide.fingertip_kinematics import JOINTS  # noqa: E402
from openpi.touchguide.fingertip_kinematics import FingertipKinematics  # noqa: E402
from openpi.touchguide.fingertip_recovery import TactileSynthesisConfig  # noqa: E402
from openpi.touchguide.fingertip_recovery import synthesize_tactile  # noqa: E402

DEFAULT = Path("/path/to/datasets/touchguidence/augmentation/poker_fingertip_recovery_rules_v1_preview")
MODEL = Path("/path/to/kaitactilesim/src/kaihand_tactile_env/tasks/poker_draw/scene.xml")


def box_mesh(size):
    vertices = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * size
    return vertices, ConvexHull(vertices).simplices


def scene_geometry(k):
    """Convex link shells from compiled model meshes; static scene uses actual boxes."""
    objects = []
    for b in range(k.model.nbody):
        name = k.model.body(b).name
        if not name or not name.startswith("hand_r_"):
            continue
        ids = np.flatnonzero(k.model.geom_bodyid == b)
        if not len(ids):
            continue
        g = int(ids[0])
        mid = int(k.model.geom_dataid[g])
        start, count = int(k.model.mesh_vertadr[mid]), int(k.model.mesh_vertnum[mid])
        vertices = k.model.mesh_vert[start : start + count].astype(np.float64)
        # Bound browser rendering cost while retaining the compiled link shape.
        # Farthest-point sampling covers the convex shell instead of dropping
        # arbitrary triangles or modifying the FK/state/taxel coordinates.
        hull_vertices = vertices[ConvexHull(vertices).vertices]
        if len(hull_vertices) > 96:
            distances = np.full(len(hull_vertices), np.inf)
            selected = []
            chosen = int(np.argmax(np.linalg.norm(hull_vertices - hull_vertices.mean(axis=0), axis=1)))
            for _ in range(96):
                selected.append(chosen)
                distances = np.minimum(
                    distances,
                    np.sum((hull_vertices - hull_vertices[chosen]) ** 2, axis=1),
                )
                chosen = int(np.argmax(distances))
            hull_vertices = hull_vertices[selected]
        vertices = hull_vertices
        hull = ConvexHull(vertices)
        used = np.unique(hull.simplices)
        lookup = {int(old): new for new, old in enumerate(used)}
        faces = [[lookup[int(i)] for i in face] for face in hull.simplices]
        finger = next((i for i, f in enumerate(FINGERS) if f"_{f}_" in name), -1)
        objects.append(
            {
                "name": name,
                "geom": g,
                "finger": finger,
                "vertices": vertices[used].tolist(),
                "faces": faces,
            }
        )
    for name in (
        "poker_table_base",
        "poker_table_top",
        "card_core_geom",
        "card_back_visual",
        "card_face_visual",
    ):
        g = k.model.geom(name).id
        vertices, faces = box_mesh(k.model.geom_size[g])
        objects.append(
            {
                "name": name,
                "geom": g,
                "finger": -2,
                "vertices": vertices.tolist(),
                "faces": faces.tolist(),
            }
        )
    return objects


def build_pair(args, k, clean, aug, raw, anchor, objects, variant):
    root = args.data_root
    config = TactileSynthesisConfig()
    row = args.row
    matches = np.flatnonzero((aug["source_clean_row"] == row) & (aug["variant_index"] == variant))
    if len(matches) != 1:
        raise ValueError("Expected exactly one pair for the requested row and variant")
    index = int(matches[0])
    goal, original = clean["state_raw"][row], clean["tactile_map_raw"][row]
    initial, stored_map = aug["state_raw"][index], aug["tactile_map_raw"][index]
    path = np.vstack([initial, aug["actions_absolute"][index]])
    mask = aug["perturbed_fingers"][index].astype(bool)
    requested, goals = (
        aug["requested_local_components_mm"][index],
        aug["requested_contact_changes"][index],
    )
    full_states, times = raw["full_states"], raw["times"]
    time_s = float(anchor["observation_state_timestamp_s"])
    right = min(int(np.searchsorted(times, time_s, side="left")), len(times) - 1)
    left = max(0, right - 1)
    weight = 0 if right == left else (time_s - times[left]) / (times[right] - times[left])
    full = (1 - weight) * full_states[left] + weight * full_states[right]
    assert np.max(np.abs(full[k.state_qpos] - goal)) <= 2e-7
    for j in range(k.model.njnt):
        if int(k.model.jnt_type[j]) == 0:
            address = int(k.model.jnt_qposadr[j]) + 3
            full[address : address + 4] /= np.linalg.norm(full[address : address + 4])
    full[k.state_qpos] = goal
    position, rotation = k.fk(full)
    eligible = np.asarray(anchor["eligible_fingers"], bool)
    basis = np.zeros((5, 3, 3))
    for finger in np.flatnonzero(eligible):
        basis[finger] = [
            anchor[key][finger]
            for key in (
                "contact_normal_away_world",
                "contact_tangent0_world",
                "contact_tangent1_world",
            )
        ]
    local_taxels, local_tangents = raw["local_taxels"], raw["local_tangents"]
    bodies, sigma = raw["bodies"], raw["sigma"]
    body_rotation = k.data.xmat[bodies].reshape(5, 3, 3).copy()
    body_position = k.data.xpos[bodies].copy()
    mesh_world = np.einsum("fij,ftj->fti", body_rotation, local_taxels) + body_position[:, None]
    mesh_site = np.einsum("fji,ftj->fti", rotation, mesh_world - position[:, None])
    tangent_world = np.einsum("fij,ftj->fti", body_rotation, local_tangents)
    tangent_site = np.einsum("fji,ftj->fti", rotation, tangent_world)
    centroid = np.array([[np.nan] * 3 if v is None else v for v in anchor["contact_centroid_world_m"]])
    frames, generated_maps = [], []
    for q in path:
        replay = full.copy()
        replay[k.state_qpos] = q
        replay[k.thumb6_qpos] += float(q[10] - goal[10])
        tips, rotations = k.fk(replay)
        mesh = np.einsum("fij,ftj->fti", rotations, mesh_site) + tips[:, None]
        tangents = np.einsum("fij,ftj->fti", rotations, tangent_site)
        rot_deg = np.rad2deg(np.arccos(np.clip((np.einsum("fij,fij->f", rotation, rotations) - 1) / 2, -1, 1)))
        synthesized = synthesize_tactile(
            original,
            mask,
            eligible,
            tips - position,
            rot_deg,
            basis,
            centroid,
            mesh_world,
            mesh,
            tangent_world,
            tangents,
            sigma,
            config,
        )
        assert synthesized["recommended_training"], "Intermediate frame exceeds tactile surrogate support"
        generated_maps.append(synthesized["tactile_map_raw"])
        poses = [
            {
                "p": k.data.geom_xpos[o["geom"]].tolist(),
                "r": k.data.geom_xmat[o["geom"]].tolist(),
            }
            for o in objects
        ]
        frames.append(
            {
                "poses": poses,
                "taxels_world": mesh.tolist(),
                "tips_world": tips.tolist(),
                "tip_local_mm": (np.einsum("fj,faj->fa", tips - position, basis) * 1000).tolist(),
            }
        )
    maps = np.asarray(generated_maps)
    initial_error = float(np.max(np.abs(maps[0] - stored_map)))
    endpoint_error = float(np.max(np.abs(maps[-1] - original)))
    assert initial_error < 1e-6
    assert endpoint_error < 1e-6
    # Display the saved observations exactly at the two endpoints. The start
    # recomputation differs only through the stored FP32 vs precise IK state.
    maps[0], maps[-1] = stored_map, original
    checks = {
        "endpoint_state_exact": bool(np.array_equal(path[-1], goal)),
        "arm_fixed": bool(np.array_equal(path[:, :7], np.broadcast_to(goal[:7], (len(path), 7)))),
        "inactive_joints_fixed": bool(
            np.array_equal(
                path[:, 7:].reshape(-1, 5, 4)[:, ~mask],
                np.broadcast_to(goal[7:].reshape(5, 4)[~mask], (len(path), sum(~mask), 4)),
            )
        ),
        "inactive_tactile_fixed": bool(
            np.array_equal(
                maps[:, ~mask],
                np.broadcast_to(original[~mask], maps[:, ~mask].shape),
            )
        ),
        "finite_data": bool(np.isfinite(path).all() and np.isfinite(maps).all()),
        "joint_limits": bool(np.all(path >= k.lower[None] - 1e-6) and np.all(path <= k.upper[None] + 1e-6)),
        "state_matches_saved_chunk": True,
    }
    assert all(checks.values()), checks
    for i, frame in enumerate(frames):
        frame["state"] = path[i].tolist()
        frame["tactile"] = maps[i].tolist()
        frame["force"] = maps[i].sum(axis=(-2, -1), dtype=np.float64).tolist()
    return {
        "root": str(root),
        "episode": args.episode,
        "row": args.row,
        "variant": variant,
        "frame": int(anchor["frame_index"]),
        "phase": anchor["demonstration_phase"],
        "fps": 30,
        "horizon": len(path) - 1,
        "joints": list(JOINTS),
        "mask": mask.tolist(),
        "requested_components_mm": requested.tolist(),
        "contact_goals": goals.tolist(),
        "frames": frames,
        "clean_state": goal.tolist(),
        "checks": checks,
        "synthesis_config": asdict(config),
        "sigma_m": sigma,
        "initial_recompute_max_error_n": initial_error,
        "endpoint_recompute_max_error_n": endpoint_error,
        "scene_mode": "FK only; frozen recorded card and table; no dynamics or collision/contact validation",
        "tactile_mode": "Saved endpoint observations; intermediate frames use the existing local contact surrogate",
    }


def export(args):
    root = args.data_root.resolve()
    records = pq.read_table(root / "geometry/episodes" / f"episode_{args.episode:06d}.parquet").to_pylist()
    anchor = next((record for record in records if int(record["sample_row"]) == args.row), None)
    if anchor is None:
        raise ValueError(f"No clean row {args.row} in episode {args.episode}")
    k = FingertipKinematics(args.model)
    objects = scene_geometry(k)
    with tempfile.TemporaryDirectory(prefix="recovery_pairs_", dir="/tmp") as scratch:
        local = Path(scratch)
        for part in ("clean", "augmented"):
            shutil.copyfile(
                root / part / "episodes" / f"episode_{args.episode:06d}.h5",
                local / f"{part}.h5",
            )
        with (
            h5py.File(local / "clean.h5") as clean_file,
            h5py.File(local / "augmented.h5") as aug_file,
        ):
            clean = {name: clean_file[name][:] for name in ("state_raw", "tactile_map_raw")}
            aug = {
                name: aug_file[name][:]
                for name in (
                    "source_clean_row",
                    "variant_index",
                    "state_raw",
                    "tactile_map_raw",
                    "actions_absolute",
                    "perturbed_fingers",
                    "requested_local_components_mm",
                    "requested_contact_changes",
                )
            }
        with h5py.File(anchor["source_raw_h5"]) as raw_file:
            columns = k.source_columns(raw_file["state/full_qpos_names"].asstr()[:].tolist())
            group = raw_file["tactile_contact_force"]
            names = group["link_names"].asstr()[:].tolist()
            links = [f"hand_r_{finger}_link{6 if finger == 'thumb' else 4}" for finger in FINGERS]
            ids = [names.index(name) for name in links]
            raw = {
                "full_states": raw_file["state/qpos"][:][:, columns],
                "times": raw_file["state/timestamp"][:],
                "local_taxels": group["taxel_positions_local_m"][:][ids],
                "local_tangents": group["tangent_basis_local"][:][ids],
                "bodies": [k.model.body(name).id for name in links],
                "sigma": float(json.loads(group.attrs["metadata_json"])["kernel_sigma_m"]),
            }
    available = sorted({int(v) for v in aug["variant_index"][aug["source_clean_row"] == args.row]})
    variants = available if args.variants == "all" else [int(v.strip()) for v in args.variants.split(",")]
    if not variants or len(set(variants)) != len(variants) or any(v not in available for v in variants):
        raise ValueError(f"Select unique available variants from {available}")
    pairs = []
    for variant in variants:
        print(
            f"Exporting variant {variant} ({len(pairs) + 1}/{len(variants)})",
            flush=True,
        )
        pairs.append(build_pair(args, k, clean, aug, raw, anchor, objects, variant))
    for obj in objects:
        del obj["geom"]
    default = variants.index(args.variant) if args.variant in variants else 0
    payload = {
        "schema_version": 2,
        "root": str(root),
        "objects": objects,
        "pairs": pairs,
        "default_pair": default,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    template = Path(__file__).with_name("recovery_pair_viewer.html").read_text()
    page = args.output_dir / "recovery_pair.html"
    temporary = page.with_suffix(".html.tmp")
    temporary.write_text(template.replace("__PAYLOAD__", encoded))
    temporary.replace(page)
    report = {
        "root": str(root),
        "episode": args.episode,
        "row": args.row,
        "frame": int(anchor["frame_index"]),
        "variants": variants,
        "default_variant": variants[default],
        "pair_count": len(pairs),
        "exported_frames": sum(len(pair["frames"]) for pair in pairs),
        "mesh_triangles": sum(len(obj["faces"]) for obj in objects),
        "file_bytes": page.stat().st_size,
        "pairs": [
            {
                key: pair[key]
                for key in (
                    "variant",
                    "mask",
                    "checks",
                    "initial_recompute_max_error_n",
                    "endpoint_recompute_max_error_n",
                )
            }
            for pair in pairs
        ],
        "scene_mode": pairs[0]["scene_mode"],
        "tactile_mode": pairs[0]["tactile_mode"],
    }
    (args.output_dir / "recovery_pair_review.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(
        json.dumps(
            {
                "page": str(page),
                "pairs": len(pairs),
                "frames": report["exported_frames"],
                "bytes": report["file_bytes"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--episode", type=int, default=7)
    parser.add_argument("--row", type=int, default=3)
    parser.add_argument("--variant", type=int, default=9, help="Initially selected variant")
    parser.add_argument(
        "--variants",
        default="2,4,9",
        help="all, or comma-separated variant indices to export",
    )
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = args.data_root / "visualization"
    export(args)


if __name__ == "__main__":
    main()
