"""Export self-contained, interactive review of actual contact recovery pairs."""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
from datetime import UTC
from datetime import datetime
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile

import h5py
import numpy as np
from PIL import Image
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from openpi.touchguide.data import _decode_video_frame  # noqa: E402
from openpi.touchguide.fingertip_kinematics import JOINTS  # noqa: E402
from openpi.touchguide.fingertip_kinematics import FingertipKinematics  # noqa: E402

DEFAULT = Path("/path/to/datasets/touchguidence/augmentation/poker_fingertip_recovery_v2_random10")
MODEL = Path("/path/to/kaitactilesim/src/kaihand_tactile_env/tasks/poker_draw/scene.xml")


def read(path):
    return json.loads(path.read_text())


def image_uri(path, frame):
    array = _decode_video_frame(str(path), int(frame), 30)
    image = Image.fromarray(array)
    image.thumbnail((640, 480))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def selected_records(records, count):
    phases = {}
    for record in records:
        phases.setdefault(record["demonstration_phase"], []).append(record)
    groups = list(phases.values())
    ids = np.unique(np.linspace(0, len(groups) - 1, min(count, len(groups))).astype(int))
    chosen = [groups[i][len(groups[i]) // 2] for i in ids]
    for i in np.linspace(0, len(records) - 1, count).astype(int):
        if len(chosen) >= count:
            break
        record = records[i]
        if record["sample_row"] not in {r["sample_row"] for r in chosen}:
            chosen.append(record)
    return sorted(chosen, key=lambda r: r["frame_index"])


def export_episode(task):
    root, eid, split, per_episode = task
    records = pq.read_table(root / "geometry/episodes" / f"episode_{eid:06d}.parquet").to_pylist()
    chosen = selected_records(records, per_episode)
    k = FingertipKinematics(MODEL)
    output = []
    with tempfile.TemporaryDirectory(prefix="contact_review_", dir="/tmp") as scratch:
        local = Path(scratch)
        for part in ("clean", "augmented"):
            shutil.copyfile(root / part / "episodes" / f"episode_{eid:06d}.h5", local / f"{part}.h5")
        with (
            h5py.File(local / "clean.h5") as clean,
            h5py.File(local / "augmented.h5") as aug,
            h5py.File(chosen[0]["source_raw_h5"]) as raw,
        ):
            columns = k.source_columns(raw["state/full_qpos_names"].asstr()[:].tolist())
            full_states = raw["state/qpos"][:][:, columns]
            times = raw["state/timestamp"][:]
            for anchor in chosen:
                row = int(anchor["sample_row"])
                state = clean["state_raw"][row]
                time_s = float(anchor["observation_state_timestamp_s"])
                right = min(int(np.searchsorted(times, time_s, side="left")), len(times) - 1)
                left = max(0, right - 1)
                weight = 0 if left == right else (time_s - times[left]) / (times[right] - times[left])
                full = (1 - weight) * full_states[left] + weight * full_states[right]
                assert np.max(np.abs(full[k.state_qpos] - state)) <= 2e-7
                for j in range(k.model.njnt):
                    if int(k.model.jnt_type[j]) == 0:
                        a = int(k.model.jnt_qposadr[j]) + 3
                        full[a : a + 4] /= np.linalg.norm(full[a : a + 4])
                full[k.state_qpos] = state
                original_tip, _ = k.fk(full)
                eligible = np.asarray(anchor["eligible_fingers"], bool)
                basis = np.zeros((5, 3, 3))
                for f in range(5):
                    if eligible[f]:
                        basis[f] = [
                            anchor[key][f]
                            for key in ("contact_normal_away_world", "contact_tangent0_world", "contact_tangent1_world")
                        ]
                original_map = clean["tactile_map_raw"][row]
                original_force = original_map[:, 0].sum(axis=(-2, -1), dtype=np.float64)
                variants = []
                for slot in range(10):
                    index = row * 10 + slot
                    assert int(aug["source_clean_row"][index]) == row
                    assert int(aug["variant_index"][index]) == slot
                    start = aug["state_raw"][index]
                    precise = aug["ik_state_raw"][index]
                    targets = aug["actions_absolute"][index]
                    path = np.vstack([start, targets])
                    mask = aug["perturbed_fingers"][index]
                    world = []
                    for q in path:
                        replay = full.copy()
                        replay[k.state_qpos] = q
                        replay[k.thumb6_qpos] += float(q[10] - state[10])
                        world.append(k.fk(replay)[0])
                    world = np.asarray(world)
                    assert np.max(np.abs(world[-1] - original_tip)) < 1e-7
                    displacement = aug["actual_displacement_world_m"][index]
                    replay = full.copy()
                    replay[k.state_qpos] = precise
                    replay[k.thumb6_qpos] += float(precise[10] - state[10])
                    assert np.max(np.abs(k.fk(replay)[0] - original_tip - displacement)) < 1e-10
                    local_path = np.einsum("tfj,faj->tfa", world - original_tip[None], basis) * 1000
                    maps = aug["tactile_map_raw"][index]
                    force = maps[:, 0].sum(axis=(-2, -1), dtype=np.float64)
                    gain = aug["normal_force_gain"][index]
                    joints_mask = np.r_[np.zeros(7, bool), np.repeat(mask, 4)]
                    checks = {
                        "手臂保持不变": bool(
                            np.array_equal(start[:7], state[:7])
                            and np.array_equal(targets[:, :7], np.broadcast_to(state[:7], (30, 7)))
                        ),
                        "未扰动关节保持不变": bool(np.array_equal(start[~joints_mask], state[~joints_mask])),
                        "未扰动触觉保持不变": bool(np.array_equal(maps[~mask], original_map[~mask])),
                        "恢复终点正确": bool(np.array_equal(targets[-1], state)),
                        "法向合力符合合成规则": bool(np.max(np.abs(force - original_force * gain)) < 1e-6),
                        "恢复段位于起终点之间": bool(
                            np.all(path >= np.minimum(start, state)[None] - 1e-7)
                            and np.all(path <= np.maximum(start, state)[None] + 1e-7)
                        ),
                    }
                    assert all(checks.values())
                    variants.append(
                        {
                            "slot": slot,
                            "magnitude_mm": float(aug["magnitude_mm"][index]),
                            "finger_magnitudes_mm": (
                                aug["finger_magnitudes_mm"][index].tolist()
                                if "finger_magnitudes_mm" in aug
                                else [float(aug["magnitude_mm"][index]) if selected else 0.0 for selected in mask]
                            ),
                            "requested_components_mm": aug["requested_local_components_mm"][index].tolist()
                            if "requested_local_components_mm" in aug
                            else None,
                            "requested_contact_changes": aug["requested_contact_changes"][index].tolist()
                            if "requested_contact_changes" in aug
                            else None,
                            "codes": aug["direction_codes"][index].tolist(),
                            "mask": mask.tolist(),
                            "tactile": maps.tolist(),
                            "force": force.tolist(),
                            "path": path.tolist(),
                            "tip_world_mm": (world * 1000).tolist(),
                            "tip_local_mm": local_path.tolist(),
                            "local_delta_mm": (np.einsum("fj,faj->fa", displacement, basis) * 1000).tolist(),
                            "rotation_rad": aug["rotation_change_rad"][index].tolist(),
                            "ik_error_m": aug["ik_error_m"][index].tolist(),
                            "checks": checks,
                        }
                    )
                output.append(
                    {
                        "episode": eid,
                        "split": split,
                        "row": row,
                        "frame": int(anchor["frame_index"]),
                        "time_s": float(clean["timestamp_s"][row]),
                        "phase": anchor["demonstration_phase"],
                        "state": state.tolist(),
                        "tactile": original_map.tolist(),
                        "force": original_force.tolist(),
                        "tip_world_mm": (original_tip * 1000).tolist(),
                        "head": image_uri(clean.attrs["head_video"], anchor["frame_index"]),
                        "wrist": image_uri(clean.attrs["right_wrist_video"], anchor["frame_index"]),
                        "variants": variants,
                    }
                )
    return output


def preview_png(sample, path):
    import matplotlib as mpl  # noqa: PLC0415

    mpl.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    v = max(sample["variants"], key=lambda v: sum(v["mask"]))
    f = int(np.flatnonzero(v["mask"])[0])
    old = np.asarray(sample["tactile"])[f, 0]
    new = np.asarray(v["tactile"])[f, 0]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    limit = max(old.max(), new.max(), 1e-10)
    for ax, label, a in zip(
        axes[0], ("Original normal", "Synthetic normal", "Synthetic - original"), (old, new, new - old), strict=True
    ):
        if label.startswith("Synthetic -"):
            bound = max(np.abs(a).max(), 1e-10)
            art = ax.imshow(a, vmin=-bound, vmax=bound, cmap="RdBu_r")
        else:
            art = ax.imshow(a, vmin=0, vmax=limit, cmap="viridis")
        ax.set_title(label)
        fig.colorbar(art, ax=ax, label="N / taxel")
    path_q = np.asarray(v["path"])
    goal = np.asarray(sample["state"])
    axes[1, 0].plot(path_q[:, 7 + 4 * f : 11 + 4 * f] - goal[None, 7 + 4 * f : 11 + 4 * f])
    axes[1, 0].set(title="Recovery joint error", xlabel="Step (0 = perturbed state)", ylabel="rad")
    axes[1, 1].plot(np.asarray(v["tip_local_mm"])[:, f])
    axes[1, 1].legend(["Away normal", "Tangent1", "Tangent2"])
    axes[1, 1].set(title="Actual FK recovery", xlabel="Step", ylabel="mm from clean tip")
    x = np.arange(5)
    axes[1, 2].bar(x - 0.18, sample["force"], 0.36, label="Original")
    axes[1, 2].bar(x + 0.18, v["force"], 0.36, label="Synthetic")
    axes[1, 2].set(
        xticks=x,
        xticklabels=["Thumb", "Index", "Middle", "Ring", "Pinky"],
        title="Five-finger normal force",
        ylabel="N",
    )
    axes[1, 2].legend()
    fig.suptitle(
        f"Episode {sample['episode']} / frame {sample['frame']} / variant {v['slot']} — local surrogate; not simulated execution"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(args):
    root = args.data_root.resolve()
    out = (args.output_dir or root / "visualization").resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest = read(root / "clean/meta/manifest.json")
    ids = args.episode_ids
    split_for = {e: s for s in ("train", "val") for e in manifest["splits"][s]["episode_indices"]}
    if ids is None:
        ids = []
        for split in ("train", "val"):
            episodes = manifest["splits"][split]["episode_indices"]
            ids.extend(
                episodes[i] for i in np.unique(np.linspace(0, len(episodes) - 1, min(2, len(episodes))).astype(int))
            )
    samples = []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(ids))) as pool:
        for future in as_completed(
            [pool.submit(export_episode, (root, e, split_for[e], args.anchors_per_episode)) for e in ids]
        ):
            batch = future.result()
            samples.extend(batch)
            print(
                f"Reviewed episode {batch[0]['episode']}: {len(batch)} anchors / {sum(len(s['variants']) for s in batch)} variants",
                flush=True,
            )
    samples.sort(key=lambda s: (s["split"], s["episode"], s["frame"]))
    totals = {
        "clean": read(root / "clean/summary.json")["pairs"],
        "augmented": read(root / "augmented/summary.json")["pairs"],
    }
    data = {
        "root": str(root),
        "exported_at": datetime.now(UTC).isoformat(),
        "totals": totals,
        "joints": list(JOINTS),
        "samples": samples,
        "rules": read(root / "augmentation_rules_v1.json") if (root / "augmentation_rules_v1.json").exists() else None,
    }
    template = (REPO / "scripts/touchguide/contact_augmentation_viewer.html").read_text()
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":")).replace("</", r"<\/")
    with tempfile.TemporaryDirectory(prefix="contact_review_output_", dir="/tmp") as scratch:
        local = Path(scratch)
        (local / "index.html").write_text(template.replace("__PAYLOAD__", payload))
        preview_png(samples[0], local / "preview.png")
        checks = {
            "reviewed_anchors": len(samples),
            "reviewed_variants": sum(len(s["variants"]) for s in samples),
            "episodes": ids,
            "all_sample_consistency_checks_passed": True,
            "all_variant_FK_endpoints_verified": True,
            "interactive_html": str(out / "index.html"),
            "preview_png": str(out / "preview.png"),
            "scope": "Representative examples; neither full physics validation nor execution success proof",
            "phase_counts": {
                phase: sum(s["phase"] == phase for s in samples) for phase in sorted({s["phase"] for s in samples})
            },
        }
        (local / "review.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2) + "\n")
        for name in ("index.html", "preview.png", "review.json"):
            temp = out / (name + ".tmp")
            shutil.copyfile(local / name, temp)
            temp.replace(out / name)
    print(json.dumps(checks, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--episode-ids", type=int, nargs="+")
    parser.add_argument("--anchors-per-episode", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.anchors_per_episode < 1 or args.workers < 1:
        parser.error("workers and anchors-per-episode must be positive")
    main(args)
