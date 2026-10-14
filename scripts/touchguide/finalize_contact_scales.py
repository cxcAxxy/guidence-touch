"""Finish nested ratio entries from already completed clean and augmented pools.

Stage each source episode on local disk, validate all rows locally, then publish
small virtual views with the original NAS source paths. Does not create pairs.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
from datetime import UTC
from datetime import datetime
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import time

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "contact_scale_builder", REPO / "scripts/touchguide/prepare_contact_scales.py"
)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def make_view(path, clean, aug, clean_path, aug_path, ratio):
    n = len(clean["state_raw"])
    with h5py.File(path, "w", libver="latest") as view:
        for name in builder.FIELDS:
            layout = h5py.VirtualLayout((n * (1 + ratio), *clean[name].shape[1:]), clean[name].dtype)
            source = h5py.VirtualSource(str(clean_path), name, shape=clean[name].shape)
            other = h5py.VirtualSource(str(aug_path), name, shape=aug[name].shape)
            layout[:n] = source[:]
            if ratio == 10:
                layout[n:] = other[:]
            else:
                for slot in range(ratio):
                    layout[n + slot :: ratio] = other[slot::10]
            view.create_virtual_dataset(name, layout)
        view.attrs.update(dict(clean.attrs))
        view.attrs.update(clean_augmented_ratio=f"1:{ratio}", shared_storage=True)


def episode(task):
    root, eid, split, n = task
    name = f"episode_{eid:06d}.h5"
    clean_path = root / "clean/episodes" / name
    aug_path = root / "augmented/episodes" / name
    with tempfile.TemporaryDirectory(prefix="ratio_finalizer_", dir="/tmp") as scratch_dir:
        scratch = Path(scratch_dir)
        local_clean, local_aug = scratch / "clean.h5", scratch / "aug.h5"
        shutil.copyfile(clean_path, local_clean)
        shutil.copyfile(aug_path, local_aug)
        with h5py.File(local_clean, "r") as clean, h5py.File(local_aug, "r") as aug:
            assert len(clean["state_raw"]) == n
            assert len(aug["state_raw"]) == 10 * n
            assert np.array_equal(aug["source_clean_row"][:], np.repeat(np.arange(n), 10))
            assert np.array_equal(aug["variant_index"][:], np.tile(np.arange(10), n))
            assert np.all(aug["variants_per_clean_origin"][:] == 10)
            assert np.all(aug["recommended_training"][:])
            frames, timestamps = clean["frame_index"][:], clean["timestamp_s"][:]
            for ratio in (1, 5, 10):
                shadow, production = scratch / f"shadow_{ratio}.h5", scratch / f"production_{ratio}.h5"
                make_view(shadow, clean, aug, local_clean, local_aug, ratio)
                make_view(production, clean, aug, clean_path, aug_path, ratio)
                with h5py.File(shadow, "r") as test, h5py.File(production, "r") as actual:
                    for field in builder.FIELDS:
                        c, a = clean[field][:], aug[field][:]
                        expected = np.concatenate(
                            [c, a.reshape(n, 10, *a.shape[1:])[:, :ratio].reshape(-1, *a.shape[1:])]
                        )
                        assert np.array_equal(test[field][:], expected)
                        assert actual[field].shape == test[field].shape
                        assert actual[field].dtype == test[field].dtype
                        mappings = actual[field].virtual_sources()
                        shadow_mappings = test[field].virtual_sources()
                        assert len(mappings) == len(shadow_mappings)
                        for i, (mapping, reference) in enumerate(zip(mappings, shadow_mappings, strict=True)):
                            assert mapping.file_name == str(clean_path if i == 0 else aug_path)
                            assert mapping.dset_name == reference.dset_name == field
                            assert mapping.vspace.encode() == reference.vspace.encode()
                            assert mapping.src_space.encode() == reference.src_space.encode()
                builder.publish(production, root / f"ratio_1_{ratio}/episodes" / name)
    return {"episode_index": eid, "split": split, "pairs": n, "frame_index": frames, "timestamp_s": timestamps}


def run(args):
    root = args.output_dir.resolve()
    started = time.monotonic()
    with (root / "run.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "summary.json").exists() and not args.force:
            print("Already complete", flush=True)
            return
        clean_summary = builder.read(root / "clean/summary.json")
        aug_summary = builder.read(root / "augmented/summary.json")
        assert clean_summary["passed"]
        assert aug_summary["pairs"] == clean_summary["pairs"] * 10
        assert all(
            r["ten_variants_every_origin"] and not r["shortfalls"] and r["passed"] for r in aug_summary["episodes"]
        )
        tasks = []
        for split in ("train", "val"):
            with np.load(root / "clean/index" / f"{split}.npz", allow_pickle=False) as index:
                episodes, counts = np.unique(index["episode_index"], return_counts=True)
                tasks.extend((root, int(e), split, int(n)) for e, n in zip(episodes, counts, strict=True))
        assert sum(t[3] for t in tasks) == clean_summary["pairs"]
        records = []
        try:
            builder.write(
                root / "progress.json",
                {
                    "status": "running",
                    "stage": "creating_ratio_views_parallel",
                    "completed_episodes": 0,
                    "total_episodes": len(tasks),
                },
            )
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                for future in as_completed([pool.submit(episode, task) for task in tasks]):
                    records.append(future.result())
                    builder.write(
                        root / "progress.json",
                        {
                            "status": "running",
                            "stage": "creating_ratio_views_parallel",
                            "completed_episodes": len(records),
                            "total_episodes": len(tasks),
                            "updated_at_utc": datetime.now(UTC).isoformat(),
                        },
                    )
                    print(f"All three views validated: {len(records)}/{len(tasks)} episodes", flush=True)
            records.sort(key=lambda r: r["episode_index"])
            counts = {}
            for ratio in (1, 5, 10):
                entry = root / f"ratio_1_{ratio}"
                for split in ("train", "val"):
                    parts = []
                    for record in records:
                        if record["split"] != split:
                            continue
                        n, eid = record["pairs"], record["episode_index"]
                        frames = np.r_[record["frame_index"], np.repeat(record["frame_index"], ratio)]
                        timestamps = np.r_[record["timestamp_s"], np.repeat(record["timestamp_s"], ratio)]
                        parts.append(
                            {
                                "episode_index": np.full(len(frames), eid, np.int32),
                                "sample_row": np.arange(len(frames), dtype=np.int32),
                                "frame_index": frames,
                                "timestamp_s": timestamps,
                            }
                        )
                    data = {
                        key: np.concatenate([p[key] for p in parts])
                        if parts
                        else np.empty(0, dtype=np.float64 if key == "timestamp_s" else np.int32)
                        for key in ("episode_index", "sample_row", "frame_index", "timestamp_s")
                    }
                    assert len(data["episode_index"]) == clean_summary["splits"][split] * (1 + ratio)
                    with tempfile.TemporaryDirectory(prefix="ratio_index_", dir="/tmp") as scratch:
                        path = Path(scratch) / f"{split}.npz"
                        np.savez_compressed(path, **data)
                        builder.publish(path, entry / "index" / path.name)
                for name in ("action_norm_stats.json", "tactile_scale.json", "splits.json", "sources.json"):
                    builder.publish(root / "clean/meta" / name, entry / "meta" / name)
                manifest = builder.read(root / "clean/meta/manifest.json")
                for split in ("train", "val"):
                    n = manifest["splits"][split]["count"]
                    manifest["splits"][split].update(clean_count=n, augmented_count=n * ratio, count=n * (1 + ratio))
                manifest.update(
                    clean_augmented_ratio=f"1:{ratio}",
                    storage="shared_clean_and_augmented_HDF5_virtual_views",
                    nested_variant_slots=list(range(ratio)),
                    training_started=False,
                )
                builder.write(entry / "meta/manifest.json", manifest)
                builder.write(
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
            report = {
                "status": "complete",
                "clean_pairs": clean_summary["pairs"],
                "augmented_pairs": aug_summary["pairs"],
                "clean_h5_bytes": clean_summary["bytes"],
                "augmented_h5_bytes": aug_summary["bytes"],
                "ratios": counts,
                "all_pairs_contact_supported": True,
                "ten_unique_variants_per_origin": True,
                "training_started": False,
                "finalizer_elapsed_seconds": time.monotonic() - started,
                "completed_at_utc": datetime.now(UTC).isoformat(),
                "validation": "all fields and all rows of every ratio view; production hyperslabs equal locally validated mappings",
                "finalizer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            }
            builder.write(root / "summary.json", report)
            builder.write(root / "progress.json", report)
            print(json.dumps(report), flush=True)
        except Exception as error:
            previous = builder.read(root / "progress.json")
            previous.update(status="failed", error=repr(error), updated_at_utc=datetime.now(UTC).isoformat())
            builder.write(root / "progress.json", previous)
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    run(args)
