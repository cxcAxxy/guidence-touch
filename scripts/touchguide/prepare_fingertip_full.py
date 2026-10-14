"""Generate all contact anchors in resumable, independently validated shards."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait
from datetime import UTC
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
BASE = Path("/path/to/datasets/touchguidence/augmentation/poker_fingertip_recovery_v1")
FILES = (
    "scripts/touchguide/prepare_fingertip_perturbations.py",
    "scripts/touchguide/prepare_fingertip_recovery.py",
    "scripts/touchguide/validate_fingertip_recovery.py",
    "scripts/touchguide/analyze_fingertip_recovery.py",
    "src/openpi/touchguide/fingertip_kinematics.py",
    "src/openpi/touchguide/fingertip_sampling.py",
    "src/openpi/touchguide/fingertip_recovery.py",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, content):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def timestamp():
    return datetime.now(UTC).isoformat()


def completed(shard):
    return (shard / "complete.json").is_file()


def stage(args, shard, label, script, parameters):
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    command = [sys.executable, "-u", str(args.output_dir / "code" / "scripts/touchguide" / script), *parameters]
    with (shard / f"{label}.log").open("a") as log:
        log.write(json.dumps(command) + "\n")
        log.flush()
        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


def quarantine(path):
    if path.exists():
        target = path.with_name(path.name + ".incomplete_" + str(time.time_ns()))
        path.rename(target)


def process_shard(args, number):
    shard = args.output_dir / "shards" / f"shard_{number:04d}"
    shard.mkdir(parents=True, exist_ok=True)
    selected = args.anchor_table.slice(number * args.shard_size, args.shard_size)
    destination = shard / "selected_anchors.parquet"
    if destination.exists():
        if not pq.read_table(destination).equals(selected):
            raise ValueError(f"Shard anchor identity changed: {number}")
    else:
        pq.write_table(selected, destination, compression="zstd")
    if completed(shard):
        return read(shard / "complete.json")
    kin, recovery = shard / "kinematics", shard / "recovery"
    # An incomplete stage has no trusted checkpoint. Preserve it for inspection
    # and rebuild this small shard; completed stages are reused on resume.
    if not (kin / "summary.json").is_file():
        quarantine(kin)
        stage(
            args,
            shard,
            "kinematics",
            "prepare_fingertip_perturbations.py",
            [
                "--anchor-root",
                str(args.anchor_root),
                "--selected-anchors",
                str(shard / "selected_anchors.parquet"),
                "--output-dir",
                str(kin),
                "--sim-root",
                str(args.sim_root),
                "--combination-mode",
                "all_subsets_mixed",
                "--mixed-patterns-per-subset",
                "6",
                "--magnitudes-mm",
                "0.5",
                "1.0",
                "2.0",
                "--max-joint-delta-rad",
                "0.15",
                "--position-tolerance-mm",
                "0.05",
                "--seed",
                str(args.seed + number),
            ],
        )
    if not (recovery / "summary.json").is_file():
        quarantine(recovery)
        stage(
            args,
            shard,
            "recovery",
            "prepare_fingertip_recovery.py",
            [
                "--kinematic-root",
                str(kin),
                "--output-dir",
                str(recovery),
                "--release-distance-mm",
                "1.0",
                "--max-force-gain",
                "3.0",
                "--max-rotation-deg",
                "15.0",
            ],
        )
    if not (recovery / "validation.json").is_file():
        stage(args, shard, "validation", "validate_fingertip_recovery.py", ["--data-root", str(recovery)])
    validation = read(recovery / "validation.json")
    if validation.get("passed") is not True:
        raise ValueError(f"Shard {number} did not pass validation")
    summary, kin_summary = read(recovery / "summary.json"), read(kin / "summary.json")
    selected = pq.read_table(shard / "selected_anchors.parquet", columns=["anchor_uid"])
    if summary["clean_anchors"] != len(selected) or validation["clean_pairs_checked"] != len(selected):
        raise ValueError(f"Missing clean anchor in shard {number}")
    if summary["augmented_pairs"] != kin_summary["valid"] or validation["pairs_checked"] != kin_summary["valid"]:
        raise ValueError(f"Missing recovery pair in shard {number}")
    import h5py  # noqa: PLC0415

    with h5py.File(recovery / "recovery.h5", "r") as data:
        if data["anchors/anchor_uid"].asstr()[:].tolist() != selected["anchor_uid"].to_pylist():
            raise ValueError(f"Wrong source identity in shard {number}")
    result = {
        "shard": number,
        "relative_path": str(shard.relative_to(args.output_dir)),
        "recovery_h5": str((recovery / "recovery.h5").relative_to(args.output_dir)),
        "summary": summary,
        "validation": validation,
        "kinematic_attempts": kin_summary["attempts"],
        "file_bytes": sum(p.stat().st_size for p in shard.rglob("*") if p.is_file()),
        "completed_at_utc": timestamp(),
    }
    write(shard / "complete.json", result)
    return result


def prepare(args):
    source = args.anchor_root / "index/contact_anchors.parquet"
    table = pq.read_table(source)
    rows = table.select(["anchor_uid", "episode_index", "split", "contact_fingers"]).to_pylist()
    if (
        not rows
        or len({r["anchor_uid"] for r in rows}) != len(rows)
        or not all(any(r["contact_fingers"]) for r in rows)
    ):
        raise ValueError("Contact anchors must be nonempty, unique and have recorded contact")
    split_episodes = {s: {r["episode_index"] for r in rows if r["split"] == s} for s in ("train", "val")}
    if split_episodes["train"] & split_episodes["val"]:
        raise ValueError("Source episode leakage")
    config = {
        "schema": "poker_fingertip_full_sharded_v1",
        "anchor_root": str(args.anchor_root),
        "source_index_sha256": digest(source),
        "selection_config_sha256": digest(args.anchor_root / "meta/selection_config.json"),
        "selected_anchors": len(table),
        "anchor_splits": dict(Counter(r["split"] for r in rows)),
        "shard_size": args.shard_size,
        "shards": (len(table) + args.shard_size - 1) // args.shard_size,
        "seed": args.seed,
        "sim_root": str(args.sim_root),
        "all_contact_anchors_selected": True,
        "magnitudes_mm": [0.5, 1.0, 2.0],
        "mixed_patterns_per_subset": 6,
        "expected_kinematic_attempts": len(table) * 1026,
        "training_sampling_ratio": None,
        "training_started": False,
        "original_image_frames_referenced": True,
        "synthetic_contact_is_physics_ground_truth": False,
    }
    path = args.output_dir / "config.json"
    if path.exists():
        previous = read(path)
        if any(previous.get(k) != v for k, v in config.items()):
            raise ValueError("Resume configuration/source mismatch; choose a new destination")
        for name, expected in previous["frozen_code_sha256"].items():
            if digest(args.output_dir / "code" / name) != expected:
                raise ValueError(f"Frozen implementation changed: {name}")
    else:
        for name in FILES:
            destination = args.output_dir / "code" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO / name, destination)
        config["frozen_code_sha256"] = {name: digest(args.output_dir / "code" / name) for name in FILES}
        write(path, config)
        shutil.copy2(source, args.output_dir / "contact_anchors.parquet")
        (args.output_dir / "README.md").write_text(
            "# All-contact fingertip recovery augmentation\n\n"
            "All selected contact origins retain their original episode train/val split.\n"
            "Each shard contains kinematic candidates, noisy state, synthetic tactile, 30-step recovery chunks, "
            "clean reference pairs and original image/video references.\n"
            "`manifest.json` lists validated shards; `progress.json` includes live status and failures. "
            "Only status=complete indicates the entire contact pool is ready.\n"
            "Use recovery/index/train_supported.npz and val_supported.npz for contact-supported variants. "
            "Unsupported variants are retained for inspection, not labeled as task failures.\n"
            "Training clean/augmented sampling ratio is undecided. Original videos are referenced, not copied.\n"
            "Force synthesis is an approximate 1mm-release surrogate, not physical recovery validation.\n"
        )
    # Prepare each small index in its worker, so NAS metadata latency does not
    # delay all computation until hundreds of directories have been created.
    args.anchor_table = table
    return config


def progress(args, config, results, failures, running, started, status):
    summaries = [v["summary"] for v in results.values()]
    content = {
        "status": status,
        "updated_at_utc": timestamp(),
        "workers": args.workers,
        "total_anchors": config["selected_anchors"],
        "total_shards": config["shards"],
        "completed_shards": len(results),
        "completed_anchors": sum(s["clean_anchors"] for s in summaries),
        "kinematic_attempts": sum(v["kinematic_attempts"] for v in results.values()),
        "augmented_pairs": sum(s["augmented_pairs"] for s in summaries),
        "supported_pairs": sum(s["supported_pairs"] for s in summaries),
        "unsupported_pairs": sum(s["unsupported_pairs"] for s in summaries),
        "splits": {
            split: {
                key: sum(s["splits"][split][key] for s in summaries)
                for key in ("clean_anchors", "pairs", "supported_pairs")
            }
            for split in ("train", "val")
        },
        "completed_file_bytes": sum(v["file_bytes"] for v in results.values()),
        "running_shards": sorted(running),
        "failed_shards": failures,
        "elapsed_this_launch_s": time.monotonic() - started,
        "training_sampling_ratio": None,
    }
    write(args.output_dir / "progress.json", content)
    write(
        args.output_dir / "manifest.json",
        {
            "schema": config["schema"],
            "status": status,
            "shards": [
                {k: v[k] for k in ("shard", "relative_path", "recovery_h5")} for _, v in sorted(results.items())
            ],
        },
    )
    return content


def run(args):
    args.output_dir = args.output_dir.resolve()
    args.anchor_root, args.sim_root = args.anchor_root.resolve(), args.sim_root.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "run.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = prepare(args)
        started, results, failures = time.monotonic(), {}, {}
        manifest_path = args.output_dir / "manifest.json"
        if manifest_path.is_file():
            for saved in read(manifest_path)["shards"]:
                number = saved["shard"]
                shard = args.output_dir / saved["relative_path"]
                results[number] = read(shard / "complete.json")
        pending = iter(n for n in range(config["shards"]) if n not in results)
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            jobs = {}

            def submit():
                number = next(pending, None)
                if number is not None:
                    jobs[executor.submit(process_shard, args, number)] = number

            for _ in range(args.workers):
                submit()
            while jobs:
                progress(args, config, results, failures, jobs.values(), started, "running")
                done, _ = wait(jobs, timeout=15, return_when=FIRST_COMPLETED)
                for future in done:
                    number = jobs.pop(future)
                    try:
                        result = future.result()
                        results[number] = result
                        print(
                            f"Completed shard {number}: {result['summary']['augmented_pairs']} pairs; "
                            f"{len(results)}/{config['shards']} shards",
                            flush=True,
                        )
                    except Exception as error:
                        failures[str(number)] = repr(error)
                        print(f"FAILED shard {number}: {error!r}; see its stage logs", flush=True)
                    submit()
        status = "failed" if failures or len(results) != config["shards"] else "complete"
        report = progress(args, config, results, failures, [], started, status)
        if status == "complete":
            assert report["completed_anchors"] == config["selected_anchors"]
            assert report["kinematic_attempts"] == config["expected_kinematic_attempts"]
            report["all_shards_validated"] = True
            write(args.output_dir / "summary.json", report)
        print(json.dumps(report, ensure_ascii=False), flush=True)
        if status != "complete":
            raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-root", type=Path, default=BASE)
    parser.add_argument("--output-dir", type=Path, default=BASE / "full_v1_30steps")
    parser.add_argument("--sim-root", type=Path, default=Path("/path/to/kaitactilesim"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--shard-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.workers < 1 or args.shard_size < 1:
        parser.error("workers and shard-size must be positive")
    run(args)
