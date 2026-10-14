"""Report candidate counts, combination coverage and measured disk size."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np
import pyarrow.parquet as pq

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
DEFAULT_ROOT = Path(
    "/path/to/datasets/touchguidence/augmentation/poker_fingertip_recovery_v1/kinematics/pilot_v2_mixed"
)


def sizes(root):
    files = sorted(p for p in root.rglob("*") if p.is_file())
    return {
        str(p.relative_to(root)): {"file_bytes": p.stat().st_size, "allocated_bytes": p.stat().st_blocks * 512}
        for p in files
    }


def summarize(root):
    root = root.resolve()
    config = json.loads((root / "config.json").read_text())
    with h5py.File(root / "perturbations.h5", "r") as f:
        success = f["attempts/success"][:]
        subset = f["attempts/subset_code"][:]
        masks = f["attempts/perturbed_fingers"][:]
        count = masks.sum(axis=1)
        kind = f["attempts/pattern_kind"].asstr()[:]
        anchor_indices = f["attempts/anchor_index"][:]
        anchors = len(f["anchors/state"])
        splits = f["anchors/split"].asstr()[:][anchor_indices]
        numeric_bytes = 0

        def accumulate(_name, obj):
            nonlocal numeric_bytes
            if isinstance(obj, h5py.Dataset) and obj.dtype.kind not in "OUS":
                numeric_bytes += obj.size * obj.dtype.itemsize

        f.visititems(accumulate)
        combos = []
        for code in range(1, 32):
            selected = subset == code
            n = int(selected.sum())
            good = int(np.sum(success & selected))
            fingers = [name for i, name in enumerate(FINGERS) if code & (1 << i)]
            combos.append(
                {
                    "subset_code": code,
                    "fingers": "+".join(fingers),
                    "finger_count": len(fingers),
                    "attempts": n,
                    "valid": good,
                    "rejected": n - good,
                    "valid_ratio": good / n if n else None,
                    "uniform_attempts": int(np.sum(selected & (kind == "uniform"))),
                    "mixed_attempts": int(np.sum(selected & (kind == "mixed"))),
                }
            )
        rows = []
        for k in range(1, 6):
            selected = count == k
            n = int(selected.sum())
            good = int(np.sum(success & selected))
            rows.append(
                {
                    "finger_count": k,
                    "subset_count": sum(c["finger_count"] == k for c in combos),
                    "attempts": n,
                    "valid": good,
                    "rejected": n - good,
                    "valid_ratio": good / n,
                }
            )
        pattern_counts = {
            name: {"attempts": int(np.sum(kind == name)), "valid": int(np.sum(success & (kind == name)))}
            for name in ("uniform", "mixed")
        }
        split_counts = {
            name: {"attempts": int(np.sum(splits == name)), "valid": int(np.sum(success & (splits == name)))}
            for name in ("train", "val")
        }
    with (root / "combination_summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(combos[0]))
        writer.writeheader()
        writer.writerows(combos)
    with (root / "finger_count_summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    source_count = pq.read_metadata(Path(config["anchor_root"]) / "index/contact_anchors.parquet").num_rows
    report = {
        "selected_anchors": anchors,
        "attempts": len(success),
        "valid": int(success.sum()),
        "rejected": int((~success).sum()),
        "valid_ratio": float(success.mean()),
        "attempts_per_anchor": len(success) / anchors,
        "subset_count": len(np.unique(subset)),
        "patterns": pattern_counts,
        "splits": split_counts,
        "by_finger_count": rows,
        "by_subset": combos,
        "uncompressed_numeric_payload_bytes": numeric_bytes,
        "source_contact_anchors": source_count,
        "all_source_anchors_attempts_projection": int(source_count * len(success) / anchors),
        "source_images_videos_and_tactile_copied": False,
        "new_tactile_or_recovery_actions_generated": False,
        "output_dir": str(root),
    }
    previous = root.parent / "pilot_v1"
    if (previous / "summary.json").exists():
        old = json.loads((previous / "summary.json").read_text())
        old_sizes = sizes(previous)
        report["previous_pilot"] = {
            "selected_anchors": old["selected_anchors"],
            "attempts": old["attempts"],
            "valid": old["valid"],
            "file_bytes": sum(v["file_bytes"] for v in old_sizes.values()),
        }
    target = root / "size_report.json"
    # Include this report itself in the measured file size (settles in <=3 passes).
    for _ in range(5):
        details = sizes(root)
        report["files"] = details
        report["total_file_bytes"] = sum(v["file_bytes"] for v in details.values())
        report["total_allocated_bytes"] = sum(v["allocated_bytes"] for v in details.values())
        report["total_file_mib"] = round(report["total_file_bytes"] / 1024**2, 6)
        report["total_file_mb"] = round(report["total_file_bytes"] / 1e6, 6)
        payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if target.exists() and target.read_text() == payload:
            break
        target.write_text(payload)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "selected_anchors",
                    "attempts",
                    "valid",
                    "rejected",
                    "total_file_mb",
                    "total_file_mib",
                    "by_finger_count",
                    "patterns",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    summarize(parser.parse_args().data_root)
