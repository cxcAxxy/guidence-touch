"""Audit full recovery/clean numerical ranges without training or modifying data."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
import json
import math
from pathlib import Path

import h5py
import numpy as np

ROOT = Path("/path/to/datasets/touchguidence/augmentation/poker_fingertip_recovery_rules_v1_random10")
PERCENTILES = [0, 1, 50, 95, 99, 99.9, 100]


def summarize(values):
    x = np.asarray(values, dtype=np.float64).reshape(-1, 27)
    return {
        "count_per_dimension": len(x),
        "min": x.min(0),
        "max": x.max(0),
        "sum": x.sum(0),
        "sum_sq": np.square(x).sum(0),
        "outside_unit": (np.abs(x) > 1).sum(0),
        "outside_three": (np.abs(x) > 3).sum(0),
        "outside_five": (np.abs(x) > 5).sum(0),
    }


def episode(task):
    root, eid, split, stats, rms, samples = task
    rng = np.random.default_rng(np.random.SeedSequence([42, eid, 123]))
    low, high = (np.asarray(stats["actions"][key], dtype=np.float32) for key in ("q01", "q99"))
    state_low, state_high = (np.asarray(stats["state"][key], dtype=np.float32) for key in ("q01", "q99"))
    result = {"episode": eid, "split": split, "data": {}}
    sampled = {}
    selected = None
    clean_all_state = None
    for kind in ("clean", "augmented"):
        with h5py.File(root / kind / "episodes" / f"episode_{eid:06d}.h5", "r") as f:
            state, absolute, normalized, touch = [
                f[name][:]
                for name in (
                    "state_raw",
                    "actions_absolute",
                    "actions",
                    "tactile_map_raw",
                )
            ]
            assert all(np.isfinite(x).all() for x in (state, absolute, normalized, touch))
            if kind == "clean":
                selected = np.sort(rng.choice(len(state), min(samples, len(state)), replace=False))
                ids = selected
                clean_state = state[selected]
            else:
                assert len(state) == result["data"]["clean"]["pairs"] * 10
                assert np.array_equal(f["source_clean_row"][:], np.repeat(np.arange(len(state) // 10), 10))
                ids = (selected[:, None] * 10 + np.arange(10)).ravel()
                assert np.array_equal(absolute[:, -1], np.repeat(clean_all_state, 10, axis=0))
            physical = absolute.copy()
            physical[:, :, :7] -= state[:, None, :7]
            expected = (physical - low) / (high - low + 1e-6) * 2 - 1
            error = float(np.max(np.abs(expected - normalized)))
            assert error < 1e-5
            if kind == "augmented":
                assert np.all(physical[:, :, :7] == 0)
            displacement = absolute - state[:, None, :]
            t = touch.transpose(0, 1, 3, 4, 2).reshape(-1, 3).astype(np.float64)
            ratio = t / np.asarray(rms)
            pixel = 0.5 + np.arcsinh(ratio) / (2 * math.asinh(32))
            result["data"][kind] = {
                "pairs": len(state),
                "actions": summarize(normalized),
                "state": summarize((state - state_low) / (state_high - state_low + 1e-6) * 2 - 1),
                "max_normalization_error": error,
                "motion_sum_sq": np.square(displacement.astype(np.float64)).sum(axis=(0, 1)),
                "motion_max_abs": np.abs(displacement).max(axis=(0, 1)),
                "touch_count_per_channel": len(t),
                "touch_sum_sq": np.square(t).sum(0),
                "touch_min": t.min(0),
                "touch_max": t.max(0),
                "touch_nonzero": (t != 0).sum(0),
                "pixel_min": pixel.min(0),
                "pixel_max": pixel.max(0),
                "pixel_outside_unit": ((pixel < 0) | (pixel > 1)).sum(0),
                "pixel_sum": pixel.sum(0),
                "pixel_sum_sq": np.square(pixel).sum(0),
            }
            sampled[kind] = {
                "actions": normalized[ids],
                "physical": physical[ids],
                "motion": displacement[ids],
                "touch": touch[ids],
                "state": state[ids],
            }
            if kind == "clean":
                clean_all_state = state
    sampled["clean_state"] = clean_state
    return result, sampled


def aggregate(records, kind, split):
    selected = [r["data"][kind] for r in records if split == "all" or r["split"] == split]
    out = {"pairs": sum(v["pairs"] for v in selected)}
    for field in ("actions", "state"):
        n = sum(v[field]["count_per_dimension"] for v in selected)
        mean = sum(v[field]["sum"] for v in selected) / n
        out[field] = {
            "count_per_dimension": n,
            "min": np.min([v[field]["min"] for v in selected], 0),
            "max": np.max([v[field]["max"] for v in selected], 0),
            "mean": mean,
            "std": np.sqrt(np.maximum(0, sum(v[field]["sum_sq"] for v in selected) / n - mean**2)),
            **{
                f"fraction_abs_gt_{limit}": sum(v[field][key] for v in selected) / n
                for limit, key in [
                    (1, "outside_unit"),
                    (3, "outside_three"),
                    (5, "outside_five"),
                ]
            },
        }
    n = sum(v["touch_count_per_channel"] for v in selected)
    out["touch"] = {
        "count_per_channel": n,
        "rms_n": np.sqrt(sum(v["touch_sum_sq"] for v in selected) / n),
        "min_n": np.min([v["touch_min"] for v in selected], 0),
        "max_n": np.max([v["touch_max"] for v in selected], 0),
        "nonzero_fraction": sum(v["touch_nonzero"] for v in selected) / n,
        "mapped_pixel_min": np.min([v["pixel_min"] for v in selected], 0),
        "mapped_pixel_max": np.max([v["pixel_max"] for v in selected], 0),
        "mapped_pixel_outside_0_1_fraction": sum(v["pixel_outside_unit"] for v in selected) / n,
    }
    out["motion_rms_rad"] = np.sqrt(sum(v["motion_sum_sq"] for v in selected) / out["actions"]["count_per_dimension"])
    out["motion_max_abs_rad"] = np.max([v["motion_max_abs"] for v in selected], 0)
    out["max_normalization_error"] = max(v["max_normalization_error"] for v in selected)
    return out


def quantiles(x):
    x = np.asarray(x)
    return {"percentiles": PERCENTILES, "values": np.percentile(x, PERCENTILES, axis=0)}


def encode_json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value))


def run(args):
    root = args.data_root.resolve()
    output = args.output_dir or root / "validation/asinh_action_audit"
    output.mkdir(parents=True, exist_ok=True)
    stats = json.loads((root / "clean/meta/action_norm_stats.json").read_text())["norm_stats"]
    rms = json.loads((root / "clean/meta/tactile_scale.json").read_text())["rms"]
    manifest = json.loads((root / "clean/meta/manifest.json").read_text())
    tasks = [
        (root, eid, split, stats, rms, args.samples_per_episode)
        for split in ("train", "val")
        for eid in manifest["splits"][split]["episode_indices"]
    ]
    records, samples = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(episode, t) for t in tasks]):
            record, sample = future.result()
            records.append(record)
            samples.append(sample)
            if len(records) % 10 == 0:
                print(f"Audited {len(records)}/{len(tasks)} episodes", flush=True)
    report = {
        "status": "passed",
        "data_root": str(root),
        "episodes": len(records),
        "scope": "full exact counts/ranges/moments; quantiles and paired changes sampled; no execution validation",
        "current_tactile_rms_n": rms,
        "reference_multiple": 32,
        "exact": {
            kind: {split: aggregate(records, kind, split) for split in ("train", "val", "all")}
            for kind in ("clean", "augmented")
        },
        "sampled": {},
    }
    collected = {}
    for kind in ("clean", "augmented"):
        collected[kind] = {
            key: np.concatenate([s[kind][key] for s in samples])
            for key in ("actions", "physical", "motion", "touch", "state")
        }
        d = collected[kind]
        report["sampled"][kind] = {
            "pairs": len(d["state"]),
            "normalized_action_per_dimension": quantiles(d["actions"].reshape(-1, 27)),
            "physical_action_per_dimension": quantiles(d["physical"].reshape(-1, 27)),
            "terminal_motion_per_dimension": quantiles(d["motion"][:, -1]),
            "touch_abs_nonzero_n": [],
            "asinh_reference_candidates": {},
        }
        touch = d["touch"].transpose(0, 1, 3, 4, 2).reshape(-1, 3)
        for c in range(3):
            values = np.abs(touch[:, c])
            report["sampled"][kind]["touch_abs_nonzero_n"].append(quantiles(values[values > 0]))
        for reference in (8, 16, 32, 64):
            pixel = 0.5 + np.arcsinh(touch / np.asarray(rms)) / (2 * math.asinh(reference))
            report["sampled"][kind]["asinh_reference_candidates"][str(reference)] = {
                "pixel_quantiles": quantiles(pixel),
                "outside_fraction": ((pixel < 0) | (pixel > 1)).mean(0),
            }
    clean, aug = collected["clean"], collected["augmented"]
    assert len(aug["state"]) == 10 * len(clean["state"])
    delta = aug["actions"] - np.repeat(clean["actions"], 10, axis=0)
    report["sampled"]["same_origin_action_difference"] = {
        "normalized_rms_per_dimension": np.sqrt(np.square(delta.astype(np.float64)).mean(axis=(0, 1))),
        "normalized_max_abs_per_dimension": np.abs(delta).max(axis=(0, 1)),
    }
    report["sampled"]["tactile_pair_change"] = {}
    original = np.repeat(clean["touch"], 10, axis=0)
    changed = np.abs(aug["touch"] - original).sum(axis=(2, 3, 4)) > 1e-8
    for reference in (8, 16, 32, 64):

        def mapped(x, multiple=reference):
            return 0.5 + np.arcsinh(x / np.asarray(rms)[None, None, :, None, None]) / (2 * math.asinh(multiple))

        difference = mapped(aug["touch"]) - mapped(original)
        per_finger_rms = np.sqrt(np.square(difference).mean(axis=(2, 3, 4)))
        per_finger_max = np.abs(difference).max(axis=(2, 3, 4))
        report["sampled"]["tactile_pair_change"][str(reference)] = {
            "changed_finger_pairs": int(changed.sum()),
            "changed_finger_pixel_rms": quantiles(per_finger_rms[changed]),
            "changed_finger_pixel_max_abs": quantiles(per_finger_max[changed]),
        }
    # A bounded, paired fixture for testing the real DINO preprocessor.
    fixture = []
    for variant in range(10):
        distances = np.sqrt(np.square(aug["touch"][variant::10] - clean["touch"]).mean(axis=(1, 2, 3, 4)))
        choices = np.argsort(distances)
        fixture.extend(
            (int(choices[rank]), variant) for rank in (len(choices) // 4, len(choices) // 2, 3 * len(choices) // 4)
        )
    np.savez_compressed(
        output / "paired_touch_fixture.npz",
        clean=np.stack([clean["touch"][i] for i, v in fixture]),
        augmented=np.stack([aug["touch"][10 * i + v] for i, v in fixture]),
        rms=np.asarray(rms),
    )
    (output / "report.json").write_text(json.dumps(report, default=encode_json, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": "passed",
                "report": str(output / "report.json"),
                "clean_pairs": report["exact"]["clean"]["all"]["pairs"],
                "augmented_pairs": report["exact"]["augmented"]["all"]["pairs"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--samples-per-episode", type=int, default=32)
    run(parser.parse_args())
