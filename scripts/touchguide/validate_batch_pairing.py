#!/usr/bin/env python3
"""Validate TouchGuide Step 2 identity batching using real clean samples.

This script performs CPU data checks only. It does not create negative action
chunks, add noise, instantiate a model, or train. The sole output artifact is
an explicit --report JSON file; source datasets and source indices are read-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from openpi.touchguide.batching import CPMBatchLoader, collate_matched_samples, iter_index_batches
from openpi.touchguide.data import CPMDataset

DEFAULT_ROOT = Path(
    "/path/to/datasets/sim/poker-draw/pi05_trex/touchguide/"
    "cpm_datasets/poker_0920_200_touchguide_v1"
)
TENSOR_SHAPES = {
    "head": (3, 224, 224), "wrist": (3, 224, 224),
    "tactile_map_raw": (5, 3, 7, 5), "actions": (30, 27),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def expect_rejection(operation: Callable, name: str) -> dict:
    try:
        operation()
    except (ValueError, TypeError, RuntimeError) as error:
        return {"rejected": True, "exception": type(error).__name__, "message": str(error)}
    raise AssertionError(f"{name} must reject invalid input")


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def verify_index_iterator() -> dict:
    source = [11, 4, 8, 2, 5, 9]
    full = list(iter_index_batches(source, 4, drop_last=False))
    dropped = list(iter_index_batches(source, 4, drop_last=True))
    require(full == [[11, 4, 8, 2], [5, 9]], "Index iterator reordered or lost an eligible tail")
    require(dropped == [[11, 4, 8, 2]], "drop_last did not discard only the incomplete tail")
    require(all(isinstance(batch, list) and all(type(i) is int for i in batch) for batch in full),
            "Index iterator must yield lists of Python integers")
    require(list(iter_index_batches([], 4, drop_last=False)) == [], "Empty index input should yield no batches")
    rejected = {
        "batch_size_one": expect_rejection(lambda: list(iter_index_batches(source, 1)), "batch_size=1"),
        "batch_size_zero": expect_rejection(lambda: list(iter_index_batches(source, 0)), "batch_size=0"),
        "singleton_tail": expect_rejection(lambda: list(iter_index_batches([0, 1, 2, 3, 4], 4)),
                                           "singleton final batch"),
    }
    require(list(iter_index_batches([0, 1, 2, 3, 4], 4, drop_last=True)) == [[0, 1, 2, 3]],
            "drop_last=True should handle a singleton tail by discarding it")
    return {"full_batches": full, "drop_last_batches": dropped, "rejections": rejected}


def verify_fixed_validation(path: Path, expected_length: int) -> tuple[dict, list[int]]:
    with np.load(path, allow_pickle=False) as source:
        indices = source["indices"].copy()
        offsets = source["batch_offsets"].copy()
        batch_size = source["batch_size"].copy()
        seed = source["seed"].copy()
    require(expected_length == 4372, "Expected 4,372 validation samples")
    require(indices.dtype == np.int64 and indices.shape == (4372,), "Invalid validation indices schema")
    require(offsets.dtype == np.int64 and offsets.shape == (70,), "Invalid validation offsets schema")
    require(batch_size.dtype == np.int64 and batch_size.shape == () and batch_size.item() == 64,
            "batch_size must be an int64 scalar equal to 64")
    require(seed.dtype == np.int64 and seed.shape == () and seed.item() == 42,
            "seed must be an int64 scalar equal to 42")
    require(np.array_equal(np.sort(indices), np.arange(expected_length)),
            "Validation indices have duplicates, omissions, or out-of-range values")
    regenerated = np.random.default_rng(int(seed)).permutation(expected_length).astype(np.int64)
    require(np.array_equal(indices, regenerated), "Validation permutation does not reproduce seed 42")
    expected_offsets = np.concatenate((np.arange(0, expected_length, 64, dtype=np.int64),
                                       np.asarray([expected_length], dtype=np.int64)))
    require(np.array_equal(offsets, expected_offsets), "Validation offsets do not partition all samples")
    lengths = np.diff(offsets)
    require(np.array_equal(lengths, np.asarray([64] * 68 + [20], dtype=np.int64)),
            "Expected 68 complete batches and one final batch of 20")
    iterator_batches = list(iter_index_batches(indices.tolist(), 64, drop_last=False))
    fixed_batches = [indices[start:stop].tolist() for start, stop in zip(offsets[:-1], offsets[1:])]
    require(iterator_batches == fixed_batches, "Index iterator disagrees with fixed validation offsets")
    dropped_batches = list(iter_index_batches(indices.tolist(), 64, drop_last=True))
    require(dropped_batches == fixed_batches[:-1], "Validation drop_last partition is wrong")
    return {
        "path": str(path.resolve()), "sha256": digest(path), "sample_count": expected_length,
        "seed": int(seed), "batch_size": int(batch_size), "batch_count": len(fixed_batches),
        "complete_batch_count": 68, "tail_batch_size": 20,
        "permutation_has_no_duplicates_or_omissions": True, "seed_reproduces_exactly": True,
        "offsets_match_index_iterator": True, "first_batch_indices": fixed_batches[0],
    }, indices[:3].tolist()


def verify_real_batch(dataset: CPMDataset, indices: list[int]) -> tuple[dict, list[dict]]:
    require(len(indices) == 3 and len(set(indices)) == 3, "Real sample check requires three distinct indices")
    # The loader decodes its own samples. Read again independently to catch any
    # sample reordering, tensor changes, or action re-expression by the collator.
    batches = list(CPMBatchLoader(dataset, [indices], include_auxiliary=False))
    require(len(batches) == 1, "One index batch must yield exactly one collated batch")
    batch = batches[0]
    samples = [dataset[index] for index in indices]
    for sample in samples:
        require("negative_actions" not in sample, "The positive reader returned legacy negative chunks")
    for key, shape in TENSOR_SHAPES.items():
        tensor = batch[key]
        require(isinstance(tensor, torch.Tensor), f"{key} must be a Torch tensor")
        require(tuple(tensor.shape) == (3, *shape), f"Wrong batched shape for {key}: {tuple(tensor.shape)}")
        require(tensor.dtype == torch.float32, f"Unexpected dtype for {key}: {tensor.dtype}")
        require(bool(torch.isfinite(tensor).all()), f"Nonfinite values in {key}")
        for position, sample in enumerate(samples):
            require(torch.equal(tensor[position], sample[key]),
                    f"{dataset.split} {key}[{position}] differs from the individual reader")
        if key in ("head", "wrist"):
            require(bool((tensor >= -1).all() and (tensor <= 1).all()), f"{key} is outside [-1,1]")
    require("negative_actions" not in batch and "noisy_actions" not in batch,
            "Clean identity batching returned transformed action candidates")
    require("state_raw" not in batch and "contact_fingers" not in batch,
            "include_auxiliary=False leaked auxiliary features into the top-level batch")
    targets = batch["targets"]
    require(isinstance(targets, torch.Tensor) and targets.dtype == torch.int64,
            "Identity targets must be an int64 Torch tensor")
    require(torch.equal(targets.cpu(), torch.arange(3)), "Targets are not the diagonal identities")
    ids = batch["sample_ids"]
    require(isinstance(ids, list) and ids == [sample["id"] for sample in samples],
            "sample_ids must preserve the reader and index order")
    require(len(set(ids)) == 3, "Batch contains duplicate sample IDs")
    metadata = batch["metadata"]
    require(isinstance(metadata, list) and len(metadata) == 3 and all(isinstance(item, dict) for item in metadata),
            "Metadata must be a list of three dictionaries")
    require(metadata == [sample["metadata"] for sample in samples], "Metadata does not preserve sample provenance")
    expected_positive = torch.eye(3, dtype=torch.bool)
    for key, expected in (("positive_mask", expected_positive), ("negative_mask", ~expected_positive)):
        mask = batch[key]
        require(isinstance(mask, torch.Tensor) and mask.dtype == torch.bool and tuple(mask.shape) == (3, 3),
                f"Invalid {key} schema")
        require(torch.equal(mask.cpu(), expected), f"{key} does not represent identity matching")
    require(int(batch["positive_mask"].sum()) == 3 and int(batch["negative_mask"].sum()) == 6,
            "Wrong positive/mismatched pair counts")
    return {
        "split": dataset.split, "sample_indices": indices, "sample_ids": ids,
        "tensor_shapes": {key: list(batch[key].shape) for key in TENSOR_SHAPES},
        "individual_reader_tensors_match_exactly": True, "metadata_matches_exactly": True,
        "targets": targets.tolist(), "positive_mask": batch["positive_mask"].tolist(),
        "negative_mask": batch["negative_mask"].tolist(),
        "positive_identity_pairs": 3, "mismatched_identity_pairs": 6,
        "mismatched_pairs_are_not_physical_failure_labels": True,
    }, samples


def verify_collator_rejections(samples: list[dict]) -> dict:
    forbidden = dict(samples[0])
    # This deliberately malformed input is only a guard test. No data-derived
    # negatives are produced, and it must be rejected before any collation.
    forbidden["negative_actions"] = torch.empty((1, 30, 27))
    return {
        "duplicate_sample_ids": expect_rejection(lambda: collate_matched_samples([samples[0], samples[0]]),
                                                 "duplicate sample IDs"),
        "legacy_negative_actions": expect_rejection(lambda: collate_matched_samples([forbidden, samples[1]]),
                                                    "legacy negative_actions in a matched sample"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--val-batches", type=Path)
    parser.add_argument("--legacy-root", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    manifest_path = root / "meta/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    val_batches_path = args.val_batches or root / "index/val_batches.npz"
    legacy_root = args.legacy_root or Path(manifest["parent"]["dataset_root"])
    train = val = legacy = None
    try:
        train = CPMDataset(root, "train", include_images=True, include_negatives=False)
        val = CPMDataset(root, "val", include_images=True, include_negatives=False)
        require(len(train) == 39333, "Expected 39,333 training samples")
        iterator_report = verify_index_iterator()
        fixed_report, val_indices = verify_fixed_validation(val_batches_path, len(val))
        train_indices = [0, len(train) // 2, len(train) - 1]
        train_report, samples = verify_real_batch(train, train_indices)
        val_report, _ = verify_real_batch(val, val_indices)
        rejection_report = verify_collator_rejections(samples)
        legacy = CPMDataset(legacy_root, "train", include_images=False, include_negatives=True)
        rejection_report["legacy_dataset_include_negatives"] = expect_rejection(
            lambda: CPMBatchLoader(legacy, [[0, 1]], include_auxiliary=False),
            "CPMBatchLoader construction with a legacy include_negatives=True dataset",
        )
        report = {
            "status": "passed", "scope": "step2_clean_identity_batching",
            "dataset_root": str(root), "manifest_sha256": digest(manifest_path),
            "sample_counts": {"train": len(train), "val": len(val)},
            "fixed_validation": fixed_report, "index_iterator": iterator_report,
            "real_batches": [train_report, val_report], "guard_rejections": rejection_report,
            "noise_generated": False, "model_loaded": False, "training_run": False,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"status": "passed", "report": str(args.report.resolve()),
                          "real_batches": 2, "fixed_validation_batches": 69}, indent=2))
    finally:
        for dataset in (train, val, legacy):
            if dataset is not None:
                dataset.close()


if __name__ == "__main__":
    main()
