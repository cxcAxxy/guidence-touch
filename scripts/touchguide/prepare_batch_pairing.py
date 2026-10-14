"""Save fixed validation groups and pairing provenance for the poker CPM entry."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import h5py
import numpy as np

from openpi.touchguide.batching import EpochIndexBatchSampler, iter_index_batches


DEFAULT_ROOT = Path("/path/to/datasets/sim/poker-draw/pi05_trex/touchguide/"
                    "cpm_datasets/poker_0920_200_touchguide_v1")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def load_index(root: Path, split: str) -> dict[str, np.ndarray]:
    with np.load(root / f"index/{split}.npz", allow_pickle=False) as source:
        return {key: source[key].copy() for key in ("episode_index", "sample_row", "frame_index", "timestamp_s")}


def sample_reference(root: Path, split: str, index: dict[str, np.ndarray], position: int) -> dict:
    episode_id, row, frame = (int(index[key][position]) for key in ("episode_index", "sample_row", "frame_index"))
    relative = f"episodes/episode_{episode_id:06d}.h5"
    with h5py.File(root / relative, "r") as episode:
        if int(episode["frame_index"][row]) != frame:
            raise ValueError("Index and example episode disagree")
        cameras = {key: str(episode.attrs[key]) for key in ("head_video", "right_wrist_video")}
    return {"id": f"{split}:{position:08d}", "split_index": position, "episode_index": episode_id,
            "sample_row": row, "frame_index": frame, "timestamp_s": float(index["timestamp_s"][position]),
            "episode_file": relative, "camera_video_time_s": frame / 30.0, **cameras,
            "tactile_dataset": "tactile_map_raw", "action_dataset": "actions"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--example-size", type=int, default=3)
    args = parser.parse_args()
    root = args.dataset_root.expanduser().resolve(strict=True)
    if args.batch_size < 2 or args.seed < 0 or not 2 <= args.example_size <= args.batch_size:
        raise ValueError("Invalid batching/example settings")
    manifest = json.loads((root / "meta/manifest.json").read_text())
    if manifest["schema"] != "touchguide_cpm_positive_entry_v1":
        raise ValueError("Use the dedicated TouchGuide positive entry")
    if manifest["implementation_status"]["batch_pairing"]:
        raise FileExistsError("Batch pairing is already prepared; existing groups are not overwritten")
    new_files = ("index/val_batches.npz", "examples/batch_examples.json", "examples/pair_examples.jsonl",
                 "meta/batch_pairing.json")
    if any((root / name).exists() for name in new_files):
        raise FileExistsError("One or more pairing artifacts already exist")
    staging = root / ".pairing.building"
    if staging.exists():
        raise FileExistsError(staging)
    indices = {split: load_index(root, split) for split in ("train", "val")}
    lengths = {split: len(index["episode_index"]) for split, index in indices.items()}
    if any(lengths[split] != manifest["splits"][split]["count"] for split in lengths):
        raise ValueError("Split index counts differ from the manifest")
    train_sampler = EpochIndexBatchSampler(lengths["train"], args.batch_size, seed=args.seed, drop_last=True)
    validation_indices = np.random.default_rng(args.seed).permutation(lengths["val"]).astype(np.int64)
    validation_batches = list(iter_index_batches(validation_indices, args.batch_size, drop_last=False))
    validation_offsets = np.asarray([0, *np.cumsum([len(batch) for batch in validation_batches])], dtype=np.int64)
    batch_settings = {
        "train": {"loader": "openpi.touchguide.batching.CPMBatchLoader.from_config",
                  "sampler": "epoch_permutation", "batch_size": args.batch_size, "seed": args.seed,
                  "seed_sequence": "[seed, epoch]", "drop_last": True, "include_auxiliary": False},
        "val": {"loader": "openpi.touchguide.batching.CPMBatchLoader.from_config",
                "sampler": "fixed_index_file", "batch_size": args.batch_size, "seed": args.seed,
                "drop_last": False, "index_file": "index/val_batches.npz",
                "source_index_sha256": sha256(root / "index/val.npz"), "include_auxiliary": False},
    }
    first_indices = {"train": next(iter(train_sampler))[:args.example_size],
                     "val": validation_batches[0][:args.example_size]}
    examples, pair_records = [], []
    for split, chosen in first_indices.items():
        samples = [sample_reference(root, split, indices[split], position) for position in chosen]
        batch_id = f"{split}_example_000"
        examples.append({"batch_id": batch_id, "kind": "small_illustrative_batch",
                         "configured_batch_size": args.batch_size, "example_batch_size": len(samples),
                         "source": "first samples of epoch 0 training / fixed validation batch",
                         "sample_indices": chosen, "samples": samples,
                         "targets": list(range(len(samples))),
                         "positive_mask": np.eye(len(samples), dtype=bool).tolist()})
        for row, observation in enumerate(samples):
            for column, action in enumerate(samples):
                pair_records.append({"batch_id": batch_id, "observation_slot": row, "action_slot": column,
                                     "is_positive": row == column, "observation_id": observation["id"],
                                     "action_id": action["id"], "label_semantics": "same_sample_matching"})
    pairing = {
        "profile": "paper_batch_symmetric_infonce_v1", "batch_size": args.batch_size, "seed": args.seed,
        "positive_rule": "row_index == column_index", "negative_rule": "row_index != column_index",
        "targets": "arange(actual_batch_size), shared by row-wise and column-wise matching",
        "action_representation": "stored_normalized_action_in_own_anchor_reference",
        "recipient_action_rebasing": False, "offline_negative_pool_used": False,
        "negative_label_semantics": "mismatched pairing, not verified physical task failure",
        "train": {"samples": lengths["train"], "batches_per_epoch": len(train_sampler),
                  "samples_per_epoch": len(train_sampler) * args.batch_size,
                  "shuffled_tail_samples_dropped": lengths["train"] % args.batch_size},
        "val": {"samples": lengths["val"], "batches": len(validation_batches),
                "last_batch_size": len(validation_batches[-1]), "every_sample_used_once": True},
        "implemented": {"batch_assembly": True, "pairing_labels": True, "fixed_validation_batches": True,
                        "similarity_scores": False, "contrastive_loss": False,
                        "noise_augmentation": False, "cpm_training": False},
        "files": list(new_files),
    }
    staging.mkdir()
    try:
        (staging / "index").mkdir()
        np.savez_compressed(staging / "index/val_batches.npz", indices=validation_indices,
                            batch_offsets=validation_offsets, batch_size=np.int64(args.batch_size),
                            seed=np.int64(args.seed))
        write_json(staging / "examples/batch_examples.json", examples)
        (staging / "examples/pair_examples.jsonl").write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in pair_records))
        write_json(staging / "meta/batch_pairing.json", pairing)
        updated_files = []
        for split in ("train", "val"):
            config = json.loads((root / f"config/{split}.json").read_text())
            if config["reader_kwargs"]["include_negatives"] or config["noise_on_read"]:
                raise ValueError("Unexpected legacy negative/noise reader configuration")
            config["batching"] = batch_settings[split]
            write_json(staging / f"config/{split}.json", config)
            updated_files.append(f"config/{split}.json")
        manifest["stage"] = "batch_pairing_ready"
        manifest["implementation_status"]["batch_pairing"] = True
        manifest["planned_contrastive_training"]["pairing_implemented"] = True
        manifest["batching"] = pairing
        write_json(staging / "meta/manifest.json", manifest)
        entry = json.loads((root / "meta/entry_config.json").read_text())
        entry.update(stage="batch_pairing_ready", batch_pairing_implemented=True,
                     batch_loader="openpi.touchguide.batching.CPMBatchLoader.from_config")
        write_json(staging / "meta/entry_config.json", entry)
        updated_files.extend(("meta/manifest.json", "meta/entry_config.json"))
        readme = (root / "README.md").read_text().replace(
            "This is step 1 only: aligned positive examples and explicit reader configuration.",
            "Steps 1 and 2 are ready: aligned positive data, batch assembly and pairing labels.").replace(
            "Batch pairing, action noise augmentation and CPM training are not implemented here.",
            "Action noise, similarity scores, contrastive loss and CPM training are not implemented here.")
        readme += (
            "\n## Batch matching (step 2)\n\n"
            "Load batching explicitly; CPMDataset alone still reads individual matched examples.\n\n"
            "```python\nfrom openpi.touchguide.batching import CPMBatchLoader\n"
            "loader = CPMBatchLoader.from_config(root, 'train', epoch=0, num_workers=0)\n"
            "batch = next(iter(loader))\n"
            "# Feed head, wrist, tactile_map_raw and actions to the future CPM.\n"
            "# targets = [0, 1, ..., B-1] applies to rows and columns of its score matrix.\n"
            "# At the next training epoch: loader.set_epoch(1).\n"
            "loader.dataset.close()\n```\n\n"
            f"Training grouping: batch_size={args.batch_size}, seed={args.seed}, a new permutation per epoch, "
            f"drop_last=True ({pairing['train']['shuffled_tail_samples_dropped']} shuffled tail samples omitted per epoch).\n"
            f"Validation: {len(validation_batches)} fixed groups including a final batch of "
            f"{len(validation_batches[-1])}; all {lengths['val']} examples are used once.\n"
            "Saved val groups depend on batch_size; changing only val JSON batch_size is rejected.\n"
            "examples/batch_examples.json describes two small illustrative batches and source references.\n"
            "examples/pair_examples.jsonl lists all pairs in those illustrative batches, not all training batches.\n"
            "Pair labels indicate identity matching rather than simulated success or failure.\n"
        )
        (staging / "README.md").write_text(readme)
        updated_files.append("README.md")
        reproducibility = json.loads((root / "meta/reproducibility.json").read_text())
        reproducibility["batch_pairing_builder"] = {"path": str(Path(__file__).resolve()),
                                                     "sha256": sha256(Path(__file__).resolve())}
        module_path = Path(__import__("openpi.touchguide.batching", fromlist=["__file__"]).__file__)
        reproducibility["batching_module"] = {"path": str(module_path), "sha256": sha256(module_path)}
        for name in (*new_files, *updated_files):
            reproducibility["files_sha256"][name] = sha256(staging / name)
        write_json(staging / "meta/reproducibility.json", reproducibility)
        updated_files.append("meta/reproducibility.json")
        for name in new_files:
            destination = root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise FileExistsError(destination)
            (staging / name).rename(destination)
        for name in updated_files:
            (staging / name).replace(root / name)
    finally:
        shutil.rmtree(staging)
    print(json.dumps({"dataset_root": str(root), "batch_size": args.batch_size,
                      "train": pairing["train"], "val": pairing["val"],
                      "example_pairs_saved": len(pair_records), "training_started": False}, indent=2))


if __name__ == "__main__":
    main()
