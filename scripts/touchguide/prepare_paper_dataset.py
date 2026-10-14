"""Create a positive-only TouchGuide entry from the verified poker CPM data."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import h5py
import numpy as np


DATASETS = Path("/path/to/datasets/sim/poker-draw/pi05_trex/touchguide/cpm_datasets")


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DATASETS / "poker_0920_200_v1")
    parser.add_argument("--output-dir", type=Path, default=DATASETS / "poker_0920_200_touchguide_v1")
    args = parser.parse_args()
    source = args.source_root.expanduser().resolve(strict=True)
    output = args.output_dir.expanduser().resolve()
    staging = output.with_name(output.name + ".building")
    if any(path == source or source in path.parents for path in (output, staging)):
        raise ValueError("The new entry must be outside the source dataset directory")
    if any(path.exists() or path.is_symlink() for path in (output, staging)):
        raise FileExistsError(f"Output/staging already exists: {output}, {staging}")
    started = time.monotonic()
    parent_manifest_path = source / "meta/manifest.json"
    parent = json.loads(parent_manifest_path.read_text())
    configuration = parent["configuration"]
    if parent.get("schema") != "touchguide_cpm_dataset_v1":
        raise ValueError("Expected the verified v1 positive sample source")
    if any(configuration[key] != expected for key, expected in
           (("fps", 30), ("action_horizon", 30), ("action_dim", 27), ("stride", 4))):
        raise ValueError("Expected 30 Hz, 30-step, 27-dimensional, stride-4 poker data")
    if parent.get("normalized_action_checkpoint_step") != 20000:
        raise ValueError("Expected the 20k checkpoint action representation")
    norm_path = source / "meta/action_norm_stats.json"
    checkpoint_norm = Path(parent["sources"]["base_checkpoint"]) / "assets/normalizer/norm_stats.json"
    norm_hash = sha256(norm_path)
    if norm_hash != parent["sources"]["normalizer_sha256"] or norm_hash != sha256(checkpoint_norm):
        raise ValueError("Action normalization differs from the original 20k checkpoint")

    fields = {"frame_index": (), "timestamp_s": (), "state_raw": (27,),
              "actions_absolute": (30, 27), "actions": (30, 27),
              "tactile_map_raw": (5, 3, 7, 5), "contact_fingers": (5,),
              "source_state_index": (), "source_camera_timestamp_s": ()}
    splits, indices = {}, {}
    for split, expected_episodes, expected_count in (("train", 180, 39333), ("val", 20, 4372)):
        info = parent["splits"][split]
        if info["episodes"] != expected_episodes or info["count"] != expected_count:
            raise ValueError(f"Unexpected {split} episode/sample counts")
        with np.load(source / f"index/{split}.npz", allow_pickle=False) as data:
            index = {key: data[key].copy() for key in
                     ("episode_index", "sample_row", "frame_index", "timestamp_s")}
        if any(array.shape != (expected_count,) for array in index.values()):
            raise ValueError(f"Invalid {split} index shapes")
        if sorted(np.unique(index["episode_index"]).tolist()) != info["episode_indices"]:
            raise ValueError(f"Invalid {split} episode membership")
        indices[split] = index
        splits[split] = {key: info[key] for key in
                         ("episode_indices", "episodes", "count", "contact_finger_count_histogram")}
    if set(splits["train"]["episode_indices"]) & set(splits["val"]["episode_indices"]):
        raise ValueError("Training and validation episodes overlap")

    artifacts = []
    checked_videos = set()
    for split, index in indices.items():
        for episode_id in splits[split]["episode_indices"]:
            relative = f"episodes/episode_{episode_id:06d}.h5"
            path = source / relative
            expected = next(item for item in parent["artifacts"] if item["path"] == relative)
            actual_hash = sha256(path)
            if actual_hash != expected["sha256"]:
                raise ValueError(f"Source episode checksum changed: {path}")
            mask = index["episode_index"] == episode_id
            rows = index["sample_row"][mask]
            with h5py.File(path, "r") as episode:
                count = len(rows)
                if not np.array_equal(rows, np.arange(count)):
                    raise ValueError(f"Incomplete/duplicate episode row index: {episode_id}")
                for key, shape in fields.items():
                    if episode[key].shape != (count, *shape):
                        raise ValueError(f"Invalid episode field shape: {episode_id}/{key}")
                if str(episode.attrs["split"]) != split:
                    raise ValueError(f"Episode split attribute disagrees: {episode_id}")
                if not np.array_equal(episode["frame_index"][:], index["frame_index"][mask]):
                    raise ValueError(f"Episode frame index disagrees: {episode_id}")
                if not np.array_equal(episode["timestamp_s"][:], index["timestamp_s"][mask]):
                    raise ValueError(f"Episode timestamps disagree: {episode_id}")
                for key in ("head_video", "right_wrist_video"):
                    video = Path(episode.attrs[key])
                    if not video.is_file():
                        raise FileNotFoundError(video)
                    checked_videos.add(str(video))
            artifacts.append({**expected, "source_path": str(path), "storage": "source_symlink"})

    # This entry has its own indices, configuration and metadata. Only the
    # immutable positive H5 payload is referenced; no legacy negatives are linked.
    output.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    try:
        for name in ("meta", "index", "config"):
            (staging / name).mkdir()
        (staging / "episodes").symlink_to(source / "episodes", target_is_directory=True)
        for split in ("train", "val"):
            shutil.copyfile(source / f"index/{split}.npz", staging / f"index/{split}.npz")
        for name in ("action_norm_stats.json", "tactile_scale.json", "sources.json", "splits.json"):
            shutil.copyfile(source / "meta" / name, staging / "meta" / name)
        source_config = {key: configuration[key] for key in
                         ("fps", "action_horizon", "action_dim", "stride", "seed", "contact_taxel_threshold_newtons")}
        manifest = {
            "schema": "touchguide_cpm_positive_entry_v1", "stage": "positive_data_entry_only",
            "dataset_root": str(output),
            "parent": {"dataset_root": str(source), "manifest_sha256": sha256(parent_manifest_path)},
            "sources": parent["sources"], "normalized_action_checkpoint_step": 20000,
            "configuration": source_config, "splits": splits,
            "model_inputs": parent["model_inputs"], "auxiliary_only": parent["auxiliary_only"],
            "f6_included": False, "tactile_history_included": False,
            "action_convention": parent["action_convention"],
            "image_storage": parent["image_storage"], "tactile_storage": parent["tactile_storage"],
            "excluded": parent["excluded"], "artifacts": sorted(artifacts, key=lambda item: item["path"]),
            "paper_reference": "https://arxiv.org/html/2601.20239v6#S4.SS2",
            "planned_contrastive_training": {
                "implemented": False, "profile": "paper_batch_symmetric_infonce_v1",
                "negative_source": "in_batch_off_diagonal", "positive_source": "same_sample_diagonal",
                "action_representation": "stored_normalized_action_in_own_anchor_reference",
                "recipient_action_rebasing": False, "offline_negative_pool_used": False,
                "label_semantics": "observation-action pairing, not measured task success or failure",
            },
            "implementation_status": {"positive_entry": True, "batch_pairing": False,
                                      "noise_augmentation": False, "cpm_training": False},
        }
        write_json(staging / "meta/manifest.json", manifest)
        configs = {}
        for split in ("train", "val"):
            reader_kwargs = {"root": str(output), "split": split, "include_images": True,
                             "include_negatives": False, "max_open_h5": 4}
            configs[split] = {"reader": "openpi.touchguide.data.CPMDataset", "reader_kwargs": reader_kwargs,
                             "noise_on_read": False,
                             "image_preprocessing": {"size": [224, 224], "layout": "CHW",
                                                     "range": [-1.0, 1.0], "resize": "openpi_resize_with_pad"},
                             "tactile_preprocessing": {"mode": "raw", "unit": "N", "clip": False},
                             "model_input_keys": ["head", "wrist", "tactile_map_raw", "actions"],
                             "auxiliary_keys": ["state_raw", "contact_fingers", "id", "metadata"]}
            write_json(staging / f"config/{split}.json", configs[split])
        write_json(staging / "meta/entry_config.json", {
            "stage": "positive_data_entry_only", "reader_configs": {s: f"config/{s}.json" for s in configs},
            "configs_must_be_explicitly_loaded": True, "reader_default_root_changed": False,
            "legacy_negative_pool_used": False, "noise_augmentation_implemented": False,
            "batch_pairing_implemented": False, "training_started": False,
        })
        write_json(staging / "meta/entry_validation.json", {
            "status": "passed", "source_episode_hashes_checked": len(artifacts),
            "source_video_paths_checked": len(checked_videos), "split_disjoint": True,
            "episode_shapes_and_index_alignment_checked": True,
            "action_normalizer_matches_20k_checkpoint": True, "action_normalizer_sha256": norm_hash,
            "positive_payload_modified": False, "image_decode_performed": False,
            "source_full_validation": str(source / "meta/validation.json"),
            "source_reader_validation": str(source / "meta/reader_validation.json"),
        })
        own_artifacts = ("index/train.npz", "index/val.npz", "config/train.json", "config/val.json",
                         "meta/manifest.json", "meta/action_norm_stats.json", "meta/tactile_scale.json",
                         "meta/sources.json", "meta/splits.json", "meta/entry_config.json")
        write_json(staging / "meta/reproducibility.json", {
            "builder_path": str(Path(__file__).resolve()), "builder_sha256": sha256(Path(__file__).resolve()),
            "parent_manifest_sha256": sha256(parent_manifest_path),
            "files_sha256": {name: sha256(staging / name) for name in own_artifacts},
        })
        (staging / "README.md").write_text(
            "# Poker TouchGuide positive data entry\n\n"
            "This is step 1 only: aligned positive examples and explicit reader configuration.\n"
            "The planned objective is same-sample matching with in-batch, symmetric InfoNCE.\n"
            "Batch pairing, action noise augmentation and CPM training are not implemented here.\n\n"
            "- Train: 180 episodes / 39,333 samples; validation: 20 episodes / 4,372 samples.\n"
            "- 30 Hz, stride 4, complete 30-step chunks, 27 action dimensions.\n"
            "- Inputs: current head/wrist RGB, current right five-finger 3D taxels, candidate actions.\n"
            "- Stored actions: arm delta 7 + absolute hand targets 20, original 20k quantile normalization.\n"
            "- State, contact flags and provenance are auxiliary, not independent model inputs.\n"
            "- Raw tactile values are in N; no clipping/scaling/noise is applied on read.\n"
            "- Reader RGB output is CHW 224x224 in [-1, 1]; encoder preprocessing is a later step.\n"
            "- meta/tactile_scale.json contains inherited train-only statistics, not an active transform.\n"
            "- episodes is a symlink to the verified source; keep that source directory available.\n"
            "- Camera videos remain referenced by absolute source paths.\n"
            "- No legacy negative pool, rebasing, RMSE filter or noise-level grid is enabled.\n"
            "- Validation episodes are held out for CPM; they may have been used to train base pi05.\n\n"
            "## Read explicitly\n\n"
            "The existing reader does not automatically load JSON configuration, and its default\n"
            "root still points to the older dataset. Load this entry's config explicitly:\n\n"
            "```python\nimport json\nfrom pathlib import Path\n"
            "from openpi.touchguide.data import CPMDataset\n\n"
            f"root = Path({str(output)!r})\n"
            "config = json.loads((root / 'config/train.json').read_text())\n"
            "dataset = CPMDataset(**config['reader_kwargs'])\n"
            "# Supply only config['model_input_keys'] to the future CPM.\n"
            "```\n"
        )
        if output.exists() or output.is_symlink():
            raise FileExistsError(output)
        staging.rename(output)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    print(json.dumps({"dataset_root": str(output), "train_samples": 39333, "val_samples": 4372,
                      "source_episode_hashes_checked": len(artifacts), "training_started": False,
                      "duration_seconds": round(time.monotonic() - started, 3)}, indent=2))


if __name__ == "__main__":
    main()
