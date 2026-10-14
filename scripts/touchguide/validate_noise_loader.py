"""Check configured CPM noise wrappers and saved examples without loading a model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from openpi.touchguide.batching import CPMBatchLoader
from openpi.touchguide.noise import NoiseAugmentedBatchLoader


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    loaders = []
    try:
        train = NoiseAugmentedBatchLoader.from_config(root, "train", include_images=False)
        loaders.append(train)
        first = next(iter(train))
        repeat = next(iter(train))
        assert first["actions"].shape == (64, 30, 27)
        assert first["sample_ids"] == repeat["sample_ids"]
        assert torch.equal(first["noisy_actions"], repeat["noisy_actions"])
        assert torch.equal(first["noise_epsilon"], repeat["noise_epsilon"])
        assert first["flow_time"].shape == ()
        train.set_epoch(1)
        changed = next(iter(train))
        assert first["sample_ids"] != changed["sample_ids"]
        assert not torch.equal(first["noise_epsilon"], changed["noise_epsilon"])
        independent = NoiseAugmentedBatchLoader.from_config(root, "train", epoch=1, include_images=False)
        loaders.append(independent)
        reproduced = next(iter(independent))
        assert changed["sample_ids"] == reproduced["sample_ids"]
        assert torch.equal(changed["noisy_actions"], reproduced["noisy_actions"])
        val = NoiseAugmentedBatchLoader.from_config(root, "val", validation_time=0.4, include_images=False)
        loaders.append(val)
        fixed = next(iter(val))
        assert fixed["actions"].shape == (64, 30, 27)
        baseline = NoiseAugmentedBatchLoader.from_config(root, "val", include_images=False)
        loaders.append(baseline)
        clean_baseline = next(iter(baseline))
        assert fixed["sample_ids"] == clean_baseline["sample_ids"]
        assert torch.equal(fixed["noise_epsilon"], clean_baseline["noise_epsilon"])
        assert torch.equal(clean_baseline["noisy_actions"], clean_baseline["actions"])
        for batch in (first, changed, fixed, clean_baseline):
            size = len(batch["sample_ids"])
            assert torch.equal(batch["targets"], torch.arange(size))
            assert torch.equal(batch["positive_mask"], torch.eye(size, dtype=torch.bool))
            assert torch.equal(batch["negative_mask"], ~batch["positive_mask"])
            torch.testing.assert_close(
                batch["noisy_actions"],
                (1 - batch["flow_time"]) * batch["actions"] + batch["flow_time"] * batch["noise_epsilon"],
                rtol=0,
                atol=0,
            )
            assert "negative_actions" not in batch
        configurations = {split: json.loads((root / f"config/{split}.json").read_text()) for split in ("train", "val")}
        for split, config in configurations.items():
            assert config["noise_on_read"] is False
            assert config["reader_kwargs"]["include_negatives"] is False
            assert config["noise_augmentation"]["cpm_action_input_key"] == "noisy_actions"
            assert config["augmented_model_input_keys"] == ["head", "wrist", "tactile_map_raw", "noisy_actions"]
            expected_mode = "random_per_batch" if split == "train" else "fixed_per_sample_id"
            assert config["noise_augmentation"]["mode"] == expected_mode
        clean_loader = CPMBatchLoader.from_config(root, "train", include_images=False)
        loaders.append(clean_loader)
        clean = next(iter(clean_loader))
        assert "noisy_actions" not in clean
        assert torch.equal(clean["actions"], first["actions"])
        reference = json.loads((root / "examples/noise_examples.json").read_text())
        illustration = next(iter(CPMBatchLoader(val.dataset, [reference["sample_indices"]])))
        assert illustration["sample_ids"] == reference["sample_ids"]
        with np.load(root / reference["npz_file"], allow_pickle=False) as saved:
            assert np.array_equal(saved["actions"], illustration["actions"].numpy())
            assert saved["sample_ids"].tolist() == illustration["sample_ids"]
            assert np.array_equal(saved["targets"], illustration["targets"].numpy())
            assert np.array_equal(saved["positive_mask"], illustration["positive_mask"].numpy())
            assert np.array_equal(saved["negative_mask"], illustration["negative_mask"].numpy())
            assert reference["flow_times"] == configurations["val"]["noise_augmentation"]["validation_times"]
            for slot, time in enumerate(reference["flow_times"]):
                noisy = val.augmenter.augment_validation_at_time(illustration, time, seed=reference["seed"])
                assert np.array_equal(saved["noise_epsilon"], noisy["noise_epsilon"].numpy())
                assert np.array_equal(saved["noisy_actions"][slot], noisy["noisy_actions"].numpy())
                assert saved["flow_times"][slot] == np.float32(time)
        report = {
            "status": "passed",
            "scope": "configured_noise_wrappers_and_saved_examples_cpu",
            "dataset_root": str(root),
            "manifest_sha256": hashlib.sha256((root / "meta/manifest.json").read_bytes()).hexdigest(),
            "train": {
                "batch_size": 64,
                "batches_per_epoch": len(train),
                "first_flow_time": float(first["flow_time"]),
                "epoch0_reproducible": True,
                "set_epoch_changes_groups_and_epsilon": True,
                "epoch1_matches_independent_from_config": True,
            },
            "val": {
                "batch_size": 64,
                "batches": len(val),
                "checked_flow_time": 0.4,
                "default_clean_baseline_verified": True,
                "epsilon_shared_across_levels": True,
            },
            "clean_loader_unchanged": True,
            "targets_and_masks_verified": True,
            "saved_examples_verified_against_real_sources_at_all_levels": True,
            "augmented_model_input_key_verified": True,
            "images_enabled": False,
            "gpu_tested": False,
            "model_loaded": False,
            "training_run": False,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"status": "passed", "report": str(args.report)}, indent=2))
    finally:
        for loader in loaders:
            loader.dataset.close()


if __name__ == "__main__":
    main()
