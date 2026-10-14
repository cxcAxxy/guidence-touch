"""Configure online flow action noise without rewriting clean source episodes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from openpi.touchguide.batching import CPMBatchLoader
from openpi.touchguide.noise import FlowActionNoise
from openpi.touchguide.noise import NoiseConfig

DEFAULT_ROOT = Path(
    "/path/to/datasets/sim/poker-draw/pi05_trex/touchguide/cpm_datasets/poker_0920_200_touchguide_v1"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--num-flow-steps", type=int, default=10)
    parser.add_argument("--geometric-p", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    noise = FlowActionNoise(NoiseConfig(num_flow_steps=args.num_flow_steps, geometric_p=args.geometric_p))
    if args.seed < 0:
        raise ValueError("seed must be nonnegative")
    root = args.dataset_root.expanduser().resolve(strict=True)
    manifest = json.loads((root / "meta/manifest.json").read_text())
    if (
        manifest["schema"] != "touchguide_cpm_positive_entry_v1"
        or not manifest["implementation_status"]["batch_pairing"]
    ):
        raise ValueError("Prepare the dedicated positive entry and batch pairing first")
    if manifest["implementation_status"]["noise_augmentation"]:
        raise FileExistsError("Noise configuration is already prepared; existing artifacts are not overwritten")
    new_files = ("meta/noise_augmentation.json", "examples/noise_examples.npz", "examples/noise_examples.json")
    if any((root / name).exists() for name in new_files):
        raise FileExistsError("One or more noise artifacts already exist")
    staging = root / ".noise.building"
    if staging.exists():
        raise FileExistsError(staging)
    flow_times = [level / args.num_flow_steps for level in range(args.num_flow_steps)]
    settings = {
        "loader": "openpi.touchguide.noise.NoiseAugmentedBatchLoader.from_config",
        "enabled": True,
        "application_stage": "after_clean_batch_collation",
        "num_flow_steps": args.num_flow_steps,
        "geometric_p": args.geometric_p,
        "time_scope": "batch",
        "seed": args.seed,
        "distribution": "truncated_renormalized_geometric",
        "random_noise_levels": list(range(args.num_flow_steps)),
        "flow_time_mapping": "noise_level / num_flow_steps",
        "formula": "noisy_actions = (1 - flow_time) * actions + flow_time * epsilon",
        "epsilon_distribution": "standard_normal_in_normalized_action_coordinates",
        "epsilon_scope": "independent_per_action_column_reused_for_all_observation_rows",
        "cpm_action_input_key": "noisy_actions",
        "clean_action_key": "actions",
        "clip": False,
        "renormalize_after_noise": False,
        "local_parameter_choices": ["geometric_p", "num_flow_steps", "time_grid", "batch_time_scope"],
    }
    profile = {
        "profile": "flow_action_noise_v1",
        **settings,
        "probabilities": noise.probabilities.tolist(),
        "random_flow_times": flow_times,
        "pure_noise_t1_randomly_sampled": False,
        "train_mode": "random_time_per_batch_and_independent_epsilon_per_column",
        "train_seed_recipe": "blake2b(seed:epoch:{epoch},8 bytes,little endian) & (2**63-1)",
        "validation_mode": "fixed_epsilon_per_sample_id_reused_across_all_times",
        "validation_seed_recipe": "blake2b(seed:sample_id,8 bytes,little endian) & (2**63-1)",
        "validation_times": flow_times,
        "validation_default_time": 0.0,
        "validation_default_is_clean_baseline": True,
        "targets_and_identity_masks_changed": False,
        "negative_semantics": "identity_mismatch_not_verified_physical_failure",
        "raw_h5_rewritten": False,
        "offline_noisy_dataset_created": False,
        "implemented": {
            "noise_augmentation": True,
            "similarity_scores": False,
            "contrastive_loss": False,
            "cpm_training": False,
        },
        "paper_method": "geometric sampling of noising steps and linear flow interpolation",
        "paper_url": "https://arxiv.org/html/2601.20239v6",
        "files": list(new_files),
    }
    configs = {}
    for split in ("train", "val"):
        config = json.loads((root / f"config/{split}.json").read_text())
        if config["reader_kwargs"]["include_negatives"] or config["noise_on_read"]:
            raise ValueError("Keep the reader clean and legacy negatives disabled")
        if "noise_augmentation" in config:
            raise FileExistsError("Existing noise configuration is not overwritten")
        config["noise_augmentation"] = {
            **settings,
            "mode": "random_per_batch" if split == "train" else "fixed_per_sample_id",
        }
        config["augmented_model_input_keys"] = ["head", "wrist", "tactile_map_raw", "noisy_actions"]
        if split == "val":
            config["noise_augmentation"].update(validation_times=flow_times, validation_default_time=0.0)
        configs[split] = config

    # Illustrative artifacts only: full training augmentation remains online.
    loader = CPMBatchLoader.from_config(root, "val", include_images=False)
    try:
        example_indices = next(iter(loader.index_batches))[:3]
        clean = next(iter(CPMBatchLoader(loader.dataset, [example_indices])))
        results = [noise.augment_validation_at_time(clean, time, seed=args.seed) for time in flow_times]
    finally:
        loader.dataset.close()
    examples = {
        "kind": "small_fixed_validation_noise_example",
        "split": "val",
        "sample_indices": example_indices,
        "sample_ids": clean["sample_ids"],
        "source_metadata": clean["metadata"],
        "seed": args.seed,
        "flow_times": flow_times,
        "npz_file": "examples/noise_examples.npz",
        "actions_shape": list(clean["actions"].shape),
        "noisy_actions_shape": [len(flow_times), *clean["actions"].shape],
        "epsilon_shared_across_times": True,
        "identical_action_columns_used_in_all_pair_rows": True,
    }
    staging.mkdir()
    try:
        write_json(staging / "meta/noise_augmentation.json", profile)
        write_json(staging / "examples/noise_examples.json", examples)
        np.savez_compressed(
            staging / "examples/noise_examples.npz",
            actions=clean["actions"].numpy(),
            noise_epsilon=results[0]["noise_epsilon"].numpy(),
            noisy_actions=np.stack([result["noisy_actions"].numpy() for result in results]),
            flow_times=np.asarray(flow_times, dtype=np.float32),
            sample_ids=np.asarray(clean["sample_ids"]),
            targets=clean["targets"].numpy(),
            positive_mask=clean["positive_mask"].numpy(),
            negative_mask=clean["negative_mask"].numpy(),
        )
        updated = []
        for split, config in configs.items():
            name = f"config/{split}.json"
            write_json(staging / name, config)
            updated.append(name)
        manifest["stage"] = "noise_augmentation_ready"
        manifest["implementation_status"]["noise_augmentation"] = True
        manifest["noise_augmentation"] = profile
        # This nested record describes stage 2; current overall status is above.
        manifest["batching"]["status_scope"] = "step_2_clean_batch_pairing_snapshot"
        write_json(staging / "meta/manifest.json", manifest)
        entry = json.loads((root / "meta/entry_config.json").read_text())
        entry.update(
            stage="noise_augmentation_ready",
            noise_augmentation_implemented=True,
            noise_loader=settings["loader"],
            cpm_action_input_key="noisy_actions",
            training_started=False,
        )
        write_json(staging / "meta/entry_config.json", entry)
        updated.extend(("meta/manifest.json", "meta/entry_config.json"))
        readme = (
            (root / "README.md")
            .read_text()
            .replace(
                "Steps 1 and 2 are ready: aligned positive data, batch assembly and pairing labels.",
                "Steps 1 to 3 are ready: aligned positive data, batch pairing and online action noise.",
            )
            .replace(
                "Action noise, similarity scores, contrastive loss and CPM training are not implemented here.",
                "Similarity scores, contrastive loss and CPM training are not implemented here.",
            )
            .replace(
                "No legacy negative pool, rebasing, RMSE filter or noise-level grid is enabled.",
                "No legacy negative pool, rebasing or RMSE filter is enabled. Noise is applied after clean collation.",
            )
        )
        readme += (
            "\n## Online action noise (step 3)\n\n"
            "Use the noise wrapper explicitly. noise_on_read stays false: H5 and the clean loader are unchanged.\n\n"
            "```python\nfrom openpi.touchguide.noise import NoiseAugmentedBatchLoader\n"
            "train_loader = NoiseAugmentedBatchLoader.from_config(root, 'train', epoch=0)\n"
            "batch = next(iter(train_loader))\n"
            "# Future CPM input: head, wrist, tactile_map_raw, noisy_actions (and flow_time if time-conditioned).\n"
            "# actions is retained for reference; identity targets/masks are unchanged.\n"
            "train_loader.set_epoch(1)\ntrain_loader.dataset.close()\n\n"
            f"# Validate every noise level explicitly, using the same epsilon for each sample ID.\n"
            f"for time in {flow_times!r}:\n"
            "    val_loader = NoiseAugmentedBatchLoader.from_config(root, 'val', validation_time=time)\n"
            "    batch = next(iter(val_loader))  # A future evaluation loop must consume all groups.\n"
            "    val_loader.dataset.close()\n```\n\n"
            f"Local settings: K={args.num_flow_steps}, p={args.geometric_p}, seed={args.seed}; "
            "P(k)=p(1-p)^k/[1-(1-p)^K], k=0..K-1; time=k/K.\n"
            "Time is shared by the batch; independent standard-normal epsilon is drawn for each action column.\n"
            "noisy_actions=(1-time)*actions+time*epsilon in the existing normalized action space.\n"
            "Do not clip or renormalize noisy actions, and do not make separate noisy candidates per observation row.\n"
            "Training excludes time=1 because it contains no action identity. time=0 is a clean training batch.\n"
            "Validation default time=0 is only the clean baseline; assess all configured validation_times.\n"
            "Fixed validation epsilon follows sample ID and seed, independent of grouping and time.\n"
            "The paper specifies geometric sampling and linear flow interpolation, but does not fix our p/grid/batch-time settings.\n"
            "examples/noise_examples.npz and .json contain a small three-sample illustration, not a new full dataset.\n"
        )
        (staging / "README.md").write_text(readme)
        updated.append("README.md")
        reproducibility = json.loads((root / "meta/reproducibility.json").read_text())
        reproducibility["action_noise_builder"] = {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
        }
        module_path = Path(__import__("openpi.touchguide.noise", fromlist=["__file__"]).__file__)
        reproducibility["action_noise_module"] = {"path": str(module_path), "sha256": sha256(module_path)}
        for name in (*new_files, *updated):
            reproducibility["files_sha256"][name] = sha256(staging / name)
        write_json(staging / "meta/reproducibility.json", reproducibility)
        updated.append("meta/reproducibility.json")
        for name in new_files:
            destination = root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise FileExistsError(destination)
            (staging / name).rename(destination)
        for name in updated:
            (staging / name).replace(root / name)
    finally:
        shutil.rmtree(staging)
    print(
        json.dumps(
            {
                "dataset_root": str(root),
                "num_flow_steps": args.num_flow_steps,
                "geometric_p": args.geometric_p,
                "flow_times": flow_times,
                "probabilities": noise.probabilities.tolist(),
                "raw_h5_rewritten": False,
                "training_started": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
