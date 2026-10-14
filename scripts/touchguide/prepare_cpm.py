"""Prepare the frozen-DINO CPM v2 configuration without loading a model or training.

An explicit --upgrade-from-cnn-v1 archives the old configuration, metadata and
CPM validation reports before changing the positive data entry. Repeating this
command on v2 verifies the artifacts and returns without modifying them.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import UTC
from datetime import datetime
import hashlib
import importlib
import json
import math
from pathlib import Path
import re
import shutil

from openpi.touchguide.model import CPMConfig

DEFAULT_ROOT = Path(
    "/path/to/datasets/sim/poker-draw/pi05_trex/touchguide/cpm_datasets/poker_0920_200_touchguide_v1"
)
DEFAULT_BACKBONE = Path("/path/to/datasets/sim/poker-draw/pi05_trex/touchguide/backbones/dinov2_base_f9e44c8")
SCHEMA = "openpi_touchguide_cpm_config_v2"
SNAPSHOT = "meta/cpm_cnn_v1_snapshot"
COUNTS = {"trainable": 122262529, "frozen": 86580480, "total": 208843009}
UPDATED_FILES = (
    "config/cpm.json",
    "meta/cpm_architecture.json",
    "meta/manifest.json",
    "meta/entry_config.json",
    "meta/reproducibility.json",
    "config/train.json",
    "config/val.json",
    "README.md",
)
PINNED_FILES = {
    "config.json": (548, "f7ff4cfa73d2f70647dbf6950541ad25d73082d54c2e7e9bded160c7656b2a70"),
    "preprocessor_config.json": (436, "14e780d86fa1861f8751f868d7f45425b5feb55c38ca26f152ca5097ab30f828"),
    "model.safetensors": (346345912, "d73036b56966966d07975d696bde331762f37297e2f095de8cea0040c3aa0841"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def verify_backbone(vision: Path) -> dict:
    manifest_path = vision / "backbone_manifest.json"
    backbone = read_json(manifest_path)
    if (
        backbone.get("schema") != "touchguide_public_backbone_v1"
        or backbone.get("repository") != "facebook/dinov2-base"
        or backbone.get("revision") != "f9e44c814b77203eaa57a6bdbbd535f21ede1415"
        or backbone.get("published_target") != str(vision)
    ):
        raise ValueError("Use the verified pinned facebook/dinov2-base dependency")
    for name, (size, expected_hash) in PINNED_FILES.items():
        path = vision / name
        record = backbone["files"][name]
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != size
            or sha256(path) != expected_hash
            or record["size_bytes"] != size
            or record["sha256"] != expected_hash
        ):
            raise ValueError(f"Pinned backbone artifact mismatch: {name}")
    return backbone


def source_records() -> dict:
    files = {"cpm_builder": Path(__file__).resolve()}
    for key, module in (
        ("cpm_model", "openpi.touchguide.model"),
        ("cpm_loss", "openpi.touchguide.losses"),
        ("cpm_tactile_preprocessor", "openpi.touchguide.tactile_dino"),
        ("cpm_trainer", "openpi.touchguide.train"),
        ("cpm_checkpoint", "openpi.touchguide.checkpoint"),
    ):
        files[key] = Path(importlib.import_module(module).__file__)
    return {key: {"path": str(path), "sha256": sha256(path)} for key, path in files.items()}


def preserved_hashes(root: Path, reproducibility: dict) -> dict:
    return {
        name: sha256(root / name)
        for name in reproducibility["files_sha256"]
        if name not in UPDATED_FILES and not name.startswith("meta/cpm")
    }


def verify_preserved(root: Path, records: dict) -> None:
    for name, expected in records.items():
        if sha256(root / name) != expected:
            raise ValueError(f"Positive-data/batching/noise artifact changed: {name}")


def verify_v2(root: Path, vision: Path, config: dict) -> dict:
    verify_backbone(vision)
    if config["vision"]["directory"] != str(vision):
        raise ValueError("Existing v2 config uses a different backbone directory")
    if config["vision"]["manifest_sha256"] != sha256(vision / "backbone_manifest.json"):
        raise ValueError("Backbone provenance changed after CPM preparation")
    tactile = config["tactile"]
    if (
        tactile["normalization"] != "signed_asinh_train_rms_no_clip"
        or tactile["trainable"]
        or tactile["shared_backbone"] != "vision"
        or not config["vision"]["frozen"]
        or config["observation_tokens"] != 456
    ):
        raise ValueError("Existing config does not satisfy the frozen-DINO CPM v2 contract")
    if sha256(root / tactile["statistics_file"]) != tactile["statistics_sha256"]:
        raise ValueError("Train-only tactile scale changed")
    kwargs = json.loads(json.dumps(asdict(CPMConfig())))
    if config["model_kwargs"] != kwargs:
        raise ValueError("Existing v2 architecture differs from CPMConfig defaults")
    architecture = read_json(root / "meta/cpm_architecture.json")
    if architecture["parameter_counts"] != COUNTS:
        raise ValueError("Existing v2 parameter budget differs")
    reproducibility = read_json(root / "meta/reproducibility.json")
    for name in UPDATED_FILES:
        if name == "meta/reproducibility.json":
            continue
        if sha256(root / name) != reproducibility["files_sha256"][name]:
            raise ValueError(f"Existing v2 metadata hash mismatch: {name}")
    verify_preserved(root, reproducibility["cpm_v2_preparation"]["preserved_inputs_sha256"])
    return {"status": "verified_existing_v2", "dataset_root": str(root), "parameter_counts": COUNTS}


def make_config(root: Path, vision: Path, backbone: dict, stats: dict) -> dict:
    kwargs = asdict(CPMConfig())
    return {
        "schema": SCHEMA,
        "model": "openpi.touchguide.model.CPMModel.from_config",
        "model_kwargs": kwargs,
        "vision": {
            "directory": str(vision),
            "repository": backbone["repository"],
            "revision": backbone["revision"],
            "frozen": True,
            "manifest_sha256": sha256(vision / "backbone_manifest.json"),
            "input": "reader RGB [-1,1] converted once to [0,1], then ImageNet mean/std",
            "resize_or_crop_again": False,
            "tokens_per_view_before_pooling": 257,
            "tokens_per_view": 65,
            "spatial_pooling": "keep_CLS_and_average_pool_16x16_patches_to_8x8",
            "views": ["head", "wrist"],
        },
        "tactile": {
            "shape": [5, 3, 7, 5],
            "statistics_file": "meta/tactile_scale.json",
            "statistics_sha256": sha256(root / "meta/tactile_scale.json"),
            "train_scale_file": "meta/tactile_scale.json",
            "train_scale_sha256": sha256(root / "meta/tactile_scale.json"),
            "normalization": "signed_asinh_train_rms_no_clip",
            "rms": stats["rms"],
            "channel_order": stats["channel_order"],
            "unit": "N",
            "clip": False,
            "center": False,
            "encoder": "shared_frozen_DINOv2_Base",
            "preprocessor": "openpi.touchguide.tactile_dino.TactileDinoPreprocessor",
            "shared_backbone": "vision",
            "trainable": False,
            "tokens_per_finger": 65,
            "finger_identity_embedding": True,
            "mapping_formula": "0.5 + asinh(force_N / train_channel_rms_N) / (2 * asinh(reference_multiple))",
            "reference_multiple": kwargs["tactile_reference_multiple"],
            "image_size": [224, 224],
            "resized_force_grid": [224, 160],
            "horizontal_padding": [32, 32],
            "resize": "bilinear_align_corners_false_preserve_7_by_5_geometry",
            "padding_value": 0.5,
            "pixel_range": "unclipped_may_exceed_0_to_1",
            "image_normalization": "ImageNet_mean_std_once",
            "per_frame_normalization": False,
            "local_preprocessing_choice": True,
        },
        "observation_tokens": 456,
        "action": {
            "input_key": "noisy_actions",
            "time_key": "flow_time",
            "shape": [30, 27],
            "encoder": "temporal_1d_cnn_then_ordered_pooling_and_mlp",
            "conv_channels": [27, *kwargs["action_channels"]],
            "conv_kernel_sizes": [5, 3, 3],
            "ordered_pool_bins": 6,
            "mlp_dims": [6144, 2048, 1024],
            "clip": False,
            "renormalize": False,
        },
        "score": "raw_cosine_of_l2_normalized_observation_and_action_embeddings",
        "training_logits": "cosine_scores / learnable_temperature",
        "guidance_score": "raw_cosine_not_softmax_not_temperature_scaled",
        "loss": "openpi.touchguide.losses.cpm_contrastive_loss",
        "loss_formula": "0.5*(CE(logits,arange(B))+CE(logits.T,arange(B)))",
        "metrics": [
            "loss",
            "loss_obs_to_action",
            "loss_action_to_obs",
            "accuracy_obs_to_action",
            "accuracy_action_to_obs",
            "positive_score",
            "negative_score",
            "score_gap",
            "temperature",
        ],
        "labels": "same_sample_matching_not_verified_physical_success",
        "parameter_counts": COUNTS,
        "pi05_loaded_by_cpm": False,
        "cpm_training_started": False,
    }


def make_architecture(manifest: dict, config: dict) -> dict:
    training = deepcopy(manifest.get("cpm", {}).get("training", {}))
    training.update(
        entry="openpi.touchguide.train",
        default_epochs=200,
        default_batch_size=64,
        learning_rate=1e-5,
        warmup_ratio=0.05,
        optimizer="AdamW",
        formal_training_started=False,
        steps_per_epoch=manifest["splits"]["train"]["count"] // 64,
        validation="fixed_all_groups_complete_noise_grid_at_epoch_end",
        checkpoint_best_metric="val/loss",
    )
    training["total_steps"] = training["steps_per_epoch"] * training["default_epochs"]
    return {
        "implementation": "local_paper_inspired_cpm_v2",
        "config_file": "config/cpm.json",
        "config_schema": SCHEMA,
        "paper": {
            "url": "https://arxiv.org/html/2601.20239v6",
            "reported_trainable_parameters": 118000000,
            "implemented_elements": [
                "shared_frozen_pretrained_visual_and_tactile_DINOv2",
                "transformer_observation_fusion",
                "1d_cnn_plus_mlp_action_encoder",
                "l2_cosine_score",
                "learnable_temperature",
                "symmetric_infonce",
            ],
            "training_defaults_reference": "Appendix_L_Table_XIV",
        },
        "official_cpm_source_available_in_local_checkout": False,
        "local_adaptations": [
            "DINOv2_Base_variant",
            "signed_asinh_train_rms_force_to_image_without_clipping",
            "same_shared_frozen_backbone_for_two_camera_and_five_finger_views",
            "preserve_each_view_CLS_and_average_pool_patch_grid_to_8x8",
            "eight_layer_1024dim_16head_4096ff_fusion_with_456_tokens",
            "expanded_256_512_1024_temporal_CNN_and_6144_2048_1024_action_MLP",
            "30_by_27_action_shape_and_six_ordered_temporal_pool_bins",
            "explicit_sinusoidal_flow_time_embedding",
            "temperature_initialization_and_bounds",
        ],
        "trainable_components": [
            "vision_projection",
            "tactile_projection",
            "modality_and_finger_embeddings",
            "observation_cls",
            "fusion",
            "observation_projection",
            "action_encoder",
            "logit_scale",
        ],
        "frozen_components": ["shared_DINOv2_Base_backbone"],
        "parameter_counts": COUNTS,
        "observation_tokens": config["observation_tokens"],
        "base_pi05_updated": False,
        "existing_trex_checkpoints_updated": False,
        "network_implemented": True,
        "similarity_scores_implemented": True,
        "contrastive_loss_implemented": True,
        "optimizer_or_training_loop_implemented": True,
        "training_started": False,
        "validation_status": "pending_for_v2_architecture",
        "training": training,
        "guidance_gradient_api": "CPMModel.encode_observation + CPMModel.paired_scores",
        "guidance_sampler_integrated": False,
        "legacy_cnn_checkpoints_compatible": False,
    }


def make_readme(previous: str) -> str:
    heading = re.search(r"\n## CPM (?:network|training)", previous)
    prefix = previous[: heading.start()] if heading else previous
    prefix = prefix.replace(
        "CPM network, cosine scores and symmetric InfoNCE are implemented; the training loop remains pending.",
        "CPM v2 network, cosine scores, symmetric InfoNCE and training loop are implemented.",
    )
    return (
        prefix.rstrip()
        + """\n\n## CPM network and loss (v2)\n\n
config/cpm.json selects one verified, shared frozen DINOv2-Base for both camera RGB and current force maps.
The local architecture is 8 Transformer layers, width 1024, 16 heads and FF width 4096.
Each of seven views retains DINO CLS plus an 8x8 average-pooled patch grid; a fusion CLS gives 456 tokens.
Five force maps preserve normal/tangent0/tangent1 channels, signs and physical zero.
The local signed asinh map uses fixed train-only channel RMS, preserves 7:5 geometry and applies no clipping or per-frame scaling.
Force images are bilinearly resized to 224x160, padded horizontally with neutral 0.5, then ImageNet-normalized once.
The trainable action branch uses Conv1D 27->256->512->1024, six ordered pool bins and MLP 6144->2048->1024.
A local sinusoidal flow-time embedding is retained. Observation and action embeddings are L2-normalized.
Training averages both directions of in-batch InfoNCE using cosine/learnable_temperature; guidance uses raw cosine.
Trainable parameters: 122,262,529; frozen shared backbone: 86,580,480; total: 208,843,009.
DINO variant, layer widths, spatial pooling, action widths and force mapping are local choices near the paper's reported 118M capacity.
Default training: batch 64, 200 epochs, AdamW learning rate 1e-5 and warmup ratio 0.05.
Training has not started. The previous CNN configuration and validation reports are archived under meta/cpm_cnn_v1_snapshot when upgraded.
New architecture checks are pending; archived CNN reports do not validate this model.
Scores indicate identity matching, not calibrated success or physical failure. No pi05 weights are loaded or updated by CPM preparation.
"""
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--vision-directory", type=Path, default=DEFAULT_BACKBONE)
    parser.add_argument("--upgrade-from-cnn-v1", action="store_true")
    args = parser.parse_args()
    root = args.dataset_root.expanduser().resolve(strict=True)
    vision = args.vision_directory.expanduser().resolve(strict=True)
    current_path = root / "config/cpm.json"
    current = read_json(current_path) if current_path.exists() else None
    if current is not None and current.get("schema") == SCHEMA:
        print(json.dumps(verify_v2(root, vision, current), indent=2))
        return
    upgrading = current is not None
    if upgrading and (not args.upgrade_from_cnn_v1 or current.get("schema") != "openpi_touchguide_cpm_config_v1"):
        raise FileExistsError(
            "Existing CPM config requires explicit --upgrade-from-cnn-v1; unknown schemas are never overwritten"
        )
    if not upgrading and args.upgrade_from_cnn_v1:
        raise FileNotFoundError("There is no CNN v1 configuration to upgrade")
    staging = root / ".cpm_v2.building"
    if staging.exists() or staging.is_symlink():
        raise FileExistsError(f"Existing staging directory is not overwritten: {staging}")
    if upgrading and (root / SNAPSHOT).exists():
        raise FileExistsError(f"Existing CNN snapshot is not overwritten: {root / SNAPSHOT}")
    manifest = read_json(root / "meta/manifest.json")
    if not manifest["implementation_status"]["noise_augmentation"]:
        raise ValueError("Prepare batching and noise augmentation before CPM")
    backbone = verify_backbone(vision)
    stats = read_json(root / "meta/tactile_scale.json")
    if (
        stats["channel_order"] != ["normal", "tangent0", "tangent1"]
        or stats["unit"] != "N"
        or len(stats["rms"]) != 3
        or any(not math.isfinite(value) or value <= 0 for value in stats["rms"])
    ):
        raise ValueError("Expected three positive train-only RMS values in force channel order")
    reproducibility = read_json(root / "meta/reproducibility.json")
    preserved = preserved_hashes(root, reproducibility)
    config = make_config(root, vision, backbone, stats)
    architecture = make_architecture(manifest, config)
    reports = sorted(str(path.relative_to(root)) for path in (root / "meta").glob("cpm*validation*.json"))
    snapshot_names = [name for name in (*UPDATED_FILES, *reports) if (root / name).is_file()]
    configs = {}
    for split in ("train", "val"):
        entry = read_json(root / f"config/{split}.json")
        if entry["noise_on_read"] or entry["reader_kwargs"]["include_negatives"]:
            raise ValueError("Use the clean positive reader with explicit action noise")
        if (
            entry["noise_augmentation"]["cpm_action_input_key"] != "noisy_actions"
            or entry["batching"]["batch_size"] != 64
        ):
            raise ValueError("CPM uses noisy action columns and preserved batch size 64")
        entry["cpm"] = {
            "config_file": "config/cpm.json",
            "config_schema": SCHEMA,
            "model": config["model"],
            "loss": config["loss"],
            "action_input_key": "noisy_actions",
            "flow_time_key": "flow_time",
        }
        configs[f"config/{split}.json"] = entry
    entry_config = read_json(root / "meta/entry_config.json")
    entry_config.update(
        stage="cpm_v2_validation_pending",
        cpm_model=config["model"],
        cpm_loss=config["loss"],
        cpm_config_file="config/cpm.json",
        cpm_config_schema=SCHEMA,
        cpm_network_implemented=True,
        contrastive_loss_implemented=True,
        training_started=False,
    )
    manifest["stage"] = "cpm_v2_validation_pending"
    manifest["implementation_status"].update(
        cpm_network=True, similarity_scores=True, contrastive_loss=True, cpm_training=True, cpm_v2_validation=False
    )
    manifest["cpm"] = architecture
    manifest["training_started"] = False
    values = {
        "config/cpm.json": config,
        "meta/cpm_architecture.json": architecture,
        "meta/manifest.json": manifest,
        "meta/entry_config.json": entry_config,
        **configs,
    }
    staging.mkdir()
    try:
        snapshot_record = None
        if upgrading:
            backup = staging / SNAPSHOT
            for name in snapshot_names:
                destination = backup / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(root / name, destination)
                if sha256(destination) != sha256(root / name):
                    raise ValueError(f"Snapshot verification failed: {name}")
            snapshot_record = {
                "schema": "touchguide_cpm_cnn_v1_snapshot_v1",
                "prepared_at_utc": datetime.now(UTC).isoformat(),
                "source_root": str(root),
                "source_config_schema": current["schema"],
                "files_sha256": {name: sha256(backup / name) for name in snapshot_names},
            }
            write_json(backup / "snapshot_manifest.json", snapshot_record)
        for name, value in values.items():
            write_json(staging / name, value)
        (staging / "README.md").write_text(make_readme((root / "README.md").read_text()))
        reproducibility.update(source_records())
        reproducibility["cpm_visual_backbone"] = {
            "path": str(vision / "backbone_manifest.json"),
            "sha256": sha256(vision / "backbone_manifest.json"),
        }
        backbone_builder = Path(__file__).resolve().with_name("prepare_dinov2_base.py")
        if backbone_builder.is_file():
            reproducibility["cpm_backbone_builder"] = {
                "path": str(backbone_builder),
                "sha256": sha256(backbone_builder),
            }
        reproducibility["cpm_checks"] = {"status": "pending_for_v2_architecture", "parameter_counts": COUNTS}
        reproducibility["cpm_training_checks"] = {
            "status": "pending_for_v2_architecture",
            "formal_training_started": False,
        }
        reproducibility["cpm_model_update_note"] = (
            "Shared frozen DINOv2-Base for RGB and tactile; CNN v1 validation is archived."
        )
        reproducibility["cpm_v2_preparation"] = {
            "schema": SCHEMA,
            "prepared_at_utc": datetime.now(UTC).isoformat(),
            "upgraded_from_cnn_v1": upgrading,
            "legacy_snapshot": SNAPSHOT if upgrading else None,
            "preserved_inputs_sha256": preserved,
            "model_loaded": False,
            "training_started": False,
        }
        for name in reports:
            reproducibility["files_sha256"].pop(name, None)
        if snapshot_record is not None:
            for name, digest in snapshot_record["files_sha256"].items():
                reproducibility["files_sha256"][f"{SNAPSHOT}/{name}"] = digest
            reproducibility["files_sha256"][f"{SNAPSHOT}/snapshot_manifest.json"] = sha256(
                staging / SNAPSHOT / "snapshot_manifest.json"
            )
        for name in UPDATED_FILES:
            if name != "meta/reproducibility.json":
                reproducibility["files_sha256"][name] = sha256(staging / name)
        write_json(staging / "meta/reproducibility.json", reproducibility)
        verify_preserved(root, preserved)
        if upgrading:
            (staging / SNAPSHOT).rename(root / SNAPSHOT)
        for name in UPDATED_FILES:
            destination = root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            (staging / name).replace(destination)
        if upgrading:
            for name in reports:
                (root / name).unlink()
        verify_v2(root, vision, read_json(root / "config/cpm.json"))
    except BaseException:
        print(f"CPM preparation failed; staging retained for inspection: {staging}", flush=True)
        raise
    else:
        shutil.rmtree(staging)
    print(
        json.dumps(
            {
                "status": "upgraded_from_cnn_v1" if upgrading else "prepared_v2",
                "dataset_root": str(root),
                "config": str(root / "config/cpm.json"),
                "vision_directory": str(vision),
                "legacy_snapshot": str(root / SNAPSHOT) if upgrading else None,
                "parameter_counts": COUNTS,
                "model_loaded": False,
                "training_started": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
