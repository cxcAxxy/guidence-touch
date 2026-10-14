"""Calibrate ambiguous negatives and prepare three shared recovery training views."""

# ruff: noqa: RUF001
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

ROOT = Path("/path/to/datasets/touchguidence/augmentation/poker_fingertip_recovery_rules_v1_random10")
BASE = Path("/path/to/datasets/sim/poker-draw/pi05_trex/touchguide/cpm_datasets/poker_0920_200_touchguide_v1")


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    if path.exists() and json.loads(path.read_text()) != json.loads(text):
        raise ValueError(f"Existing prepared contract differs; refusing silent replacement: {path}")
    if not path.exists():
        path.write_text(text)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def calibrate(root, profile):
    manifest = json.loads((root / "clean/meta/manifest.json").read_text())
    low, high = [np.asarray(profile["actions"][k], np.float32) for k in ("low", "high")]
    values = []
    sample_origins = 0
    for i, eid in enumerate(manifest["splits"]["train"]["episode_indices"]):
        rng = np.random.default_rng(np.random.SeedSequence([42, eid, 912]))
        with h5py.File(root / "augmented/episodes" / f"episode_{eid:06d}.h5") as f:
            n = len(f["state_raw"]) // 10
            ids = np.sort(rng.choice(n, min(n, 32), replace=False))
            rows = (ids[:, None] * 10 + np.arange(10)).ravel()
            absolute = f["actions_absolute"][rows].reshape(len(ids), 10, 30, 27)
            state = f["state_raw"][rows].reshape(len(ids), 10, 27)
            absolute[..., :7] -= state[:, :, None, :7]
            actions = 2 * (absolute - low) / (high - low) - 1
        distances = np.zeros((len(ids), 10, 10), np.float32)
        for window in (5, 30):
            for first, last in (
                (0, 7),
                (7, 11),
                (11, 15),
                (15, 19),
                (19, 23),
                (23, 27),
            ):
                x = actions[:, :, :window, first:last]
                diff = x[:, :, None] - x[:, None, :]
                d = np.sqrt(np.mean(np.square(diff), axis=(-1, -2)))
                distances = np.maximum(distances, d)
        upper = np.triu_indices(10, 1)
        values.append(distances[:, upper[0], upper[1]].ravel())
        sample_origins += len(ids)
        if (i + 1) % 30 == 0:
            print(f"Pair-distance calibration {i + 1}/180 training episodes", flush=True)
    distances = np.concatenate(values)
    positive = distances[distances > 1e-7]
    if len(positive) == 0:
        raise ValueError("No distinguishable recovery actions for calibration")
    # A conservative empirical policy: only the lowest .1% of nonzero distances
    # are ignored; full weight is reached by the first percentile. These are
    # matching heuristics, not evidence of physically equivalent actions.
    ignore, full = [float(x) for x in np.percentile(positive, [0.1, 1])]
    if full <= ignore:
        full = float(np.nextafter(np.float32(ignore), np.float32(np.inf)))
    policy = {
        "schema": "touchguide_negative_weights_v1",
        "ignore_distance": ignore,
        "full_weight_distance": full,
        "distance": "max_group_RMS_over_windows_5_and_30_in_clean_CPM_coordinates",
        "groups": ["arm7", "thumb4", "index4", "middle4", "ring4", "pinky4"],
        "weight": "clamp((distance-ignore)/(full-ignore),0,1); diagonal=1",
        "labels": "diagonal matching only; no extra positives; no physical success claims",
        "calibration": {
            "split": "train_only",
            "sampled_origins": sample_origins,
            "comparisons": len(distances),
            "samples_per_episode": 32,
            "percentiles": [0.1, 1],
            "seed": 42,
        },
        "applies_to": "all off-diagonal candidate pairs, including same-origin and nearby-frame pairs",
        "computed_before_flow_noise": True,
        "fixed_for_ratios": ["1:1", "1:5", "1:10"],
    }
    quantiles = {str(p): float(np.percentile(distances, p)) for p in (0, 0.1, 1, 5, 50, 95, 99, 100)}
    weights = np.clip((distances - ignore) / (full - ignore), 0, 1)
    review = {
        "policy": policy,
        "distance_quantiles": quantiles,
        "exact_duplicates_in_sample": int((distances == 0).sum()),
        "ignored_fraction_in_same_origin_sample": float((weights == 0).mean()),
        "downweighted_fraction_in_same_origin_sample": float(((weights > 0) & (weights < 1)).mean()),
        "same_origin_mean_negative_weight": float(weights.mean()),
    }
    out = root / "validation/negative_pairing_v1"
    save(out / "policy.json", policy)
    save(out / "report.json", review)
    import matplotlib as mpl  # noqa: PLC0415

    mpl.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax in axes:
        ax.hist(distances, bins=100, color="#2563eb", alpha=0.7)
        ax.axvline(ignore, color="#dc2626", label=f"Ignore <= {ignore:.6g}")
        ax.axvline(full, color="#ea580c", label=f"Full weight >= {full:.6g}")
        ax.set_xlabel("Clean action distance (CPM coordinates)")
        ax.set_ylabel("Comparisons")
    axes[1].set_xlim(0, float(np.percentile(distances, 5)))
    axes[1].legend(fontsize=8)
    axes[0].set_title("Sampled same-origin recovery pairs")
    axes[1].set_title("Closest 5%: conservative thresholds")
    fig.tight_layout()
    fig.savefig(out / "distance_distribution.png", dpi=150)
    plt.close(fig)
    (out / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>负配对距离校准</title><main style="max-width:1100px;margin:25px auto;font:16px system-ui"><h1>负配对距离校准</h1><p>只使用训练集。轨迹几乎重合的候选降权，不把同一起点的其他动作全部排除。阈值取非零距离的最低 0.1% 和 1% 分位数，是保守的匹配规则，不代表物理上已验证动作等价。</p><img style="width:100%" src="distance_distribution.png"><pre>'
        + json.dumps(review, ensure_ascii=False, indent=2)
        + "</pre></main>"
    )
    return policy


def run(args):
    root = args.data_root.resolve()
    base = args.base_entry.resolve()
    profile = json.loads((root / "validation/modality_normalization_v1/cpm_normalization_v1.json").read_text())
    audit = json.loads((root / "validation/modality_normalization_v1/report.json").read_text())
    if audit["status"] != "passed" or any(
        count for k in audit["new_train_and_val_outside_counts"].values() for m in k.values() for count in m.values()
    ):
        raise ValueError("New normalization has not passed full validation")
    profile["status"] = "active_for_recovery_state_cpm_v1"
    policy = calibrate(root, profile)
    entries = []
    for ratio in (1, 5, 10):
        view = root / f"ratio_1_{ratio}"
        save(view / "meta/cpm_normalization_v1.json", profile)
        save(view / "meta/negative_policy.json", policy)
        cpm = deepcopy(json.loads((base / "config/cpm.json").read_text()))
        cpm["model"] = "openpi.touchguide.recovery_model.RecoveryStateCPM.from_config"
        cpm["observation_tokens"] = 120
        cpm["recovery"] = {
            "schema": "touchguide_recovery_state_cpm_v1",
            "model_variant": "recovery-state-cpm",
            "pool_grid": [4, 4],
            "vision_tokens": 34,
            "tactile_tokens": 85,
            "state_tokens": 7,
            "fusion_cls_tokens": 1,
            "total_tokens": 127,
            "state_layout": "global27 + arm7 + five finger4 tokens; shared touch/state finger identity",
            "action_input_coordinates": "original_pi05; affine-converted inside ActionEncoder after flow noise",
            "normalization": {
                "file": "meta/cpm_normalization_v1.json",
                "sha256": sha(view / "meta/cpm_normalization_v1.json"),
            },
            "negative_policy": {
                "file": "meta/negative_policy.json",
                "sha256": sha(view / "meta/negative_policy.json"),
            },
            "clean_augmented_ratio": f"1:{ratio}",
        }
        cpm["vision"]["tokens_per_view"] = 17
        cpm["tactile"]["tokens_per_finger"] = 17
        cpm["loss_formula"] = (
            "mean_of_row_and_column(logsumexp(logits+log(negative_weights))-diagonal_logits); diagonal_weight=1"
        )
        cpm["parameter_counts"] = {"trainable": 123342081, "frozen": 86580480, "total": 209922561}
        cpm["state"] = {
            "input": "raw27_or_original_pi05_normalized27",
            "normalization": "train_only_union_clean_and_all_recovery_bounds_with_margin_no_clip",
            "tokens": 7,
            "finger_identity_embedding": "shared_with_tactile",
        }
        cpm["vision"]["spatial_pooling"] = "keep_CLS_and_average_pool_16x16_patches_to_4x4"
        cpm["tactile"]["spatial_pooling"] = cpm["vision"]["spatial_pooling"]
        cpm["action"]["renormalize"] = True
        cpm["action"]["coordinate_conversion"] = "original_pi05_noisy_or_clean_to_CPM_affine_inside_encoder"
        cpm["metrics"] = [
            "loss",
            "loss_obs_to_action",
            "loss_action_to_obs",
            "accuracy_obs_to_action",
            "accuracy_action_to_obs",
            "positive_score",
            "negative_score",
            "score_gap",
            "temperature",
            "unweighted_loss",
            "negative_weight_mean",
            "ignored_negative_fraction",
            "downweighted_negative_fraction",
            "effective_negatives_per_observation",
        ]
        save(view / "config/cpm.json", cpm)
        for split in ("train", "val"):
            cfg = deepcopy(json.loads((base / f"config/{split}.json").read_text()))
            cfg["reader_kwargs"]["root"] = str(view)
            cfg["batching"]["batch_size"] = 128
            cfg["batching"]["include_auxiliary"] = True
            cfg["model_input_keys"].append("state_raw")
            cfg["augmented_model_input_keys"].append("state_raw")
            cfg["cpm"]["model"] = cpm["model"]
            cfg["noise_augmentation"]["coordinate_conversion"] = "original_pi05_noise_then_CPM_affine_inside_model"
            if split == "val":
                cfg["batching"]["source_index_sha256"] = sha(view / "index/val.npz")
                with np.load(view / "index/val.npz") as f:
                    n = len(f["episode_index"])
                ids = np.random.default_rng(42).permutation(n)
                offsets = [*range(0, n, 128), n]
                if offsets[-1] - offsets[-2] == 1:
                    offsets[-2] -= 1
                saved = view / "index/val_batches.npz"
                if not saved.exists():
                    np.savez_compressed(
                        saved,
                        indices=ids,
                        batch_offsets=np.array(offsets),
                        batch_size=np.array(128),
                        seed=np.array(42),
                    )
            save(view / f"config/{split}.json", cfg)
        save(
            view / "meta/entry_config.json",
            {
                "stage": "recovery_state_cpm_training_ready",
                "model_variant": "recovery-state-cpm",
                "training_started": False,
                "shared_data": True,
                "ratio": f"1:{ratio}",
                "normalization": "meta/cpm_normalization_v1.json",
                "negative_policy": "meta/negative_policy.json",
                "wandb_project": "poker-draw-touchguide-cpm",
            },
        )
        entries.append(str(view))
    save(
        root / "validation/recovery_training_entries.json",
        {
            "status": "prepared",
            "entries": entries,
            "shared_data": True,
            "training_started": False,
        },
    )
    print(
        json.dumps(
            {"entries": entries, "negative_policy": policy},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=ROOT)
    p.add_argument("--base-entry", type=Path, default=BASE)
    run(p.parse_args())
