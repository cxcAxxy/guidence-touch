"""Sampled physical-information audit of the actual tactile preprocessing and DINO.

Readouts diagnose retained information, not task feasibility or guidance quality.
Episode splits are respected; probe hyperparameters use held-out TRAIN episodes.
"""

# ruff: noqa: PLC0415
# Heavy imports are deferred until the runtime source path is configured.
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import h5py
import numpy as np
import torch
from torch.nn import functional

ROOT = Path("/path/to/datasets/touchguidence/augmentation/poker_fingertip_recovery_rules_v1_random10")
TARGETS = [
    "normal_force_N",
    "tangent0_force_N",
    "tangent1_force_N",
    "centroid_row_taxel",
    "centroid_col_taxel",
    "effective_taxels",
    "contact",
]


def physical(t):
    normal = t[:, 0]
    force = t.sum((2, 3))
    den = normal.sum((1, 2))
    contact = den > 0.001
    row = (normal * np.arange(7)[None, :, None]).sum((1, 2)) / np.maximum(den, 1e-20)
    col = (normal * np.arange(5)[None, None, :]).sum((1, 2)) / np.maximum(den, 1e-20)
    effective = den**2 / np.maximum((normal**2).sum((1, 2)), 1e-30)
    return np.column_stack([force, row, col, effective, contact]).astype(np.float64)


def quantiles(x):
    x = np.asarray(x)
    return (
        {"n": len(x), "median": float(np.median(x)), "p95": float(np.quantile(x, 0.95)), "max": float(x.max())}
        if len(x)
        else {"n": 0}
    )


def sample(root):
    raw, metadata = [], []
    for i, path in enumerate(sorted((root / "augmented/episodes").glob("episode_*.h5"))):
        with h5py.File(path, "r") as aug, h5py.File(root / "clean/episodes" / path.name, "r") as clean:
            e = int(aug.attrs["episode_index"])
            rng = np.random.default_rng(261014 + e)
            origins = rng.choice(len(clean["tactile_map_raw"]), min(2, len(clean["tactile_map_raw"])), replace=False)
            for origin in origins:
                variant = int(rng.integers(0, 10))
                row = int(origin) * 10 + variant
                assert int(aug["source_clean_row"][row]) == origin
                for kind, f, index in [("clean", clean, int(origin)), ("augmented", aug, row)]:
                    raw.append(f["tactile_map_raw"][index])
                    metadata.extend(
                        [
                            {
                                "episode": e,
                                "split": str(aug.attrs["split"]),
                                "kind": kind,
                                "finger": finger,
                                "clean_row": int(origin),
                                "augmented_row": row,
                                "variant": variant,
                            }
                            for finger in range(5)
                        ]
                    )
        if (i + 1) % 50 == 0:
            print(f"Sampled {i + 1}/200 episodes", flush=True)
    return np.stack(raw), metadata


def standardized_fit(x, mask):
    mean, scale = x[mask].mean(0), x[mask].std(0)
    return mean, np.maximum(scale, 1e-6)


def probe(x, y, fit, tune, test, *, pca=False):
    device = "cuda"
    mean, scale = standardized_fit(x, fit)
    z = torch.from_numpy(((x - mean) / scale).astype(np.float32)).to(device)
    pca_size = None
    if pca:
        torch.manual_seed(20261014)
        _, _, v = torch.pca_lowrank(z[fit], q=144, center=False, niter=3)
        v = v[:, :128]
        z = z @ v
        pca_size = 128
    z = torch.cat([z, torch.ones(len(z), 1, device=device)], 1)
    candidates = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    predictions = []
    selected = []
    for j in range(y.shape[1]):
        valid = y[:, 6] > 0.5 if j in [3, 4] else np.ones(len(y), dtype=bool)
        train_mask, tune_mask = fit & valid, tune & valid
        yymean, yystd = float(y[train_mask, j].mean()), max(float(y[train_mask, j].std()), 1e-8)
        yt = torch.from_numpy(((y[:, j] - yymean) / yystd).astype(np.float32)).to(device)
        zfit = z[train_mask]
        gram = zfit.T @ zfit
        rhs = zfit.T @ yt[train_mask]
        eye = torch.eye(z.shape[1], device=device)
        eye[-1, -1] = 0
        errors = []
        coefs = []
        for alpha in candidates:
            coef = torch.linalg.solve(gram + alpha * eye, rhs)
            error = ((z[tune_mask] @ coef - yt[tune_mask]) ** 2).mean().item()
            errors.append(error)
            coefs.append(coef)
        best = int(np.argmin(errors))
        selected.append(candidates[best])
        prediction = (z @ coefs[best]).cpu().numpy() * yystd + yymean
        predictions.append(prediction)
    prediction = np.stack(predictions, 1)
    result = {"pca_dimensions": pca_size, "selected_ridge_alpha": selected, "targets": {}}
    for j, name in enumerate(TARGETS[:6]):
        mask = test & ((y[:, 6] > 0.5) if j in [3, 4] else True)
        truth, pred = y[mask, j], prediction[mask, j]
        np.mean((truth - pred) ** 2)
        sst = np.sum((truth - truth.mean()) ** 2)
        result["targets"][name] = {
            "n": len(truth),
            "mae": float(np.mean(np.abs(truth - pred))),
            "r2": float(1 - np.sum((truth - pred) ** 2) / sst),
            "mean_baseline_mae": float(np.mean(np.abs(truth - y[fit, j].mean()))),
        }
    labels = y[test, 6] > 0.5
    decisions = prediction[test, 6] > 0.5
    tpr = float(decisions[labels].mean())
    tnr = float((~decisions[~labels]).mean())
    result["contact"] = {
        "n": len(labels),
        "positive_count": int(labels.sum()),
        "negative_count": int((~labels).sum()),
        "accuracy": float((labels == decisions).mean()),
        "balanced_accuracy": (tpr + tnr) / 2,
        "recall": tpr,
        "specificity": tnr,
        "majority_baseline_accuracy": float(max(labels.mean(), 1 - labels.mean())),
    }
    return result, prediction


def run(root, output):
    started = time.monotonic()
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("DINO information audit requires the available local GPU")
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from openpi.touchguide.model import FrozenDinov2
    from openpi.touchguide.recovery_model import RecoveryStateCPM
    from openpi.touchguide.tactile_dino import TactileDinoPreprocessor

    config = json.loads((root / "ratio_1_1/config/cpm.json").read_text())
    prep = TactileDinoPreprocessor(
        config["tactile"]["rms"], reference_multiple=config["tactile"]["reference_multiple"]
    ).cuda()
    model = FrozenDinov2(config["vision"]["directory"], batch_size=32).cuda().eval()
    raw, meta = sample(root)
    touch = raw.reshape(-1, 3, 7, 5)
    targets = physical(touch.astype(np.float64))
    rms = prep.tactile_rms.flatten().cpu().numpy().astype(np.float64)
    denominator = 2 * np.arcsinh(config["tactile"]["reference_multiple"])
    pixels = (0.5 + np.arcsinh(raw.astype(np.float64) / rms[None, None, :, None, None]) / denominator).astype(
        np.float32
    )
    restored = rms[None, None, :, None, None] * np.sinh((pixels.astype(np.float64) - 0.5) * denominator)
    tiny = (pixels == 0.5) & (raw != 0)
    inverse = {}
    for c, name in enumerate(["normal", "tangent0", "tangent1"]):
        a = raw[:, :, c]
        b = restored[:, :, c]
        inverse[name] = {
            "max_taxel_abs_error_N": float(np.max(np.abs(a - b))),
            "mean_taxel_abs_error_N": float(np.mean(np.abs(a - b))),
            "nonzero_rounded_to_neutral_count": int(tiny[:, :, c].sum()),
            "rounded_force_fraction": float(np.abs(a[tiny[:, :, c]]).sum() / np.abs(a).sum()),
        }
    exact_targets = physical(restored.reshape(-1, 3, 7, 5))
    features = []
    readback = []
    for begin in range(0, len(raw), 16):
        with torch.no_grad():
            t = torch.from_numpy(raw[begin : begin + 16]).cuda()
            images = prep.encode_image(t)
            # Diagnostic only: crop padding, then read the original taxel centers.
            down = (
                functional.interpolate(images[:, :, :, 32:-32], size=(7, 5), mode="bilinear", align_corners=False)
                .cpu()
                .numpy()
            )
            back = rms[None, :, None, None] * np.sinh((down.astype(np.float64) - 0.5) * denominator)
            readback.append(back)
            z = (images - prep.imagenet_mean) / prep.imagenet_std
            f = RecoveryStateCPM.pool_dino_tokens(model.forward_pixels(z)).float().cpu().numpy()
            features.append(f.reshape(len(f), -1))
        if (begin + 16) % 160 == 0 or begin + 16 >= len(raw):
            print(f"Encoded {min(begin + 16, len(raw))}/{len(raw)} five-finger observations", flush=True)
    features = np.concatenate(features)
    back = np.concatenate(readback)
    back_targets = physical(back)
    contact = targets[:, 6] > 0.5
    after_resize = {
        "contact_agreement": float(np.mean((back_targets[:, 6] > 0.5) == contact)),
        "normal_force_relative_error": quantiles(
            np.abs(back_targets[contact, 0] - targets[contact, 0]) / targets[contact, 0]
        ),
        "centroid_row_abs_error_taxel": quantiles(np.abs(back_targets[contact, 3] - targets[contact, 3])),
        "centroid_col_abs_error_taxel": quantiles(np.abs(back_targets[contact, 4] - targets[contact, 4])),
    }
    print("Fitting small frozen-feature linear readouts (no OA scorer training)", flush=True)
    episodes = np.array([m["episode"] for m in meta])
    splits = np.array([m["split"] for m in meta])
    train_episodes = np.unique(episodes[splits == "train"])
    tune_episodes = train_episodes[::18]
    fit = (splits == "train") & ~np.isin(episodes, tune_episodes)
    tune = (splits == "train") & np.isin(episodes, tune_episodes)
    test = splits == "val"
    probes = {}
    predictions = {}
    for name, x, pca in [
        ("raw", touch.reshape(len(touch), -1), False),
        ("mapped", pixels.reshape(len(touch), -1), False),
        ("dino", features, True),
    ]:
        print("Readout:", name, flush=True)
        result, prediction = probe(x, targets, fit, tune, test, pca=pca)
        probes[name] = result
        predictions[name] = prediction
    # Clean/aug pairs are adjacent five-finger observations, 10 rows per origin.
    a, b = features.reshape(-1, 2, 5, features.shape[-1])[:, 0], features.reshape(-1, 2, 5, features.shape[-1])[:, 1]
    (a * b).sum(-1) / np.maximum(np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1), 1e-12)
    rel = np.linalg.norm(a - b, axis=-1) / np.maximum(np.linalg.norm(a, axis=-1), 1e-12)
    raw_pair = touch.reshape(-1, 2, 5, 3, 7, 5)
    changed = np.max(np.abs(raw_pair[:, 0] - raw_pair[:, 1]), axis=(2, 3, 4)) > 0
    fn = raw_pair[:, :, :, 0].sum((3, 4))
    release = (fn[:, 0] > 0.001) & (fn[:, 1] <= 0.001)
    pair_stats = {
        "changed_fingers": int(changed.sum()),
        "unchanged_fingers": int((~changed).sum()),
        "changed_relative_feature_distance": quantiles(rel[changed]),
        "unchanged_max_abs_feature_error": float(np.max(np.abs(a - b)[~changed])),
        "released_fingers": int(release.sum()),
        "released_relative_feature_distance": quantiles(rel[release]),
    }
    # Validation predictions plotted with separate selectors by source and finger.
    selected = np.flatnonzero(test)
    examples = [
        {**meta[i], "truth": targets[i].tolist(), "prediction": {k: v[i].tolist() for k, v in predictions.items()}}
        for i in selected
    ]
    report = {
        "status": "complete",
        "data_root": str(root),
        "scope": "2 random contact origins per episode, each with clean + one random augmentation; five fingers per observation; fixed seed 261014",
        "episodes": len(np.unique(episodes)),
        "observations": len(raw),
        "finger_images": len(meta),
        "fit_fingers": int(fit.sum()),
        "tune_fingers": int(tune.sum()),
        "validation_fingers": int(test.sum()),
        "fit_episodes": len(np.unique(episodes[fit])),
        "tune_train_episodes": tune_episodes.tolist(),
        "validation_episodes": np.unique(episodes[test]).tolist(),
        "rms_N": rms.tolist(),
        "reference_multiple": config["tactile"]["reference_multiple"],
        "precision": "FP32 DINO, production asinh FP64->FP32; inference only",
        "inverse_mapping": inverse,
        "inverse_contact_agreement": float(np.mean(exact_targets[:, 6] == targets[:, 6])),
        "resize_readback": after_resize,
        "probes": probes,
        "paired_feature_change": pair_stats,
        "targets": TARGETS,
        "examples": examples,
        "definitions": {
            "contact": "sum(normal)>0.001 N",
            "centroid": "force-weighted row 0..6 / col 0..4; evaluated only on contacting fingers",
            "effective_taxels": "sum(normal)^2 / sum(normal^2), zero when no normal force",
            "readout": "standardize on fit episodes; DINO train-only PCA128; ridge alpha chosen separately per target from train tuning episodes; no validation fitting; contact uses fixed prediction threshold 0.5",
            "limits": "Linear readout tests recoverable information in this sampled distribution; a poor readout does not prove information is absent. No physical-success or OA-guidance claim.",
        },
        "seconds": time.monotonic() - started,
    }
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "sampled_features.npz",
        raw=touch,
        pixels=pixels.reshape(-1, 3, 7, 5),
        dino=features.astype(np.float16),
        targets=targets,
        fit=fit,
        tune=tune,
        test=test,
    )
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    # The HTML generation can also run separately without repeating DINO.
    from visualize_tactile_information import render

    render(output)
    print(
        json.dumps(
            {
                "output": str(output),
                "probe_contact": {k: v["contact"]["balanced_accuracy"] for k, v in probes.items()},
                "dino_forces": {k: v for k, v in probes["dino"]["targets"].items() if "force" in k},
                "seconds": report["seconds"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run(args.data_root, args.output or args.data_root / "validation/tactile_information_v1")
