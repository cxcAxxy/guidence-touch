"""Audit actual tactile preprocessing and generate an offline interactive report.

All distribution counts use every original 7x5 taxel, excluding padding/resize.
Example pairs are deliberately selected for diversity, not success labels.
"""

# ruff: noqa: PLC0415
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC
from datetime import datetime
import io
import json
from pathlib import Path
import sys

import h5py
import numpy as np

ROOT = Path("/path/to/datasets/touchguidence/augmentation/poker_fingertip_recovery_rules_v1_random10")
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def mapped(force, rms, ref):
    return (0.5 + np.arcsinh(force.astype(np.float64) / rms.reshape(1, 1, 3, 1, 1)) / (2 * np.arcsinh(ref))).astype(
        np.float32
    )


def empty_stats():
    return {
        "count": 0,
        "zeros": np.zeros(3, dtype=np.int64),
        "sum_sq": np.zeros(3),
        "raw_min": np.full(3, np.inf),
        "raw_max": np.full(3, -np.inf),
        "pixel_min": np.full(3, np.inf),
        "pixel_max": np.full(3, -np.inf),
        "outside": np.zeros(3, dtype=np.int64),
        "raw_hist": np.zeros((3, 244), dtype=np.int64),
        "pixel_hist": np.zeros((3, 260), dtype=np.int64),
    }


def scan_episode(job):
    root, episode, rms, ref, raw_edges, pixel_edges = job
    result, candidates = {}, []
    for kind in ["clean", "augmented"]:
        stats = empty_stats()
        path = root / kind / "episodes" / f"episode_{episode:06d}.h5"
        with h5py.File(path, "r") as f:
            split = str(f.attrs["split"])
            count = len(f["tactile_map_raw"])
            peak_value, peak_row = -1.0, 0
            for begin in range(0, count, 2048):
                t = f["tactile_map_raw"][begin : begin + 2048]
                p = mapped(t, rms, ref)
                stats["count"] += len(t) * 175
                for channel in range(3):
                    raw, pixel = t[:, :, channel].ravel(), p[:, :, channel].ravel()
                    stats["zeros"][channel] += np.count_nonzero(raw == 0)
                    stats["sum_sq"][channel] += np.einsum("i,i->", raw.astype(np.float64), raw.astype(np.float64))
                    stats["raw_min"][channel] = min(stats["raw_min"][channel], float(raw.min()))
                    stats["raw_max"][channel] = max(stats["raw_max"][channel], float(raw.max()))
                    stats["pixel_min"][channel] = min(stats["pixel_min"][channel], float(pixel.min()))
                    stats["pixel_max"][channel] = max(stats["pixel_max"][channel], float(pixel.max()))
                    stats["outside"][channel] += np.count_nonzero((pixel < 0) | (pixel > 1))
                    stats["raw_hist"][channel] += np.histogram(raw, bins=raw_edges[channel])[0]
                    stats["pixel_hist"][channel] += np.histogram(pixel, bins=pixel_edges)[0]
                if kind == "augmented":
                    peak = np.max(np.abs(t[:, :, 1]), axis=(1, 2, 3))
                    local = int(np.argmax(peak))
                    if peak[local] > peak_value:
                        peak_value, peak_row = float(peak[local]), begin + local
            if kind == "augmented":
                rng = np.random.default_rng(episode + 261013)
                rows = sorted({*rng.choice(count, min(count, 4), replace=False).tolist(), peak_row})
                with h5py.File(root / "clean/episodes" / path.name, "r") as clean:
                    for row in rows:
                        source = int(f["source_clean_row"][row])
                        a, c = (
                            f["tactile_map_raw"][row],
                            clean["tactile_map_raw"][source],
                        )
                        candidates.append(
                            {
                                "episode": episode,
                                "split": split,
                                "row": row,
                                "clean_row": source,
                                "frame": int(f["frame_index"][row]),
                                "variant": int(f["variant_index"][row]),
                                "clean": c,
                                "augmented": a,
                                "displacement_mm": f["requested_local_components_mm"][row],
                                "perturbed": f["perturbed_fingers"][row],
                            }
                        )
            result[kind] = (split, count, stats)
    return result, candidates


def add(dst, src):
    for key in dst:
        if key.endswith("_min"):
            dst[key] = np.minimum(dst[key], src[key])
        elif key.endswith("_max"):
            dst[key] = np.maximum(dst[key], src[key])
        else:
            dst[key] += src[key]


def image_uri(image):
    from PIL import Image

    stream = io.BytesIO()
    # Display-only conversion. Actual DINO input is NOT clipped or quantized.
    Image.fromarray(np.uint8(np.rint(np.clip(image.transpose(1, 2, 0), 0, 1) * 255))).save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()


def run(root, output, workers):
    config = json.loads((root / "ratio_1_1/config/cpm.json").read_text())["tactile"]
    # Match the actual FP32 buffer, not only the JSON's FP64 literals.
    rms = np.asarray(config["rms"], dtype=np.float32).astype(np.float64)
    ref = float(config["reference_multiple"])
    raw_edges = [
        np.linspace(0, 0.61, 245),
        np.linspace(-0.38, 0.38, 245),
        np.linspace(-0.61, 0.61, 245),
    ]
    pixel_edges = np.linspace(-0.15, 1.15, 261)
    exact = {kind: {split: empty_stats() for split in ["all", "train", "val"]} for kind in ["clean", "augmented"]}
    pairs = {kind: dict.fromkeys(["all", "train", "val"], 0) for kind in exact}
    candidates = []
    episodes = sorted(int(p.stem.split("_")[-1]) for p in (root / "clean/episodes").glob("episode_*.h5"))
    jobs = [(root, e, rms, ref, raw_edges, pixel_edges) for e in episodes]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (result, cases) in enumerate(pool.map(scan_episode, jobs), 1):
            candidates.extend(cases)
            for kind, (split, count, stats) in result.items():
                for target in ["all", split]:
                    add(exact[kind][target], stats)
                    pairs[kind][target] += count
            if i % 20 == 0 or i == len(episodes):
                print(f"Scanned {i}/{len(episodes)} episodes", flush=True)
    for splits in exact.values():
        for stats in splits.values():
            for channel in range(3):
                if (
                    stats["raw_hist"][channel].sum() != stats["count"]
                    or stats["pixel_hist"][channel].sum() != stats["count"]
                ):
                    raise ValueError("Histogram bounds missed data; expand plotting bounds")
            stats["rms"] = np.sqrt(stats["sum_sq"] / stats["count"])
            stats["outside_fraction"] = stats["outside"] / stats["count"]
    train_clean = exact["clean"]["train"]
    train_aug = exact["augmented"]["train"]
    joint_rms = np.sqrt((train_clean["sum_sq"] + train_aug["sum_sq"]) / (train_clean["count"] + train_aug["count"]))

    # Deliberately select distinct paired examples of six types.
    for case in candidates:
        c, a = case["clean"], case["augmented"]
        pc, pa = mapped(c[None], rms, ref)[0], mapped(a[None], rms, ref)[0]
        fc, fa = c[:, 0].sum((1, 2)), a[:, 0].sum((1, 2))
        case["metrics"] = [
            float(np.max(np.maximum(-pa, pa - 1))),
            float(np.max(np.where(fc > 0.001, fc - fa, 0) * (fa <= 0.001))),
            float(np.max(fa - fc)),
            float(np.max(np.abs(a[:, 1:] - c[:, 1:]))),
            float(np.sqrt(np.mean((pa - pc) ** 2))),
            float(np.max(fc - fa)),
        ]
    chosen, used = [], set()
    titles = [
        "参考范围外的切向极端值",
        "有接触 → 无接触",
        "法向力增大的配对",
        "切向变化明显的配对",
        "映射变化较小的配对",
        "法向力减小的配对",
    ]
    for criterion, title in enumerate(titles):
        pool = [case for case in candidates if (case["episode"], case["row"]) not in used]
        if criterion == 4:
            pool = [case for case in pool if case["metrics"][4] > 1e-5]
            picked = min(pool, key=lambda x: x["metrics"][4])
        else:
            picked = max(pool, key=lambda x: x["metrics"][criterion])
        used.add((picked["episode"], picked["row"]))
        picked["title"] = title
        picked["default_finger"] = int(np.argmax(np.mean((picked["augmented"] - picked["clean"]) ** 2, axis=(1, 2, 3))))
        chosen.append(picked)

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    import torch

    from openpi.touchguide.tactile_dino import TactileDinoPreprocessor

    torch.set_num_threads(2)
    prep = TactileDinoPreprocessor(rms, reference_multiple=ref)
    raw = np.stack([case[k] for case in chosen for k in ["clean", "augmented"]])
    with torch.no_grad():
        pixels = prep.encode_image(torch.from_numpy(raw)).numpy()
        z = prep(torch.from_numpy(raw)).numpy()
        zero = prep.encode_image(torch.zeros(1, 5, 3, 7, 5)).numpy()
        small = torch.from_numpy(mapped(raw, rms, ref)).reshape(-1, 3, 7, 5)
        expected = torch.nn.functional.pad(
            torch.nn.functional.interpolate(small, size=(224, 160), mode="bilinear", align_corners=False),
            (32, 32, 0, 0),
            value=0.5,
        ).numpy()
    np.testing.assert_allclose(pixels, expected, atol=1e-6, rtol=0)
    np.testing.assert_array_equal(z, (pixels - MEAN[None, :, None, None]) / STD[None, :, None, None])
    assert np.all(zero == 0.5)
    assert np.isfinite(z).all()
    assert pixels.min() < 0 or pixels.max() > 1, "Extreme example should verify no clipping"
    for i, case in enumerate(chosen):
        case["images"] = {
            kind: [image_uri(pixels[(2 * i + j) * 5 + f]) for f in range(5)]
            for j, kind in enumerate(["clean", "augmented"])
        }
        case["dino_range"] = {
            kind: [
                [
                    float(z[(2 * i + j) * 5 + f, c].min()),
                    float(z[(2 * i + j) * 5 + f, c].max()),
                ]
                for f in range(5)
                for c in range(3)
            ]
            for j, kind in enumerate(["clean", "augmented"])
        }
    report = {
        "status": "passed",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "data_root": str(root),
        "scope": "all 200 episode files; exact counts on original taxels, no resized pixels/padding",
        "episodes": len(episodes),
        "pairs": pairs,
        "rms": rms,
        "reference_multiple": ref,
        "raw_edges": raw_edges,
        "pixel_edges": pixel_edges,
        "exact": exact,
        "new_clean_train_rms": train_clean["rms"],
        "joint_train_rms": joint_rms,
        "cases": chosen,
        "verification": {
            "production_preprocessor": True,
            "numpy_to_production_max_error": float(np.max(np.abs(pixels - expected))),
            "zero_pixel": 0.5,
            "finite": True,
            "outliers_not_clipped": True,
        },
        "example_selection": "six diverse examples selected from 4 random augmented rows and maximum tangent0 magnitude per episode; not representative sampling",
    }

    def convert(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.generic):
            return o.item()
        raise TypeError(type(o).__name__)

    output.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, ensure_ascii=False, default=convert, allow_nan=False)
    (output / "report.json").write_text(payload, encoding="utf-8")
    template = Path(__file__).with_name("tactile_normalization_template.html").read_text()
    (output / "index.html").write_text(
        template.replace("__DATA_JSON__", payload.replace("</", "<\\/")),
        encoding="utf-8",
    )
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for channel, name in enumerate(["Normal", "Tangent 0", "Tangent 1"]):
        ax = axes[0, channel]
        force = np.linspace(raw_edges[channel][0], raw_edges[channel][-1], 600)
        ax.plot(force, 0.5 + np.arcsinh(force / rms[channel]) / (2 * np.arcsinh(ref)))
        ax.axhspan(0, 1, color="#e4f2e9", zorder=-1)
        ax.axhline(0.5, color="gray", ls=":")
        ax.set(
            xlabel="Force (N)",
            ylabel="Mapped pixel (unclipped)",
            title=f"{name}: RMS={rms[channel]:.6f} N",
        )
        ax = axes[1, channel]
        for kind, color in [("clean", "#2563eb"), ("augmented", "#ea580c")]:
            s = exact[kind]["all"]
            hist = s["pixel_hist"][channel].copy()
            hist[np.searchsorted(pixel_edges, 0.5, side="right") - 1] -= s["zeros"][channel]
            ax.plot(
                (pixel_edges[1:] + pixel_edges[:-1]) / 2,
                hist / (s["count"] - s["zeros"][channel]),
                color=color,
                label=kind,
            )
        ax.set(
            xlabel="Mapped pixel",
            ylabel="Fraction / bin (zero force excluded)",
            yscale="log",
            ylim=(1e-8, 1),
        )
        ax.axvline(0, color="gray", ls="--")
        ax.axvline(1, color="gray", ls="--")
        ax.legend()
    fig.suptitle("Tactile normalization: exact full dataset distributions; no clipping in model")
    fig.savefig(output / "overview.png", dpi=180)
    plt.close(fig)
    print(
        json.dumps(
            {
                "output": str(output),
                "pairs": pairs,
                "outside": {k: exact[k]["all"]["outside_fraction"].tolist() for k in exact},
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    run(
        args.data_root,
        args.output or args.data_root / "validation/tactile_normalization_v1",
        args.workers,
    )
