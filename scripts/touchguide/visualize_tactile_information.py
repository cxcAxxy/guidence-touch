"""Render offline tactile physical-information audit from its saved statistics."""

# ruff: noqa: PLC0415
# Select a noninteractive matplotlib backend before importing pyplot.
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import numpy as np


def render(output):
    mpl.use("Agg")
    import matplotlib.pyplot as plt

    report = json.loads((output / "report.json").read_text())
    methods = ["raw", "mapped", "dino"] + (["dino_bf16"] if "dino_bf16" in report["probes"] else [])
    colors = ["#2563eb", "#16a085", "#e87924", "#8b5cf6"][: len(methods)]
    fig, ax = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    contact = [report["probes"][m]["contact"]["balanced_accuracy"] for m in methods]
    ax[0, 0].bar(methods, contact, color=colors)
    ax[0, 0].axhline(0.5, ls="--", color="gray", label="constant predictor: 0.5")
    ax[0, 0].set(title="Contact balanced accuracy (held-out episodes)", ylim=(0, 1.05))
    for a, targets in [(ax[0, 1], report["targets"][:3]), (ax[0, 2], report["targets"][3:6])]:
        for j, (m, c) in enumerate(zip(methods, colors, strict=True)):
            a.bar(
                np.arange(3) + (j - (len(methods) - 1) / 2) * 0.18,
                [report["probes"][m]["targets"][t]["r2"] for t in targets],
                width=0.18,
                color=c,
                label=m,
            )
        a.axhline(0, ls="--", color="gray")
        a.set_xticks(np.arange(3), [t.replace("_force_N", "").replace("_taxel", "") for t in targets], fontsize=10)
        a.set(
            title="Force readout R2" if a is ax[0, 1] else "Spatial readout R2",
            ylabel="R2 (1=perfect, 0=mean baseline)",
        )
        a.legend()
    examples = report["examples"]
    dino_method = "dino_bf16" if "dino_bf16" in report["probes"] else "dino"
    for j in range(3):
        truth = np.array([e["truth"][j] for e in examples])
        pred = np.array([e["prediction"][dino_method][j] for e in examples])
        a = ax[1, j]
        a.scatter(truth, pred, s=12, alpha=0.4, color=colors[methods.index(dino_method)])
        lo = min(truth.min(), pred.min())
        hi = max(truth.max(), pred.max())
        a.plot([lo, hi], [lo, hi], color="gray", ls="--")
        mae = report["probes"][dino_method]["targets"][report["targets"][j]]["mae"]
        a.set(
            xlabel="True summed force (N)",
            ylabel=f"{dino_method} linear prediction (N)",
            title=f"{report['targets'][j]}: MAE={mae:.4f} N",
        )
    fig.suptitle(
        f"Tactile information audit: {report['finger_images']} sampled finger images; {report['validation_fingers']} held-out validation images\nSmall linear readouts only; no OA scorer or DINO fine-tuning",
        fontsize=14,
    )
    fig.savefig(output / "overview.png", dpi=160)
    plt.close(fig)
    template = Path(__file__).with_name("tactile_information_template.html").read_text()
    payload = json.dumps(report, ensure_ascii=False, allow_nan=False).replace("</", "<\\/")
    (output / "index.html").write_text(template.replace("__DATA_JSON__", payload), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    render(args.output)
