"""Add BF16 frozen-DINO diagnostics using an existing sampled audit and split."""

# ruff: noqa: E402
# The repository source and adjacent analysis script must be importable first.
import json
from pathlib import Path
import sys

import numpy as np
import torch

repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo / "src"))
sys.path.insert(0, str(repo / "scripts/touchguide"))
from analyze_tactile_information import probe
from visualize_tactile_information import render

from openpi.touchguide.model import FrozenDinov2
from openpi.touchguide.recovery_model import RecoveryStateCPM
from openpi.touchguide.tactile_dino import TactileDinoPreprocessor


def run(output):
    torch.set_num_threads(4)
    d = json.loads((output / "report.json").read_text())
    root = Path(d["data_root"])
    cfg = json.loads((root / "ratio_1_1/config/cpm.json").read_text())
    cache = np.load(output / "sampled_features.npz")
    raw = cache["raw"].reshape(-1, 5, 3, 7, 5)
    prep = TactileDinoPreprocessor(
        cfg["tactile"]["rms"], reference_multiple=cfg["tactile"]["reference_multiple"]
    ).cuda()
    model = FrozenDinov2(cfg["vision"]["directory"], batch_size=32).cuda().eval()
    features = []
    for begin in range(0, len(raw), 16):
        with torch.no_grad():
            pixels = prep(torch.from_numpy(raw[begin : begin + 16]).cuda())
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                f = RecoveryStateCPM.pool_dino_tokens(model.forward_pixels(pixels))
            f = f.float().cpu().numpy()
            features.append(f.reshape(len(f), -1))
        if (begin + 16) % 160 == 0:
            print("BF16 encoded", begin + 16, "/", len(raw), flush=True)
    features = np.concatenate(features)
    metrics, pred = probe(features, cache["targets"], cache["fit"], cache["tune"], cache["test"], pca=True)
    d["probes"]["dino_bf16"] = metrics
    d["bf16_scope"] = (
        "Same sampled force maps, production FP32 preprocessor, BF16 DINO autocast and pooled tokens, then FP32 readout; same episode split and training-only tuning."
    )
    for example, index in zip(d["examples"], np.flatnonzero(cache["test"]), strict=True):
        example["prediction"]["dino_bf16"] = pred[index].tolist()
    (output / "report.json").write_text(json.dumps(d, ensure_ascii=False, allow_nan=False))
    np.savez_compressed(output / "bf16_features.npz", dino_bf16=features.astype(np.float16))
    render(output)
    print("BF16 results", json.dumps(metrics), flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.output)
