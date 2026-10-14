"""Compare fixed-input JAX and PyTorch actions and write a baseline manifest."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def metrics(x):
    a = np.abs(x.astype(np.float64))
    return {"mae": float(a.mean()), "rmse": float(np.sqrt(np.mean(a ** 2))),
            "p95_abs": float(np.quantile(a, .95)), "p99_abs": float(np.quantile(a, .99)),
            "max_abs": float(a.max())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--jax-checkpoint", type=Path, required=True)
    p.add_argument("--torch-checkpoint", type=Path, required=True)
    args = p.parse_args()
    root = args.directory
    shared = np.load(root / "fixed_inputs.npz", allow_pickle=False)
    jax_actions = np.load(root / "jax_outputs.npz", allow_pickle=False)["actions"]
    torch_actions = np.load(root / "torch_outputs.npz", allow_pickle=False)["actions"]
    assert jax_actions.shape == torch_actions.shape == shared["noise"].shape
    assert np.isfinite(jax_actions).all() and np.isfinite(torch_actions).all()
    norm_file = args.jax_checkpoint / "assets/normalizer/norm_stats.json"
    saved_norm = args.torch_checkpoint / "assets/normalizer/norm_stats.json"
    assert sha256(norm_file) == sha256(saved_norm), "Copied normalizer differs"
    stats = json.loads(norm_file.read_text())["norm_stats"]["actions"]
    q01, q99 = np.asarray(stats["q01"]), np.asarray(stats["q99"])
    def physical(x):
        x = (x[..., :27] + 1.) / 2. * (q99 - q01 + 1e-6) + q01
        x[..., :7] += shared["raw_state"][:, None, :7]
        return x
    difference = torch_actions - jax_actions
    rad = physical(torch_actions) - physical(jax_actions)
    norm = metrics(difference[..., :27])
    torch_meta = json.loads((root / "torch_outputs.json").read_text())
    per_observation = [metrics(x[..., :27]) for x in difference]
    passed = (all(x["rmse"] <= .01 and x["max_abs"] <= .05 for x in per_observation) and norm["rmse"] <= .01 and norm["max_abs"] <= .05 and
              torch_meta["guidance_disabled_max_abs"] == 0. and torch_meta["base_frozen"])
    report = {
        "schema": "poker_touchguide_baseline_v1", "conversion_verified": passed,
        "source_jax_checkpoint": str(args.jax_checkpoint), "pytorch_checkpoint": str(args.torch_checkpoint),
        "base_step": 20000, "action_horizon": 30, "execute_steps": 30,
        "denoising_steps": 10, "action_dim_model": 32, "action_dim_physical": 27,
        "action_representation": "right arm 7 relative targets + right hand 20 absolute targets; radians",
        "normalizer": str(norm_file), "normalizer_sha256": sha256(norm_file),
        "source_payload_files": [{"path": str(p.relative_to(args.jax_checkpoint)),
            "size_bytes": p.stat().st_size, "sha256": sha256(p)}
            for p in sorted((args.jax_checkpoint / "params").rglob("*")) if p.is_file()],
        "conversion_script_sha256": sha256(Path(__file__).resolve().parents[2] / "examples/convert_jax_model_to_pytorch.py"),
        "model_safetensors_sha256": sha256(args.torch_checkpoint / "model.safetensors"),
        "fixed_inputs_sha256": sha256(root / "fixed_inputs.npz"),
        "config": json.loads((args.torch_checkpoint / "config.json").read_text()),
        "engineering_screen": {"normalized_rmse_max": .01, "normalized_max_abs_max": .05,
            "note": "Engineering screening thresholds, not official OpenPI tolerances or a simulator success guarantee."},
        "normalized_first27": norm, "normalized_all32": metrics(difference),
        "arm_difference_radians": metrics(rad[..., :7]),
        "hand_difference_radians": metrics(rad[..., 7:]),
        "all_difference_degrees": metrics(np.rad2deg(rad)),
        "per_joint_rmse_radians": np.sqrt(np.mean(rad ** 2, axis=(0, 1))).tolist(),
        "per_horizon_step_rmse_radians": np.sqrt(np.mean(rad ** 2, axis=(0, 2))).tolist(),
        "per_observation_normalized_metrics": per_observation,
        "torch_verification": torch_meta,
        "jax_verification": json.loads((root / "jax_outputs.json").read_text()),
        "scope": "Fixed real rollout observations; no simulator runs, no guidance network, no training.",
        "tactile_expert_note": "Existing T-Rex tactile expert was trained on 24999, and is not loaded into this 20000 baseline.",
    }
    path = root / "baseline_manifest.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"conversion_verified": passed, "normalized_first27": norm,
                      "arm_radians": report["arm_difference_radians"],
                      "hand_radians": report["hand_difference_radians"], "report": str(path)}, indent=2))
    if not passed:
        raise SystemExit("Conversion screening failed. Inspect differences before adding guidance.")


if __name__ == "__main__":
    main()
