"""Plot synthetic contact changes and actual FK recovery paths for review."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import h5py
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from openpi.touchguide.fingertip_kinematics import FINGERS  # noqa: E402
from openpi.touchguide.fingertip_kinematics import JOINTS  # noqa: E402
from openpi.touchguide.fingertip_kinematics import FingertipKinematics  # noqa: E402

DEFAULT = Path(
    "/path/to/datasets/touchguidence/augmentation/poker_fingertip_recovery_v1/recovery/pilot_v1_30steps"
)


def run(root):
    config = json.loads((root / "config.json").read_text())
    kin_config = json.loads((Path(config["kinematic_root"]) / "config.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    k = FingertipKinematics(kin_config["model_path"])
    figures = root / "plots"
    examples = []
    with h5py.File(root / "recovery.h5", "r") as f:
        a, s = f["anchors"], f["samples"]
        codes, recommended = s["direction_codes"][:], s["recommended_training"][:]
        masks = s["perturbed_fingers"][:]
        magnitudes = s["magnitude_mm"][:]
        # Representative measured-contact single-finger cases; plus a mixed
        # pattern. Prefer 1mm to show release, approach and spatial deformation.
        for name, code in (("away", 0), ("approach", 1), ("tangent0+", 2), ("tangent1+", 4)):
            choices = np.flatnonzero(
                recommended & (masks.sum(axis=1) == 1) & np.any(codes == code, axis=1) & (magnitudes == 1)
            )
            if not len(choices):
                choices = np.flatnonzero(recommended & (masks.sum(axis=1) == 1) & np.any(codes == code, axis=1))
            if not len(choices):
                continue
            # Favor a pad with strong measured contact and an interior patch.
            weights = []
            for j in choices:
                i = int(s["anchor_index"][j])
                finger = int(np.flatnonzero(masks[j])[0])
                weights.append(float(a["tactile_map_raw"][i, finger, 0].sum()))
            j = int(choices[int(np.argmax(weights))])
            examples.append((name, j, int(np.flatnonzero(masks[j])[0])))
        choices = np.flatnonzero(recommended & (s["pattern_kind"].asstr()[:] == "mixed") & (masks.sum(axis=1) >= 3))
        if len(choices):
            j = int(choices[len(choices) // 2])
            examples.append(("mixed", j, int(np.flatnonzero(masks[j])[0])))
        rows = []
        fig, axes = plt.subplots(len(examples), 3, figsize=(11, 2.7 * len(examples)), squeeze=False)
        for row, (name, j, finger) in enumerate(examples):
            i = int(s["anchor_index"][j])
            old, new = a["tactile_map_raw"][i, finger, 0], s["tactile_map_raw"][j, finger, 0]
            for col, (label, image) in enumerate(
                (("Original normal", old), ("Synthetic normal", new), ("Difference", new - old))
            ):
                kwargs = {"cmap": "viridis", "vmin": 0, "vmax": max(float(old.max()), float(new.max()), 1e-8)}
                if col == 2:
                    bound = max(float(np.abs(image).max()), 1e-8)
                    kwargs = {"cmap": "RdBu_r", "vmin": -bound, "vmax": bound}
                handle = axes[row, col].imshow(image, origin="lower", **kwargs)
                axes[row, col].set_title(f"{name} / {FINGERS[finger]}: {label}")
                fig.colorbar(handle, ax=axes[row, col], label="N / taxel")
            rows.append(
                {
                    "example": name,
                    "sample_index": j,
                    "anchor_index": i,
                    "anchor_uid": a["anchor_uid"].asstr()[i],
                    "finger": FINGERS[finger],
                    "magnitude_mm": float(magnitudes[j]),
                    "before_fn_n": float(old.sum()),
                    "after_fn_n": float(new.sum()),
                    "phase": a["demonstration_phase"].asstr()[i],
                }
            )
        fig.tight_layout()
        fig.savefig(figures / "tactile_examples.png", dpi=160)
        plt.close(fig)
        fig, axes = plt.subplots(len(examples), 2, figsize=(12, 2.8 * len(examples)), squeeze=False)
        for row, (name, j, finger) in enumerate(examples):
            i = int(s["anchor_index"][j])
            goal = a["state"][i]
            path = np.vstack([s["state_raw"][j], s["actions_absolute"][j]])
            time_s = np.arange(31) / config["fps"]
            for selected in np.flatnonzero(masks[j]):
                segment = slice(7 + 4 * selected, 11 + 4 * selected)
                axes[row, 0].plot(time_s, path[:, segment] - goal[segment], alpha=0.8)
            axes[row, 0].set(
                title=f"{name}: joint error returns to zero",
                xlabel="Time (s)",
                ylabel="Joint target - clean state (rad)",
            )
            projected = []
            full = s["full_qpos"][j]
            for target in path:
                q = full.copy()
                q[k.state_qpos] = target
                q[k.thumb6_qpos] += target[10] - path[0, 10]
                position, _ = k.fk(q)
                projected.append(
                    np.einsum("fi,fai->fa", position - a["position"][i], a["direction_basis_world"][i]) * 1000
                )
            projected = np.asarray(projected)
            for axis, label in enumerate(("away normal", "tangent0", "tangent1")):
                axes[row, 1].plot(time_s, projected[:, finger, axis], label=label)
            axes[row, 1].set(
                title=f"{name}: actual FK / {FINGERS[finger]}",
                xlabel="Time (s)",
                ylabel="Tip displacement from clean (mm)",
            )
            axes[row, 1].legend()
        fig.tight_layout()
        fig.savefig(figures / "recovery_paths.png", dpi=160)
        plt.close(fig)
        with (root / "review_examples.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        before = a["tactile_map_raw"][:, :, 0].sum(axis=(-2, -1), dtype=np.float64)[s["anchor_index"][:]]
        after = s["tactile_map_raw"][:, :, 0].sum(axis=(-2, -1), dtype=np.float64)
        support = s["tactile_supported_fingers"][:]
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
        labels, ratios = [], []
        for code, name in ((0, "away"), (1, "approach"), (2, "tan0+"), (3, "tan0-"), (4, "tan1+"), (5, "tan1-")):
            for magnitude in (0.5, 1, 2):
                chosen = masks & support & (codes == code) & (magnitudes[:, None] == magnitude)
                if chosen.any():
                    labels.append(f"{name}\n{magnitude:g} mm")
                    ratios.append(after[chosen] / before[chosen])
        axes[0].boxplot(ratios, tick_labels=labels, showfliers=False)
        axes[0].axhline(1, color="gray", linestyle="--")
        axes[0].tick_params(axis="x", labelsize=7)
        plt.setp(axes[0].get_xticklabels(), rotation=45, ha="right")
        axes[0].set(title="Synthetic normal-force ratio by actual IK proposal", ylabel="Synthetic Fn / original Fn")
        transition = np.asarray(summary["contact_count_transition_original_row_synthetic_column"])
        handle = axes[1].imshow(transition, origin="lower", cmap="Blues")
        for y, x in np.ndindex(transition.shape):
            if transition[y, x]:
                axes[1].text(x, y, str(transition[y, x]), ha="center", va="center", fontsize=8)
        axes[1].set(
            title="Contact-count changes (all proposals)",
            xlabel="Synthetic contacting fingers",
            ylabel="Original contacting fingers",
        )
        fig.colorbar(handle, ax=axes[1], label="Proposals")
        fig.tight_layout()
        fig.savefig(figures / "force_and_contact_changes.png", dpi=160)
        plt.close(fig)
        # Sensitivity analysis is numerical, not another training dataset.
        displacement = s["normal_displacement_m"][:]
        sensitivity = []
        modeled = masks & support
        for distance_mm in (0.5, 1.0, 2.0):
            gain = np.clip(1 - displacement / (distance_mm / 1000), 0, config["max_force_gain"])
            force = np.where(modeled, before * gain, before)
            sensitivity.append(
                {
                    "release_distance_mm": distance_mm,
                    "modeled_lost_contact_fraction": float(np.mean(force[modeled] <= config["contact_threshold_n"])),
                    "mean_force_gain": float(gain[modeled].mean()),
                    "mean_contacting_fingers": float(np.mean((force > config["contact_threshold_n"]).sum(axis=1))),
                }
            )
        statistics = json.loads((root / "action_norm_stats.json").read_text())["norm_stats"]
        clean_state = a["state"][:].astype(np.float32)
        low, high = (np.asarray(statistics["state"][key], dtype=np.float32) for key in ("q01", "q99"))
        clean_normalized_state = (clean_state - low) / (high - low + 1e-6) * 2 - 1
        clean_actions, noisy_actions = a["actions"][:], s["actions"][:]
        noisy_states = s["state_normalized"][:]
        coordinate_rows = []
        for dimension, joint in enumerate(JOINTS):
            coordinate_rows.append(
                {
                    "dimension": dimension,
                    "joint": joint,
                    "clean_action_outside_unit_fraction": float(np.mean(np.abs(clean_actions[:, :, dimension]) > 1)),
                    "recovery_action_outside_unit_fraction": float(np.mean(np.abs(noisy_actions[:, :, dimension]) > 1)),
                    "clean_action_max_abs_normalized": float(np.max(np.abs(clean_actions[:, :, dimension]))),
                    "recovery_action_max_abs_normalized": float(np.max(np.abs(noisy_actions[:, :, dimension]))),
                    "clean_state_outside_unit_fraction": float(
                        np.mean(np.abs(clean_normalized_state[:, dimension]) > 1)
                    ),
                    "noisy_state_outside_unit_fraction": float(np.mean(np.abs(noisy_states[:, dimension]) > 1)),
                }
            )
        with (root / "normalization_by_joint.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(coordinate_rows[0]))
            writer.writeheader()
            writer.writerows(coordinate_rows)
        normalization = {
            "clean_action_outside_unit_fraction": float(np.mean(np.abs(clean_actions) > 1)),
            "recovery_action_outside_unit_fraction": float(np.mean(np.abs(noisy_actions) > 1)),
            "clean_state_outside_unit_fraction": float(np.mean(np.abs(clean_normalized_state) > 1)),
            "noisy_state_outside_unit_fraction": float(np.mean(np.abs(noisy_states) > 1)),
            "clean_action_max_abs_normalized": float(np.max(np.abs(clean_actions))),
            "recovery_action_max_abs_normalized": float(np.max(np.abs(noisy_actions))),
            "quantile_range_is_not_joint_limit": True,
            "values_are_not_clipped": True,
            "comparison_uses_phase_stratified_pilot_not_entire_demonstration_distribution": True,
        }
        source_config = json.loads((Path(kin_config["anchor_root"]) / "meta/selection_config.json").read_text())
        force_quantiles = source_config["train_positive_fn_quantiles_n"]
        force_range = {
            "source_train_positive_fn_p99_n": force_quantiles["p99"],
            "source_train_positive_fn_max_n": force_quantiles["max"],
            "synthetic_fn_max_n": float(after.max()),
            "modeled_above_train_p99_fraction": float(np.mean(after[modeled] > force_quantiles["p99"])),
            "modeled_above_train_max_fraction": float(np.mean(after[modeled] > force_quantiles["max"])),
        }
        recovery_progress = {
            "after_first_4_targets_fraction": float(f["interpolation_alpha"][3]),
            "after_first_5_targets_fraction": float(f["interpolation_alpha"][4]),
            "after_first_15_targets_fraction": float(f["interpolation_alpha"][14]),
            "inference_execute_steps_changed": False,
        }

    files = [p for p in root.rglob("*") if p.is_file() and p.name != "review_report.json"]
    report = {
        "data_root": str(root),
        "files_bytes_excluding_this_report": sum(p.stat().st_size for p in files),
        "plots": [str(p) for p in sorted(figures.glob("*.png"))],
        "examples": rows,
        "normal_rule_sensitivity": sensitivity,
        "normalization_comparison": normalization,
        "force_range_comparison": force_range,
        "recovery_progress": recovery_progress,
        "sensitivity_is_not_physics_validation": True,
    }
    (root / "review_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({k: report[k] for k in ("files_bytes_excluding_this_report", "plots", "normal_rule_sensitivity")}),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT)
    run(parser.parse_args().data_root.resolve())
