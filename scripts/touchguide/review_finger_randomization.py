"""Review signed component randomization and validated contact goals."""

import argparse
from collections import Counter
import json
from pathlib import Path
import re

import numpy as np

NAMES = ("thumb", "index", "middle", "ring", "pinky")
AXES = ("normal", "tangent1", "tangent2")


def components(variant):
    values = variant.get("requested_components_mm")
    if values is not None:
        return np.asarray(values)
    values = np.zeros((5, 3))
    for f, code in enumerate(variant["codes"]):
        if code >= 0:
            values[f, code // 2] = variant["finger_magnitudes_mm"][f] * (1 if code % 2 == 0 else -1)
    return values


def review_html(path):
    data = json.loads(
        re.search(r'<script id="payload" type="application/json">(.*?)</script>', path.read_text(), re.DOTALL)[1]
    )
    counts, axis_stats = Counter(), {name: Counter() for name in AXES}
    per_finger = [Counter() for _ in range(5)]
    patterns = []
    max_requested = max_actual = 0.0
    all_components_realized = True
    for sample in data["samples"]:
        unique = set()
        for v in sample["variants"]:
            requested = components(v)
            selected = np.asarray(v["mask"], bool)
            norms = np.linalg.norm(requested, axis=1)
            actual = np.asarray(v["local_delta_mm"])
            max_requested = max(max_requested, float(norms.max()))
            max_actual = max(max_actual, float(np.linalg.norm(actual, axis=1).max()))
            active = requested != 0
            assert np.array_equal(selected, active.any(axis=1))
            assert np.max(np.abs(norms - v["finger_magnitudes_mm"])) < 1e-8
            all_components_realized &= bool(np.all(np.abs(actual[active]) > 0.05))
            within_finger_mix = active[:, 0] & active[:, 1:].any(axis=1)
            counts["variants"] += 1
            counts["single_finger" if selected.sum() == 1 else "multi_finger"] += 1
            counts[f"finger_count_{int(selected.sum())}"] += 1
            counts["variants_with_single_finger_normal_tangent"] += int(within_finger_mix.any())
            counts["variants_with_single_finger_three_axes"] += int(active.all(axis=1).any())
            if selected.sum() > 1:
                directions = {
                    tuple(np.round(vector / length, 8))
                    for vector, length in zip(requested[selected], norms[selected], strict=True)
                }
                counts["multi_same_direction" if len(directions) == 1 else "multi_different_directions"] += 1
                counts[
                    "multi_same_amplitude"
                    if len(set(np.round(norms[selected], 8))) == 1
                    else "multi_different_amplitudes"
                ] += 1
                normal, tangent = bool(active[:, 0].any()), bool(active[:, 1:].any())
                counts[
                    "multi_normal_and_tangent"
                    if normal and tangent
                    else "multi_all_normal"
                    if normal
                    else "multi_all_tangent"
                ] += 1
            losses = gains = 0
            for f in range(5):
                stats = per_finger[f]
                before, after = sample["force"][f], v["force"][f]
                old, new = before > 0.001, after > 0.001
                stats["perturbed" if selected[f] else "unchanged"] += 1
                stats[f"contact_{int(old)}_to_{int(new)}"] += 1
                losses += int(old and not new)
                gains += int(not old and new)
                if selected[f]:
                    bitmask = sum(1 << int(axis) for axis in np.flatnonzero(active[f]))
                    stats[f"component_pattern_{bitmask}"] += 1
                    counts[f"perturbed_fingers_with_{int(active[f].sum())}_active_components"] += 1
                    for a, name in enumerate(AXES):
                        x = requested[f, a]
                        axis_stats[name]["zero" if x == 0 else "positive" if x > 0 else "negative"] += 1
                        stats[f"{name}_zero" if x == 0 else f"{name}_positive" if x > 0 else f"{name}_negative"] += 1
                    stats[
                        "force_increased_over_5pct"
                        if after > before * 1.05
                        else "force_decreased_over_5pct"
                        if after < before * 0.95
                        else "force_within_5pct"
                    ] += 1
                    goals = v.get("requested_contact_changes")
                    if goals is not None:
                        assert new == (goals[f] == 0)
                        stats["requested_release" if goals[f] else "requested_keep_contact"] += 1
            counts["variants_with_contact_loss"] += int(losses > 0)
            counts["variants_with_contact_gain"] += int(gains > 0)
            counts["lost_finger_contacts"] += losses
            counts["gained_finger_contacts"] += gains
            unique.add(tuple(requested.ravel()))
        patterns.append(
            {
                "episode": sample["episode"],
                "frame": sample["frame"],
                "unique_patterns": len(unique),
                "variants": len(sample["variants"]),
            }
        )
    return {
        "anchors": len(data["samples"]),
        "counts": dict(counts),
        "axis_sign_and_zero_counts_on_perturbed_fingers": {name: dict(stats) for name, stats in axis_stats.items()},
        "per_finger": {name: dict(stats) for name, stats in zip(NAMES, per_finger, strict=True)},
        "per_origin_patterns": patterns,
        "max_requested_norm_mm": max_requested,
        "max_actual_norm_mm": max_actual,
        "all_requested_nonzero_components_realized_above_0_05mm": all_components_realized,
    }


def main(args):
    root = args.data_root.resolve()
    result = {"current": review_html(root / "visualization/index.html")}
    summary = json.loads((root / "augmented/summary.json").read_text())
    probes, proposals = Counter(), Counter()
    for episode in summary["episodes"]:
        proposals.update(episode.get("proposal_counts", {}))
        for origin in episode.get("per_origin_feasibility", []):
            probes.update(origin["probe_counts"])
    result["proposal_audit"] = {
        "single_finger_option_probes": dict(probes),
        "random_combination_draws": dict(proposals),
    }
    if args.baseline_html:
        result["baseline"] = review_html(args.baseline_html)
    result["scope"] = (
        "All preview variants; numerical/FK/contact-goal consistency, not physical execution validation or proof of uniform sampling"
    )
    destination = root / "visualization/randomization_review.json"
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "current": result["current"],
                "baseline_counts": result.get("baseline", {}).get("counts"),
                "report": str(destination),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--baseline-html", type=Path)
    main(parser.parse_args())
