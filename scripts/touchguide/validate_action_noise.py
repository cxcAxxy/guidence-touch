#!/usr/bin/env python3
"""CPU validation of Step 3 flow action noise on a real matched batch.

Reads three clean training examples with images disabled. No model, scores,
losses, training, or additional negative chunks are created. Source files are
read-only; the only output artifact is the explicit --report JSON file.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import torch

from openpi.touchguide.batching import CPMBatchLoader
from openpi.touchguide.data import CPMDataset
from openpi.touchguide.noise import FlowActionNoise
from openpi.touchguide.noise import NoiseConfig
from openpi.touchguide.noise import truncated_geometric_probabilities

DEFAULT_ROOT = Path(
    "/path/to/datasets/sim/poker-draw/pi05_trex/touchguide/cpm_datasets/poker_0920_200_touchguide_v1"
)


def require(condition: bool, message: str) -> None:  # noqa: FBT001 - assertion helper takes a predicate
    if not condition:
        raise AssertionError(message)


def snapshot(batch: dict) -> dict:
    return {key: value.clone() if isinstance(value, torch.Tensor) else deepcopy(value) for key, value in batch.items()}


def assert_unchanged(batch: dict, saved: dict) -> None:
    require(set(batch) == set(saved), "Noise augmentation added fields to the input dictionary")
    for key, value in saved.items():
        if isinstance(value, torch.Tensor):
            require(torch.equal(batch[key], value), f"Input {key} was modified in place")
        else:
            require(batch[key] == value, f"Input {key} was modified in place")


def verify_augmented(batch: dict, saved: dict, result: dict, time: float | None = None) -> float:
    assert_unchanged(batch, saved)
    require(result is not batch, "Noise augmentation must return a separate dictionary")
    for key, value in batch.items():
        require(result[key] is value, f"Noise augmentation replaced the clean {key} object")
    require(
        "negative_actions" not in result and "noisy_negative_actions" not in result,
        "Noise augmentation created separate negative candidate tensors",
    )
    clean = batch["actions"]
    epsilon = result["noise_epsilon"]
    noisy = result["noisy_actions"]
    flow_time = result["flow_time"]
    require(
        isinstance(flow_time, torch.Tensor) and flow_time.shape == (),
        "One scalar flow_time must be shared by all batch action columns",
    )
    require(
        flow_time.dtype == clean.dtype and flow_time.device == clean.device,
        "flow_time does not preserve action dtype/device",
    )
    require(0 <= float(flow_time) <= 1, "flow_time lies outside [0,1]")
    if time is not None:
        expected_time = torch.tensor(time, dtype=clean.dtype, device=clean.device)
        require(torch.equal(flow_time, expected_time), "flow_time disagrees with requested time")
    for name, tensor in (("noisy_actions", noisy), ("noise_epsilon", epsilon)):
        require(isinstance(tensor, torch.Tensor) and tensor.shape == clean.shape, f"Invalid {name} shape")
        require(tensor.dtype == clean.dtype and tensor.device == clean.device, f"Invalid {name} dtype/device")
        require(bool(torch.isfinite(tensor).all()), f"Nonfinite {name}")
    expected = (1.0 - flow_time) * clean + flow_time * epsilon
    torch.testing.assert_close(noisy, expected, rtol=1e-6, atol=1e-6)
    return float((noisy - expected).abs().max())


def verify_probabilities() -> tuple[torch.Tensor, dict]:
    levels, p = 10, 0.3
    probabilities = truncated_geometric_probabilities(num_levels=levels, p=p)
    require(
        isinstance(probabilities, torch.Tensor)
        and probabilities.dtype == torch.float64
        and probabilities.shape == (levels,),
        "Probabilities must be float64[10]",
    )
    require(probabilities.device.type == "cpu", "CPU validation expects CPU probabilities")
    raw = p * (1.0 - p) ** torch.arange(levels, dtype=torch.float64)
    expected = raw / raw.sum()
    torch.testing.assert_close(probabilities, expected, rtol=1e-13, atol=1e-15)
    require(bool((probabilities > 0).all()), "All truncated geometric levels must have positive mass")
    require(abs(float(probabilities.sum()) - 1.0) < 1e-14, "Geometric probabilities were not normalized")
    clamped_tail_mass = (1.0 - p) ** (levels - 1)
    require(
        abs(float(probabilities[-1]) - clamped_tail_mass) > 0.01,
        "The tail looks like a clamped geometric sample rather than truncated renormalization",
    )
    return probabilities, {
        "num_levels": levels,
        "geometric_p": p,
        "probabilities": probabilities.tolist(),
        "formula": "p*(1-p)**k / (1-(1-p)**K), k=0..K-1",
        "sum": float(probabilities.sum()),
        "clamped_geometric_tail_mass": clamped_tail_mass,
        "truncated_renormalization_verified": True,
    }


def verify_time_formula(noise: FlowActionNoise, batch: dict, saved: dict) -> dict:
    checks = []
    for level in range(10):
        time = level / 10.0
        result = noise.augment_at_time(batch, time, generator=torch.Generator().manual_seed(900 + level))
        error = verify_augmented(batch, saved, result, time)
        if level == 0:
            require(torch.equal(result["noisy_actions"], batch["actions"]), "t=0 is not exactly clean")
        checks.append({"noise_level": level, "flow_time": time, "formula_max_abs_error": error})
    endpoint = noise.augment_at_time(batch, 1.0, generator=torch.Generator().manual_seed(1100))
    verify_augmented(batch, saved, endpoint, 1.0)
    require(
        torch.equal(endpoint["noisy_actions"], endpoint["noise_epsilon"]), "t=1 is not exactly the Gaussian epsilon"
    )
    return {"level_checks": checks, "t0_equals_clean_exactly": True, "t1_equals_epsilon_exactly": True}


def verify_shared_generator(noise: FlowActionNoise, batch: dict, saved: dict, probabilities: torch.Tensor) -> dict:
    seed = 271828
    global_before = torch.random.get_rng_state().clone()
    supplied = torch.Generator().manual_seed(seed)
    result = noise.augment(batch, generator=supplied)
    verify_augmented(batch, saved, result)
    require(torch.equal(torch.random.get_rng_state(), global_before), "Augment consumed the global CPU RNG")
    reference = torch.Generator().manual_seed(seed)
    expected_level = int(torch.multinomial(probabilities, 1, generator=reference).item())
    expected_epsilon = torch.randn(
        batch["actions"].shape, dtype=batch["actions"].dtype, device="cpu", generator=reference
    )
    actual_level = result["noise_level"]
    require(
        isinstance(actual_level, torch.Tensor) and actual_level.shape == () and actual_level.dtype == torch.int64,
        "noise_level must be an int64 scalar",
    )
    require(int(actual_level) == expected_level, "Noise level does not use the supplied generator")
    require(
        torch.equal(result["noise_epsilon"], expected_epsilon),
        "Epsilon does not follow the level draw on the same supplied generator",
    )
    require(
        torch.equal(supplied.get_state(), reference.get_state()),
        "Noise sampling consumed a different generator sequence",
    )
    require(
        torch.equal(result["flow_time"], torch.tensor(expected_level / 10, dtype=batch["actions"].dtype)),
        "Sampled level-to-flow-time mapping is incorrect",
    )
    repeated = noise.augment(batch, generator=torch.Generator().manual_seed(seed))
    changed = noise.augment(batch, generator=torch.Generator().manual_seed(seed + 1))
    for key in ("noise_level", "flow_time", "noise_epsilon", "noisy_actions"):
        require(torch.equal(result[key], repeated[key]), f"Same seed did not reproduce {key}")
    require(
        not torch.equal(result["noise_epsilon"], changed["noise_epsilon"]), "Changing the seed did not change epsilon"
    )
    verify_augmented(batch, saved, repeated)
    verify_augmented(batch, saved, changed)
    return {
        "seed": seed,
        "sampled_noise_level": expected_level,
        "same_seed_reproduces_exactly": True,
        "different_seed_changes_epsilon": True,
        "sampling_then_gaussian_use_one_generator": True,
        "global_rng_unchanged": True,
    }


def verify_level_frequencies(noise: FlowActionNoise, batch: dict, saved: dict, probabilities: torch.Tensor) -> dict:
    draws, seed = 20000, 20260927
    generator = torch.Generator().manual_seed(seed)
    global_before = torch.random.get_rng_state().clone()
    counts = torch.zeros(10, dtype=torch.int64)
    for _ in range(draws):
        result = noise.augment(batch, generator=generator)
        level = int(result["noise_level"])
        require(0 <= level < 10, "Sampled noise level outside 0..9")
        counts[level] += 1
    assert_unchanged(batch, saved)
    require(torch.equal(global_before, torch.random.get_rng_state()), "Frequency sampling consumed global RNG")
    frequencies = counts.to(torch.float64) / draws
    # Six binomial standard deviations, with a small finite-count allowance.
    # This comfortably permits sampling noise while detecting a clamped tail.
    tolerance = 6.0 * torch.sqrt(probabilities * (1.0 - probabilities) / draws) + 2.0 / draws
    deviations = (frequencies - probabilities).abs()
    require(
        bool((deviations <= tolerance).all()),
        f"Sampled frequencies disagree with the truncated geometric law: {frequencies.tolist()}",
    )
    return {
        "draws": draws,
        "seed": seed,
        "counts": counts.tolist(),
        "frequencies": frequencies.tolist(),
        "absolute_tolerances": tolerance.tolist(),
        "maximum_absolute_deviation": float(deviations.max()),
        "frequency_check_passed": True,
    }


def verify_fixed_validation_noise(noise: FlowActionNoise, batch: dict, saved: dict) -> dict:
    global_before = torch.random.get_rng_state().clone()
    first = noise.augment_validation_at_time(batch, 0.3, seed=42)
    repeated = noise.augment_validation_at_time(batch, 0.3, seed=42)
    another_level = noise.augment_validation_at_time(batch, 0.7, seed=42)
    changed_seed = noise.augment_validation_at_time(batch, 0.3, seed=43)
    for result, time in ((first, 0.3), (repeated, 0.3), (another_level, 0.7), (changed_seed, 0.3)):
        verify_augmented(batch, saved, result, time)
    require(
        torch.equal(first["noisy_actions"], repeated["noisy_actions"]), "Fixed validation noise is not reproducible"
    )
    require(
        torch.equal(first["noise_epsilon"], another_level["noise_epsilon"]),
        "Fixed validation epsilon changed across flow times",
    )
    require(
        not torch.equal(first["noise_epsilon"], changed_seed["noise_epsilon"]),
        "Fixed validation epsilon ignored its seed",
    )
    require(
        not torch.equal(first["noise_epsilon"][0], first["noise_epsilon"][1]),
        "Different validation sample IDs received identical epsilon",
    )
    permutation = torch.tensor([2, 0, 1])
    reordered = dict(batch)
    reordered["actions"] = batch["actions"][permutation]
    reordered["tactile_map_raw"] = batch["tactile_map_raw"][permutation]
    reordered["sample_ids"] = [batch["sample_ids"][i] for i in permutation.tolist()]
    reordered["metadata"] = [batch["metadata"][i] for i in permutation.tolist()]
    reordered_saved = snapshot(reordered)
    moved = noise.augment_validation_at_time(reordered, 0.3, seed=42)
    verify_augmented(reordered, reordered_saved, moved, 0.3)
    require(
        torch.equal(moved["noise_epsilon"], first["noise_epsilon"][permutation]),
        "Fixed validation epsilon followed batch position instead of sample identity",
    )
    require(
        torch.equal(moved["noisy_actions"], first["noisy_actions"][permutation]),
        "Reordered samples changed fixed noisy actions",
    )
    regrouped = dict(reordered)
    for key in ("actions", "tactile_map_raw"):
        regrouped[key] = reordered[key][:2]
    regrouped["sample_ids"] = reordered["sample_ids"][:2]
    regrouped["metadata"] = reordered["metadata"][:2]
    regrouped["targets"] = torch.arange(2)
    regrouped["positive_mask"] = torch.eye(2, dtype=torch.bool)
    regrouped["negative_mask"] = ~regrouped["positive_mask"]
    regrouped_saved = snapshot(regrouped)
    subset = noise.augment_validation_at_time(regrouped, 0.3, seed=42)
    verify_augmented(regrouped, regrouped_saved, subset, 0.3)
    require(
        torch.equal(subset["noise_epsilon"], moved["noise_epsilon"][:2]),
        "Changing validation batch size changed per-ID epsilon",
    )
    require(torch.equal(global_before, torch.random.get_rng_state()), "Fixed validation noise consumed global RNG")
    return {
        "seed": 42,
        "repeated_call_matches": True,
        "epsilon_shared_across_levels": True,
        "epsilon_follows_sample_id_after_reordering": True,
        "epsilon_independent_of_batch_size": True,
        "different_seed_changes_epsilon": True,
        "global_rng_unchanged": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    dataset = CPMDataset(root, "train", include_images=False, include_negatives=False)
    try:
        indices = [0, len(dataset) // 2, len(dataset) - 1]
        batch = next(iter(CPMBatchLoader(dataset, [indices], include_auxiliary=False)))
        require("head" not in batch and "wrist" not in batch, "CPU noise check unexpectedly decoded images")
        require(
            batch["actions"].shape == (3, 30, 27) and batch["actions"].device.type == "cpu",
            "Real action batch is not CPU[3,30,27]",
        )
        saved = snapshot(batch)
        noise = FlowActionNoise(NoiseConfig(num_flow_steps=10, geometric_p=0.3, time_scope="batch", enabled=True))
        probabilities, probability_report = verify_probabilities()
        time_report = verify_time_formula(noise, batch, saved)
        generator_report = verify_shared_generator(noise, batch, saved, probabilities)
        frequency_report = verify_level_frequencies(noise, batch, saved, probabilities)
        validation_report = verify_fixed_validation_noise(noise, batch, saved)
        disabled = FlowActionNoise(NoiseConfig(num_flow_steps=10, geometric_p=0.3, time_scope="batch", enabled=False))
        disabled_result = disabled.augment(batch, generator=torch.Generator().manual_seed(42))
        verify_augmented(batch, saved, disabled_result, 0.0)
        require(
            disabled_result["noisy_actions"] is batch["actions"], "Disabled noise did not retain clean action object"
        )
        require(
            torch.equal(disabled_result["noise_epsilon"], torch.zeros_like(batch["actions"])),
            "Disabled noise should expose zero epsilon",
        )
        assert_unchanged(batch, saved)
        manifest_path = root / "meta/manifest.json"
        report = {
            "status": "passed",
            "scope": "step3_flow_action_noise_cpu",
            "dataset_root": str(root),
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "real_batch": {
                "sample_indices": indices,
                "sample_ids": batch["sample_ids"],
                "action_shape": list(batch["actions"].shape),
                "images_enabled": False,
            },
            "probabilities": probability_report,
            "time_formula": time_report,
            "generator": generator_report,
            "frequency": frequency_report,
            "fixed_validation_noise": validation_report,
            "input_tensors_ids_targets_masks_and_metadata_preserved": True,
            "all_action_columns_share_scalar_flow_time": True,
            "disabled_noise_preserves_clean_actions": True,
            "separate_negative_action_tensors_created": False,
            "gpu_tested": False,
            "model_loaded": False,
            "training_run": False,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "status": "passed",
                    "report": str(args.report.resolve()),
                    "frequency_draws": 20000,
                    "formula_levels_checked": 10,
                },
                indent=2,
            )
        )
    finally:
        dataset.close()


if __name__ == "__main__":
    main()
