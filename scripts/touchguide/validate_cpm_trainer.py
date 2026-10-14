#!/usr/bin/env python3
"""Independent CPU fixtures for CPM trainer metrics, scheduling and validation.

No user dataset or pretrained backbone is read and no optimizer/training loop
is run. The fake model has deterministic tiny encoders, while augmentation
uses the existing per-ID validation noise implementation.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib.util
from itertools import pairwise
import json
import math
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
import torch.nn.functional as functional

from openpi.touchguide.noise import FlowActionNoise
from openpi.touchguide.noise import NoiseConfig


def require(condition, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def load_trainer(path: Path):
    name = "cpm_trainer_fixture_subject"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot import trainer: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def action_features(actions: torch.Tensor, flow_time: float | torch.Tensor) -> torch.Tensor:
    mean = actions[:, :, :2].mean(dim=1)
    time = torch.as_tensor(flow_time, dtype=actions.dtype, device=actions.device)
    return functional.normalize(torch.stack((mean[:, 0] + 0.2 * time, mean[:, 1] - 0.1 * time), dim=-1), dim=-1)


def observation_features(head: torch.Tensor, wrist: torch.Tensor) -> torch.Tensor:
    return functional.normalize(torch.stack((head[:, 0, 0, 0] + 1.0, wrist[:, 0, 0, 0] + 0.5), dim=-1), dim=-1)


class FixtureError(RuntimeError):
    pass


class FakeActionEncoder(nn.Module):
    def __init__(self, *, fail_on_action=False):
        super().__init__()
        self.records = []
        self.fail_on_action = fail_on_action

    def forward(self, actions: torch.Tensor, flow_time: float | torch.Tensor) -> torch.Tensor:
        if self.fail_on_action:
            raise FixtureError("Deliberate action failure to check finally restoration")
        self.records.append({"actions": actions.detach().clone(), "flow_time": float(flow_time)})
        return action_features(actions, flow_time)


class FakeCPM(nn.Module):
    def __init__(self, *, fail_on_score: bool = False):
        super().__init__()
        self.register_buffer("temperature", torch.tensor(0.4))
        self.action_encoder = FakeActionEncoder(fail_on_action=fail_on_score)
        self.encode_calls = 0
        self.score_calls = 0
        self.fail_on_score = fail_on_score

    def encode_observation(self, head, wrist, tactile_map_raw):
        require(not self.training, "Validation did not put the fake model into eval mode")
        self.encode_calls += 1
        return observation_features(head, wrist)

    def score_actions(self, observation_embedding, noisy_actions, flow_time):
        require(not self.training, "Validation scored actions in train mode")
        self.score_calls += 1
        if self.fail_on_score:
            raise FixtureError("Deliberate scoring failure to check finally restoration")
        return observation_embedding @ self.action_encoder(noisy_actions, flow_time).T

    def forward(self, batch):
        raise AssertionError("Validation must cache observation embeddings instead of full forward per noise time")


class RecordingNoise(FlowActionNoise):
    def __init__(self):
        super().__init__(NoiseConfig(num_flow_steps=10, geometric_p=0.3, time_scope="batch", enabled=True))
        self.records = []

    def augment_validation_at_time(self, batch, flow_time, *, seed=42):
        result = super().augment_validation_at_time(batch, flow_time, seed=seed)
        self.records.append(
            {
                "sample_ids": list(batch["sample_ids"]),
                "flow_time": float(flow_time),
                "epsilon": result["noise_epsilon"].detach().clone(),
            }
        )
        return result


def make_batch(identities: list[int]) -> dict:
    size = len(identities)
    head = torch.zeros(size, 3, 224, 224)
    wrist = torch.zeros_like(head)
    actions = torch.zeros(size, 30, 27)
    for row, identity in enumerate(identities):
        head[row, 0, 0, 0] = identity * 0.2
        wrist[row, 0, 0, 0] = 0.7 - identity * 0.11
        actions[row, :, 0] = 0.2 + identity * 0.17
        actions[row, :, 1] = 1.0 - identity * 0.12
    positive = torch.eye(size, dtype=torch.bool)
    return {
        "head": head,
        "wrist": wrist,
        "tactile_map_raw": torch.zeros(size, 5, 3, 7, 5),
        "actions": actions,
        "targets": torch.arange(size),
        "positive_mask": positive,
        "negative_mask": ~positive,
        "sample_ids": [f"fixture:{identity}" for identity in identities],
        "metadata": [{"split": "val", "episode_index": identity, "frame_index": 0} for identity in identities],
    }


class ToyCleanLoader:
    def __init__(self, batches: list[dict]):
        self.batches = batches
        self.dataset = SimpleNamespace(split="val")
        self.iterations = 0

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        self.iterations += 1
        # Real PyTorch DataLoader iteration can draw a worker/base seed from
        # the global CPU RNG. Deliberately emulate that to test the wrapper.
        torch.rand(())
        np.random.rand()
        random.random()
        yield from self.batches


def assert_fixed_epsilon(records: list[dict], expected_ids: set[str]) -> None:
    by_id = {}
    for record in records:
        for row, identity in enumerate(record["sample_ids"]):
            epsilon = record["epsilon"][row]
            if identity in by_id:
                require(torch.equal(by_id[identity], epsilon), f"Validation epsilon changed for {identity}")
            else:
                by_id[identity] = epsilon.clone()
    require(set(by_id) == expected_ids, "Validation did not augment exactly the expected sample IDs")


def expect_error(operation, expected, message):
    try:
        operation()
    except expected:
        return
    raise AssertionError(message)


def assert_rng_equal(left, right):
    require(torch.equal(left["torch_cpu"], right["torch_cpu"]), "Validation did not restore CPU Torch RNG")
    require(left["python"] == right["python"], "Validation did not restore Python RNG")
    require(
        left["numpy"][0] == right["numpy"][0]
        and np.array_equal(left["numpy"][1], right["numpy"][1])
        and left["numpy"][2:] == right["numpy"][2:],
        "Validation did not restore NumPy RNG",
    )


def verify_weighted_metrics(subject):
    meter = subject.WeightedMetrics()
    meter.add({"loss": torch.tensor(2.0), "accuracy": 0.25}, 4)
    meter.add({"loss": torch.tensor(8.0), "accuracy": 1.0}, 2)
    means = meter.mean()
    require(meter.count == 6 and math.isclose(means["loss"], 4.0), "Loss was averaged by batches, not samples")
    require(math.isclose(means["accuracy"], 0.5), "Accuracy ignored sample weights")
    resumed = subject.WeightedMetrics.from_state(meter.state())
    resumed.add({"loss": 1.0, "accuracy": 0.0}, 2)
    require(
        resumed.count == 8 and math.isclose(resumed.mean()["loss"], 3.25),
        "Resumed metric accumulation lost numerator/count state",
    )
    require(meter.count == 6 and meter.mean() == means, "Metric state reconstruction aliased the original accumulator")
    expect_error(lambda: subject.WeightedMetrics().mean(), ValueError, "Empty mean must reject")
    expect_error(lambda: subject.WeightedMetrics().add({"loss": 1.0}, 0), ValueError, "Zero weight must reject")
    expect_error(
        lambda: subject.WeightedMetrics().add({"loss": float("nan")}, 2),
        FloatingPointError,
        "Nonfinite metrics must reject",
    )
    return {"count": meter.count, "weighted_means": means, "resumed_loss": resumed.mean()["loss"]}


def verify_scheduler(subject):
    kwargs = {"total_steps": 100, "warmup_steps": 5, "min_ratio": 0.1}
    factors = [subject.lr_factor(step, **kwargs) for step in range(100)]
    torch.testing.assert_close(torch.tensor(factors[:5]), torch.tensor([0.2, 0.4, 0.6, 0.8, 1.0]))
    require(factors[5] == 1.0 and math.isclose(factors[99], 0.1), "Warmup/cosine transition or final floor is wrong")
    require(all(left >= right for left, right in pairwise(factors[5:])), "Cosine decay is not monotonic")
    middle = 5 + 47
    expected_middle = 0.1 + 0.9 * (1 + math.cos(math.pi * 47 / 94)) / 2
    require(math.isclose(factors[middle], expected_middle), "Cosine midpoint is wrong")
    require(math.isclose(subject.lr_factor(120, **kwargs), 0.1), "Schedule did not remain at its terminal floor")
    require(
        subject.lr_factor(0, total_steps=10, warmup_steps=0, min_ratio=0.1) == 1.0,
        "Zero-warmup schedule did not begin at full learning rate",
    )
    require(
        math.isclose(subject.lr_factor(9, total_steps=10, warmup_steps=9, min_ratio=0.1), 0.1),
        "A schedule with only one decay update did not reach its terminal floor",
    )
    require(
        math.isclose(subject.lr_factor(0, total_steps=1, warmup_steps=0, min_ratio=0.1), 0.1),
        "Single-update schedule did not follow its documented floor policy",
    )
    for arguments in ((-1, 10, 2, 0.1), (0, 0, 0, 0.1), (0, 10, 10, 0.1), (0, 10, 2, 1.1)):
        step, total, warmup, minimum = arguments
        expect_error(
            lambda step=step, total=total, warmup=warmup, minimum=minimum: subject.lr_factor(
                step, total_steps=total, warmup_steps=warmup, min_ratio=minimum
            ),
            ValueError,
            "Invalid scheduler arguments must reject",
        )
    return {
        "first_warmup_factors": factors[:5],
        "first_cosine_factor": factors[5],
        "cosine_midpoint": factors[middle],
        "last_factor": factors[-1],
        "one_decay_update_and_single_update_policy_passed": True,
    }


def reference_validation(batches, times, augmenter, seed):
    sums = [
        dict.fromkeys(
            (
                "loss",
                "loss_obs_to_action",
                "loss_action_to_obs",
                "accuracy_obs_to_action",
                "accuracy_action_to_obs",
                "positive_score",
                "negative_score",
                "score_gap",
                "temperature",
            ),
            0.0,
        )
        for _ in times
    ]
    sample_count = sum(len(batch["actions"]) for batch in batches)
    for batch in batches:
        epsilon = augmenter.augment_validation_at_time(batch, 0.0, seed=seed)["noise_epsilon"]
        observation = observation_features(batch["head"], batch["wrist"])
        size = len(batch["actions"])
        targets = torch.arange(size)
        identity = torch.eye(size, dtype=torch.bool)
        for level, time in enumerate(times):
            noisy = (1 - time) * batch["actions"] + time * epsilon
            action = action_features(noisy, time)
            scores = observation @ action.T
            logits = scores / torch.tensor(0.4)
            row = functional.cross_entropy(logits, targets)
            column = functional.cross_entropy(logits.T, targets)
            positive, negative = scores.diagonal().mean(), scores[~identity].mean()
            values = {
                "loss": (row + column) / 2,
                "loss_obs_to_action": row,
                "loss_action_to_obs": column,
                "accuracy_obs_to_action": (logits.argmax(1) == targets).float().mean(),
                "accuracy_action_to_obs": (logits.argmax(0) == targets).float().mean(),
                "positive_score": positive,
                "negative_score": negative,
                "score_gap": positive - negative,
                "temperature": torch.tensor(0.4),
            }
            for name, value in values.items():
                sums[level][name] += size * float(value)
    levels = [{name: total / sample_count for name, total in sums_at_time.items()} for sums_at_time in sums]
    weights = augmenter.probabilities.tolist()
    expected = {
        f"val/{name}": sum(weight * level[name] for weight, level in zip(weights, levels, strict=True))
        for name in levels[0]
    }
    expected.update(
        {f"val_t{index:02d}/{name}": value for index, level in enumerate(levels) for name, value in level.items()}
    )
    expected["val/chance_accuracy"] = len(batches) / sample_count
    expected["val/uninformed_loss"] = (
        sum(len(batch["actions"]) * math.log(len(batch["actions"])) for batch in batches) / sample_count
    )
    expected["val/grid_mean_loss"] = sum(level["loss"] for level in levels) / len(levels)
    late = [level["loss"] for time, level in zip(times, levels, strict=True) if 0 < time <= 0.4]
    expected["val/late_loss"] = sum(late) / len(late)
    expected["val/samples"] = sample_count
    return expected


def verify_validation(subject):
    # Unequal batch sizes expose accidental equal-weight averaging of the tail.
    batches = [make_batch([0, 1, 2]), make_batch([3, 4])]
    snapshots = [
        {key: value.clone() if isinstance(value, torch.Tensor) else deepcopy(value) for key, value in batch.items()}
        for batch in batches
    ]
    times = [level / 10 for level in range(10)]
    settings = {"validation_times": times, "seed": 42}
    model, augmenter = FakeCPM(), RecordingNoise()
    loader = ToyCleanLoader(batches)
    model.train()
    rng_before = subject.capture_rng_state()
    result = subject.validate(model, loader, augmenter, settings, torch.device("cpu"), "fp32")
    assert_rng_equal(rng_before, subject.capture_rng_state())
    require(model.training, "Validation did not restore initial train mode")
    require(model.encode_calls == 2, "Observation was not encoded exactly once per batch")
    require(len(model.action_encoder.records) == 20, "Validation did not evaluate all ten times for each group")
    require(loader.iterations == 1, "Validation iterated the clean loader separately for noise times")
    expected_ids = {identity for batch in batches for identity in batch["sample_ids"]}
    assert_fixed_epsilon(augmenter.records, expected_ids)
    epsilon_by_id = {
        identity: record["epsilon"][row]
        for record in augmenter.records
        for row, identity in enumerate(record["sample_ids"])
    }
    for batch_index, batch in enumerate(batches):
        epsilon = torch.stack([epsilon_by_id[identity] for identity in batch["sample_ids"]])
        for level, time in enumerate(times):
            record = model.action_encoder.records[batch_index * 10 + level]
            require(record["flow_time"] == time, "Action encoder received the wrong validation time")
            require(
                torch.equal(record["actions"], (1 - time) * batch["actions"] + time * epsilon),
                "Actual action-encoder inputs did not reuse the same per-ID epsilon across times",
            )
    expected = reference_validation(batches, times, FlowActionNoise(augmenter.config), 42)
    for name, value in expected.items():
        require(
            name in result and math.isclose(result[name], value, rel_tol=1e-6, abs_tol=1e-7),
            f"Wrong validation statistic {name}: observed={result.get(name)}, expected={value}",
        )
    require(
        not math.isclose(result["val/loss"], result["val/grid_mean_loss"], abs_tol=1e-5),
        "Fixture does not distinguish geometric weighting from uniform-grid loss",
    )
    repeated_model = FakeCPM().eval()
    repeated = subject.validate(repeated_model, loader, RecordingNoise(), settings, torch.device("cpu"), "fp32")
    require(not repeated_model.training, "Validation changed an initially-eval model to train mode")
    require(
        all(repeated[key] == result[key] for key in result if key != "val/seconds"),
        "Fixed validation metrics did not reproduce exactly",
    )
    limited_model = FakeCPM().train()
    limited_rng = subject.capture_rng_state()
    limited = subject.validate(
        limited_model, loader, RecordingNoise(), settings, torch.device("cpu"), "fp32", max_batches=1
    )
    assert_rng_equal(limited_rng, subject.capture_rng_state())
    require(
        limited_model.training and limited_model.encode_calls == 1 and limited["val/samples"] == 3,
        "max_batches did not limit validation to the first group",
    )
    failure_model = FakeCPM(fail_on_score=True).train()
    failure_rng = subject.capture_rng_state()
    expect_error(
        lambda: subject.validate(failure_model, loader, RecordingNoise(), settings, torch.device("cpu"), "fp32"),
        FixtureError,
        "Deliberate fixture failure was swallowed",
    )
    assert_rng_equal(failure_rng, subject.capture_rng_state())
    require(failure_model.training, "Validation exception did not restore model train mode")
    for batch, saved in zip(batches, snapshots, strict=True):
        require(set(batch) == set(saved), "Validation mutated the clean batch dictionary")
        for key, original in saved.items():
            equal = torch.equal(batch[key], original) if isinstance(original, torch.Tensor) else batch[key] == original
            require(equal, f"Validation mutated input {key}")
    return {
        "sample_count": result["val/samples"],
        "batch_sizes": [3, 2],
        "encoded_observation_batches": model.encode_calls,
        "weighted_loss": result["val/loss"],
        "uniform_grid_loss": result["val/grid_mean_loss"],
        "per_time_losses": [result[f"val_t{level:02d}/loss"] for level in range(10)],
        "all_expected_metrics_match_independent_reference": True,
        "fixed_id_epsilon_reused_at_every_time": True,
        "train_and_eval_modes_restored": True,
        "global_rng_restored_on_success_and_failure": True,
        "max_batches_checked": True,
        "clean_inputs_unchanged": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trainer",
        type=Path,
        default=Path("/path/to/guidence-touch/src/openpi/touchguide/train.py"),
    )
    parser.add_argument("--report", type=Path, default=Path("/tmp/cpm_trainer_validation.json"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    subject = load_trainer(args.trainer)
    report = {
        "status": "passed",
        "scope": "independent_cpu_trainer_fixture",
        "trainer": str(args.trainer),
        "trainer_sha256": hashlib.sha256(args.trainer.read_bytes()).hexdigest(),
        "weighted_metrics": verify_weighted_metrics(subject),
        "scheduler": verify_scheduler(subject),
        "validation": verify_validation(subject),
        "real_data_read": False,
        "pretrained_model_loaded": False,
        "optimizer_updates": 0,
        "formal_training_started": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
