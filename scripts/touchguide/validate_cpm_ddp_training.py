"""Two-rank CPU Gloo training/resume fixtures; never load or train the actual CPM."""

from __future__ import annotations

import argparse
import copy
from datetime import timedelta
import json
import logging
import math
import os
from pathlib import Path
import random
import tempfile
from typing import Any, ClassVar
from unittest import mock

import numpy as np
import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as multiprocessing
from torch.nn import functional
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import default_collate

from openpi.touchguide import train as single_trainer
from openpi.touchguide import train_distributed as trainer
from openpi.touchguide.batching import EpochIndexBatchSampler
from openpi.touchguide.distributed import BroadcastFlowActionNoise
from openpi.touchguide.distributed import DistContext
from openpi.touchguide.distributed import GlobalBatchShardSampler
from openpi.touchguide.distributed import capture_local_rng_state
from openpi.touchguide.noise import FlowActionNoise
from openpi.touchguide.noise import NoiseAugmentedBatchLoader
from openpi.touchguide.noise import NoiseConfig

WORLD_SIZE = 2
VAL_GROUPS = [list(range(4)), list(range(4, 8)), [8, 9]]


class TinyDataset(Dataset):
    def __init__(self, split: str, size: int):
        self.split = split
        generator = torch.Generator().manual_seed(800 if split == "train" else 801)
        self.samples = [
            {
                "head": torch.randn(4, generator=generator),
                "wrist": torch.randn(4, generator=generator),
                "tactile_map_raw": torch.randn(5, 3, 7, 5, generator=generator),
                "actions": torch.randn(30, 27, generator=generator),
                "sample_ids": f"synthetic-{split}-{index}",
            }
            for index in range(size)
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.samples[index]


def collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    batch = default_collate(samples)
    size = len(samples)
    positive = torch.eye(size, dtype=torch.bool)
    batch.update(targets=torch.arange(size), positive_mask=positive, negative_mask=~positive)
    return batch


class TinyCleanLoader(DataLoader):
    def set_epoch(self, epoch: int) -> None:
        self.batch_sampler.set_epoch(epoch)


class TinyAction(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(30 * 27 + 1, 16), nn.GELU(), nn.Dropout(0.2), nn.Linear(16, 8))

    def forward(self, actions: torch.Tensor, flow_time: float | torch.Tensor) -> torch.Tensor:
        time = torch.as_tensor(flow_time, dtype=actions.dtype, device=actions.device).expand(len(actions), 1)
        return functional.normalize(self.network(torch.cat([actions.flatten(1), time], dim=1)), dim=-1)


class TinyFakeCPM(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision = nn.Linear(4, 4).requires_grad_(requires_grad=False)
        self.observation = nn.Sequential(nn.Linear(11, 16), nn.GELU(), nn.Dropout(0.25), nn.Linear(16, 8))
        self.action_encoder = TinyAction()
        self.logit_scale = nn.Parameter(torch.tensor(math.log(5.0)))

    @property
    def temperature(self) -> torch.Tensor:
        return self.logit_scale.exp().reciprocal()

    def encode_observation(self, head: torch.Tensor, wrist: torch.Tensor, tactile: torch.Tensor) -> torch.Tensor:
        features = torch.cat([self.vision(head), wrist, tactile.mean(dim=(1, 3, 4))], dim=1)
        return functional.normalize(self.observation(features), dim=-1)

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        observation = self.encode_observation(batch["head"], batch["wrist"], batch["tactile_map_raw"])
        if self.training:
            # Exercise Python and NumPy RNG restoration as well as torch dropout.
            offset = (random.random() + float(np.random.rand())) * 0.001
            observation = functional.normalize(observation + offset, dim=-1)
        action = self.action_encoder(batch["noisy_actions"], batch["flow_time"])
        return single_trainer.output_from_embeddings(self, observation, action)

    def parameter_counts(self) -> dict[str, int]:
        return {
            "total": sum(parameter.numel() for parameter in self.parameters()),
            "trainable": sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad),
            "frozen": sum(parameter.numel() for parameter in self.parameters() if not parameter.requires_grad),
        }


class CountingAdamW(torch.optim.AdamW):
    instances: ClassVar[list[CountingAdamW]] = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.updates = 0
        self.instances.append(self)

    def step(self, *args, **kwargs):
        result = super().step(*args, **kwargs)
        self.updates += 1
        return result


def fixture(root: Path, output: Path, context: DistContext):
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    model = TinyFakeCPM()
    sampler = GlobalBatchShardSampler(
        EpochIndexBatchSampler(12, 4, seed=123, drop_last=True), rank=context.rank, world_size=context.world_size
    )
    train_clean = TinyCleanLoader(TinyDataset("train", 12), batch_sampler=sampler, collate_fn=collate, num_workers=0)
    local_groups = [group for index, group in enumerate(VAL_GROUPS) if index % context.world_size == context.rank]
    val = TinyCleanLoader(TinyDataset("val", 10), batch_sampler=local_groups, collate_fn=collate, num_workers=0)
    assert len(val) == (2 if context.is_primary else 1)
    augmenter = BroadcastFlowActionNoise(NoiseConfig())
    train = NoiseAugmentedBatchLoader(train_clean, augmenter, split="train", seed=123 + context.rank)
    settings = {"validation_times": [level / 10 for level in range(10)], "seed": 123}
    args = argparse.Namespace(
        root=root,
        output_dir=output,
        resume=None,
        epochs=2,
        batch_size=4,
        seed=123,
        lr=0.002,
        weight_decay=0.001,
        warmup_ratio=0.2,
        min_lr_ratio=0.1,
        grad_clip=1.0,
        precision="fp32",
        log_interval=1,
        save_interval=2,
        wandb_mode="disabled",
        wandb_entity="disabled",
        wandb_project="synthetic-only",
        run_name="tiny-gloo-fixture",
        num_workers=0,
    )
    # Production initializes the common model before choosing independent rank RNG.
    random.seed(123 + context.rank)
    np.random.seed(123 + context.rank)
    torch.manual_seed(123 + context.rank)
    return args, model, train, val, settings


def checkpoint(path: Path) -> dict[str, Any]:
    # These trusted paths are created by this synthetic regression under /tmp.
    return torch.load(path, map_location="cpu", weights_only=False)


def equal_state(left: Any, right: Any) -> None:
    if isinstance(left, torch.Tensor):
        assert left.dtype == right.dtype
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal_state(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right)
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            equal_state(a, b)
    else:
        assert left == right


def metrics(output: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    return [{key: value for key, value in row.items() if not key.endswith("/seconds")} for row in rows]


def run_case(
    args,
    model,
    train,
    val,
    settings,
    context: DistContext,
    *,
    interrupt_step: int | None = None,
    interrupt_after_validation: bool = False,
) -> int:
    original_save = trainer.save_checkpoint
    original_validate = trainer.validate
    existing_handlers = set(trainer.LOGGER.handlers)
    before_instances = len(CountingAdamW.instances)

    def save_and_interrupt(path, **kwargs):
        original_save(path, **kwargs)
        progress = kwargs["progress"]
        if (
            interrupt_step is not None
            and path.name == "last.pt"
            and progress["global_step"] == interrupt_step
            and (progress["batch_in_epoch"] == 0) == interrupt_after_validation
        ):
            # primary_action broadcasts this controlled post-save interruption so
            # both ranks stop together after a complete resumable checkpoint.
            raise RuntimeError("synthetic completed-checkpoint interruption")

    def validate_and_compare(*values, **kwargs):
        before_rng = capture_local_rng_state(context.device)
        before_mode = model.training
        result = original_validate(*values, **kwargs)
        equal_state(before_rng, capture_local_rng_state(context.device))
        assert model.training == before_mode
        error = None
        if context.is_primary:
            try:
                full_val = TinyCleanLoader(
                    TinyDataset("val", 10), batch_sampler=VAL_GROUPS, collate_fn=collate, num_workers=0
                )
                expected = single_trainer.validate(
                    model, full_val, FlowActionNoise(train.augmenter.config), settings, context.device, "fp32"
                )
                assert result.keys() == expected.keys()
                for key in result:
                    if not key.endswith("/seconds"):
                        assert math.isclose(result[key], expected[key], rel_tol=1e-12, abs_tol=1e-12), key
                assert result["val/samples"] == 10
                assert math.isclose(result["val/chance_accuracy"], 0.3, abs_tol=1e-12)
                assert math.isclose(
                    result["val/uninformed_loss"], (8 * math.log(4) + 2 * math.log(2)) / 10, abs_tol=1e-12
                )
                equal_state(before_rng, capture_local_rng_state(context.device))
            except Exception as failure:
                error = f"{type(failure).__name__}: {failure}"
        errors = [error]
        dist.broadcast_object_list(errors, src=0)
        if errors[0] is not None:
            raise AssertionError(f"Full-group weighted validation mismatch: {errors[0]}")
        return result

    interrupted = False
    try:
        with (
            mock.patch.object(trainer.torch.optim, "AdamW", CountingAdamW),
            mock.patch.object(trainer, "save_checkpoint", side_effect=save_and_interrupt),
            mock.patch.object(trainer, "validate", side_effect=validate_and_compare),
        ):
            trainer.train_model(args, model, train, val, settings, context)
    except RuntimeError as error:
        if "synthetic completed-checkpoint interruption" not in str(error):
            raise
        interrupted = True
    finally:
        for handler in set(trainer.LOGGER.handlers) - existing_handlers:
            trainer.LOGGER.removeHandler(handler)
            handler.close()
    assert interrupted == (interrupt_step is not None)
    assert len(CountingAdamW.instances) == before_instances + 1
    assert not torch.cuda.is_initialized()
    updates = CountingAdamW.instances[-1].updates
    counts = [None] * context.world_size
    dist.all_gather_object(counts, updates)
    assert counts == [updates] * context.world_size
    return updates


def verify_complete(output: Path, model: TinyFakeCPM, context: DistContext) -> dict[str, Any]:
    assert {path.name for path in output.glob("*.pt")} == {
        "step_000002.pt",
        "step_000004.pt",
        "step_000006.pt",
        "best.pt",
        "last.pt",
    }
    last = checkpoint(output / "last.pt")
    progress = last["progress"]
    assert (progress["epoch"], progress["batch_in_epoch"], progress["global_step"]) == (2, 0, 6)
    assert last["scheduler_state"]["last_epoch"] == 6
    assert progress["interval_metrics"]["count"] == 0
    assert progress["epoch_metrics"]["count"] == 0
    assert last["run"] is None
    assert all(float(state["step"]) == 6 for state in last["optimizer_state"]["state"].values())
    assert last["contract"]["world_size"] == 2
    assert last["contract"]["global_batch_size"] == 4
    assert last["contract"]["local_batch_size"] == 2
    assert len(progress["distributed_rng_states"]) == 2
    assert all(state["torch_cuda"] == [] for state in progress["distributed_rng_states"])
    equal_state(last["rng_state"], progress["distributed_rng_states"][0])
    assert not torch.equal(
        progress["distributed_rng_states"][0]["torch_cpu"], progress["distributed_rng_states"][1]["torch_cpu"]
    )
    equal_state(model.state_dict(), last["model_state"])
    equal_state(capture_local_rng_state(context.device), progress["distributed_rng_states"][context.rank])
    for step, epoch, consumed in ((2, 0, 2), (4, 1, 1), (6, 1, 3)):
        saved = checkpoint(output / f"step_{step:06d}.pt")["progress"]
        assert (saved["global_step"], saved["epoch"], saved["batch_in_epoch"]) == (step, epoch, consumed)
    rows = metrics(output)
    train_rows = [row for row in rows if "train/loss" in row]
    assert [row["step"] for row in train_rows] == list(range(1, 7))
    assert all(row["train/interval_samples"] == 4 for row in train_rows)
    validation = [row for row in rows if "val/loss" in row]
    assert [row["step"] for row in validation] == [3, 6]
    assert all(row["val/samples"] == 10 for row in validation)
    assert all(row["train_epoch/samples"] == 12 for row in validation)
    best_row = min(validation, key=lambda row: row["val/loss"])
    assert progress["best_val"] == best_row["val/loss"]
    assert progress["best_step"] == best_row["step"]
    best_progress = checkpoint(output / "best.pt")["progress"]
    assert best_progress["global_step"] == best_row["step"]
    assert best_progress["batch_in_epoch"] == 0
    assert best_progress["best_val"] == best_row["val/loss"]
    return last


def prepare_metadata(root: Path) -> None:
    for relative in (
        "config/cpm.json",
        "config/train.json",
        "config/val.json",
        "meta/tactile_scale.json",
        "meta/action_norm_stats.json",
        "meta/entry_config.json",
        "meta/splits.json",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"fixture": "tiny_cpu_gloo", "path": relative}) + "\n")
    (root / "index").mkdir()
    np.savez(root / "index/train.npz", indices=np.arange(12))
    np.savez(root / "index/val.npz", indices=np.arange(10))
    np.savez(root / "index/val_batches.npz", indices=np.arange(10), batch_offsets=np.array([0, 4, 8, 10]))


def worker(rank: int, directory_name: str) -> None:
    directory = Path(directory_name)
    torch.set_num_threads(1)
    logging.basicConfig(level=logging.WARNING)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{directory / 'rendezvous'}",
        rank=rank,
        world_size=WORLD_SIZE,
        timeout=timedelta(seconds=90),
    )
    context = DistContext(rank, WORLD_SIZE, rank, torch.device("cpu"))
    try:
        root = directory / "fixture_metadata"
        continuous = fixture(root, directory / "continuous", context)
        initial = copy.deepcopy(continuous[1].state_dict())
        assert run_case(*continuous, context) == 6
        reference = verify_complete(continuous[0].output_dir, continuous[1], context)
        assert any(not torch.equal(value, initial[key]) for key, value in reference["model_state"].items())
        equal_state(
            continuous[1].vision.state_dict(),
            {key.removeprefix("vision."): value for key, value in initial.items() if key.startswith("vision.")},
        )
        reports = []
        for name, stop, after_validation, resume_name in (
            ("mid_epoch", 2, False, "step_000002.pt"),
            ("epoch_boundary", 3, True, "last.pt"),
            ("final_update_before_validation", 6, False, "step_000006.pt"),
        ):
            interrupted = fixture(root, directory / name, context)
            assert (
                run_case(*interrupted, context, interrupt_step=stop, interrupt_after_validation=after_validation)
                == stop
            )
            saved = checkpoint(interrupted[0].output_dir / resume_name)
            assert saved["progress"]["global_step"] == stop
            resumed = fixture(root, interrupted[0].output_dir, context)
            resumed[0].resume = resumed[0].output_dir / resume_name
            assert run_case(*resumed, context) == 6 - stop
            actual = verify_complete(resumed[0].output_dir, resumed[1], context)
            for key in ("contract", "progress", "model_state", "optimizer_state", "scheduler_state", "rng_state"):
                equal_state(reference[key], actual[key])
            assert metrics(continuous[0].output_dir) == metrics(resumed[0].output_dir)
            reports.append(
                {
                    "case": name,
                    "updates_before_interrupt": stop,
                    "updates_after_resume": 6 - stop,
                    "final_model_adam_scheduler_progress_per_rank_rng_and_metrics_exact_match": True,
                }
            )
        result = {
            "rank": rank,
            "status": "passed",
            "synthetic_model_parameters": continuous[1].parameter_counts(),
            "cases": reports,
        }
        (directory / f"rank_{rank}.json").write_text(json.dumps(result, indent=2) + "\n")
    finally:
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=Path("/tmp/cpm_ddp_training_validation.json"))
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "1"
    with tempfile.TemporaryDirectory(prefix="cpm-ddp-training-fixture-", dir="/tmp") as directory_name:
        directory = Path(directory_name)
        prepare_metadata(directory / "fixture_metadata")
        multiprocessing.spawn(worker, args=(directory_name,), nprocs=WORLD_SIZE, join=True)
        ranks = [json.loads((directory / f"rank_{rank}.json").read_text()) for rank in range(WORLD_SIZE)]
    report = {
        "status": "passed",
        "scope": "tiny_fake_cpm_two_process_cpu_gloo_train_and_resume",
        "world_size": WORLD_SIZE,
        "global_batch_size": 4,
        "local_batch_size": 2,
        "epochs": 2,
        "global_updates_per_epoch": 3,
        "continuous_optimizer_updates": 6,
        "validation_group_sizes": [4, 4, 2],
        "validation_groups_per_rank": [2, 1],
        "weighted_validation_matches_unsplit_reference": True,
        "global_training_sample_counts_verified": True,
        "all_rank_rng_restored_exactly": True,
        "checkpoint_schedule_verified": True,
        "rank_zero_checkpoint_and_metric_writes": True,
        "wandb_mode": "disabled",
        "pretrained_weights_loaded": False,
        "actual_cpm_training_started": False,
        "gpu_initialized": False,
        "nas_written": False,
        "ranks": ranks,
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "passed", "report": str(args.report), "world_size": WORLD_SIZE}))


if __name__ == "__main__":
    main()
