"""Exercise the installed trainer on a tiny CPU fake CPM; never train actual CPM."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
from pathlib import Path
import random
import tempfile
from typing import Any, ClassVar
from unittest import mock

import numpy as np
import torch
from torch import nn
from torch.nn import functional
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import default_collate

from openpi.touchguide import train as trainer
from openpi.touchguide.batching import EpochIndexBatchSampler
from openpi.touchguide.noise import FlowActionNoise
from openpi.touchguide.noise import NoiseAugmentedBatchLoader
from openpi.touchguide.noise import NoiseConfig

DEFAULT_ROOT = Path(
    "/path/to/datasets/sim/poker-draw/pi05_trex/touchguide/cpm_datasets/poker_0920_200_touchguide_v1"
)


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
        action = self.action_encoder(batch["noisy_actions"], batch["flow_time"])
        return trainer.output_from_embeddings(self, observation, action)

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


def fixture(root: Path, output: Path):
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    model = TinyFakeCPM()
    sampler = EpochIndexBatchSampler(12, 4, seed=123, drop_last=True)
    train_clean = TinyCleanLoader(TinyDataset("train", 12), batch_sampler=sampler, collate_fn=collate, num_workers=0)
    val = TinyCleanLoader(
        TinyDataset("val", 10),
        batch_sampler=[list(range(4)), list(range(4, 8)), [8, 9]],
        collate_fn=collate,
        num_workers=0,
    )
    augmenter = FlowActionNoise(NoiseConfig())
    train = NoiseAugmentedBatchLoader(train_clean, augmenter, split="train", seed=123)
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
        run_name="tiny-cpu-fixture",
    )
    return args, model, train, val, settings


def checkpoint(path: Path) -> dict[str, Any]:
    # All paths here were generated by the same synthetic regression in /tmp.
    return torch.load(path, map_location="cpu", weights_only=False)


def equal_state(left: Any, right: Any) -> None:
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal_state(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            equal_state(a, b)
    else:
        assert left == right


def run_case(
    args, model, train, val, settings, *, interrupt_step: int | None = None, interrupt_after_validation: bool = False
) -> int:
    original_save = trainer.save_checkpoint
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
            raise KeyboardInterrupt("synthetic completed-checkpoint interruption")

    interrupted = False
    try:
        with (
            mock.patch.object(trainer.torch.optim, "AdamW", CountingAdamW),
            mock.patch.object(trainer, "save_checkpoint", side_effect=save_and_interrupt),
        ):
            trainer.train_model(args, model, train, val, settings, torch.device("cpu"))
    except KeyboardInterrupt:
        interrupted = True
    finally:
        for handler in set(trainer.LOGGER.handlers) - existing_handlers:
            trainer.LOGGER.removeHandler(handler)
            handler.close()
    assert interrupted == (interrupt_step is not None)
    assert len(CountingAdamW.instances) == before_instances + 1
    return CountingAdamW.instances[-1].updates


def metrics(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in (path / "metrics.jsonl").read_text().splitlines()]
    return [{key: value for key, value in row.items() if not key.endswith("/seconds")} for row in rows]


def verify_complete(output: Path, model: TinyFakeCPM) -> dict[str, Any]:
    assert {path.name for path in output.glob("*.pt")} == {
        "step_000002.pt",
        "step_000004.pt",
        "step_000006.pt",
        "best.pt",
        "last.pt",
    }
    last = checkpoint(output / "last.pt")
    progress = last["progress"]
    assert progress["epoch"] == 2
    assert progress["batch_in_epoch"] == 0
    assert progress["global_step"] == 6
    assert last["scheduler_state"]["last_epoch"] == 6
    assert progress["interval_metrics"]["count"] == 0
    assert progress["epoch_metrics"]["count"] == 0
    assert last["run"] is None
    assert all(float(state["step"]) == 6 for state in last["optimizer_state"]["state"].values())
    equal_state(model.state_dict(), last["model_state"])
    expected_progress = [(2, 0, 2), (4, 1, 1), (6, 1, 3)]
    for step, epoch, consumed in expected_progress:
        saved = checkpoint(output / f"step_{step:06d}.pt")["progress"]
        assert saved["global_step"] == step
        assert saved["epoch"] == epoch
        assert saved["batch_in_epoch"] == consumed
    rows = metrics(output)
    assert [row["step"] for row in rows if "train/loss" in row] == list(range(1, 7))
    validation = [row for row in rows if "val/loss" in row]
    assert [row["step"] for row in validation] == [3, 6]
    assert all(row["val/samples"] == 10 for row in validation)
    best_row = min(validation, key=lambda row: row["val/loss"])
    assert progress["best_val"] == best_row["val/loss"]
    assert progress["best_step"] == best_row["step"]
    best_progress = checkpoint(output / "best.pt")["progress"]
    assert best_progress["global_step"] == best_row["step"]
    assert best_progress["batch_in_epoch"] == 0
    assert best_progress["best_val"] == best_row["val/loss"]
    return last


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    torch.set_num_threads(1)
    logging.basicConfig(level=logging.WARNING)
    reports = []
    with tempfile.TemporaryDirectory(prefix="cpm-trainer-integration-") as directory_name:
        directory = Path(directory_name)
        continuous = fixture(root, directory / "continuous")
        initial = copy.deepcopy(continuous[1].state_dict())
        assert run_case(*continuous) == 6
        reference = verify_complete(continuous[0].output_dir, continuous[1])
        assert any(not torch.equal(value, initial[key]) for key, value in reference["model_state"].items())
        equal_state(
            continuous[1].vision.state_dict(),
            {key.removeprefix("vision."): value for key, value in initial.items() if key.startswith("vision.")},
        )
        for name, stop, after_validation, resume_name in (
            ("mid_epoch", 2, False, "step_000002.pt"),
            ("epoch_boundary", 3, True, "last.pt"),
            ("final_update_before_validation", 6, False, "step_000006.pt"),
        ):
            interrupted = fixture(root, directory / name)
            assert run_case(*interrupted, interrupt_step=stop, interrupt_after_validation=after_validation) == stop
            saved = checkpoint(interrupted[0].output_dir / resume_name)
            assert saved["progress"]["global_step"] == stop
            resumed = fixture(root, interrupted[0].output_dir)
            resumed[0].resume = resumed[0].output_dir / resume_name
            assert run_case(*resumed) == 6 - stop
            actual = verify_complete(resumed[0].output_dir, resumed[1])
            for key in ("contract", "progress", "model_state", "optimizer_state", "scheduler_state"):
                equal_state(reference[key], actual[key])
            assert metrics(continuous[0].output_dir) == metrics(resumed[0].output_dir)
            reports.append(
                {
                    "case": name,
                    "updates_before_interrupt": stop,
                    "updates_after_resume": 6 - stop,
                    "final_weights_optimizer_scheduler_loss_and_progress_exact_match": True,
                }
            )
        print(
            json.dumps(
                {
                    "status": "passed",
                    "scope": "tiny_fake_cpm_cpu_train_loop_not_actual_cpm_training",
                    "continuous_optimizer_updates": 6,
                    "epochs": 2,
                    "batches_per_epoch": 3,
                    "validation_includes_two_sample_tail": True,
                    "checkpoint_schedule_verified": True,
                    "wandb_mode": "disabled",
                    "cases": reports,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
