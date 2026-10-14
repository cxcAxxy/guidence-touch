#!/usr/bin/env python3
"""CPU-only regression for fixed CPM validation regrouping and old-run contracts."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from openpi.touchguide import train
from openpi.touchguide import train_distributed
from openpi.touchguide.batching import CPMBatchLoader
from openpi.touchguide.batching import EpochIndexBatchSampler
from openpi.touchguide.batching import FixedIndexBatchSampler
from openpi.touchguide.batching import regroup_fixed_validation_batches
from openpi.touchguide.distributed import DistContext
from openpi.touchguide.noise import NoiseConfig


class DummyDataset:
    include_negatives = False

    def __init__(self, split: str, size: int):
        self.split = split
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int):
        raise AssertionError(f"CPU batching regression must not decode sample {index}")


def make_fixed(path: Path, indices: np.ndarray, offsets: list[int], batch_size: int) -> FixedIndexBatchSampler:
    np.savez(path, indices=indices, batch_offsets=np.asarray(offsets), batch_size=batch_size, seed=42)
    return FixedIndexBatchSampler(path, len(indices))


def flatten(groups: list[list[int]]) -> list[int]:
    return [index for group in groups for index in group]


def main() -> None:
    with TemporaryDirectory(prefix="touchguide-val-rebatch-") as directory:
        root = Path(directory)
        (root / "index").mkdir()
        (root / "config").mkdir()
        (root / "meta").mkdir()
        indices = np.random.default_rng(42).permutation(4372)
        saved = make_fixed(root / "index/val_batches.npz", indices, [*range(0, 4372, 64), 4372], 64)
        regrouped = regroup_fixed_validation_batches(saved, 128)
        assert [len(group) for group in regrouped] == [128] * 34 + [20]
        assert flatten(regrouped) == indices.tolist()
        assert regroup_fixed_validation_batches(saved, 128) == regrouped
        assert len(saved) == 69
        assert saved.batch_size == 64
        assert flatten(list(saved)) == indices.tolist()

        singleton = make_fixed(root / "singleton.npz", np.arange(129), [0, 65, 129], 65)
        assert [len(group) for group in regroup_fixed_validation_batches(singleton, 128)] == [127, 2]
        assert flatten(regroup_fixed_validation_batches(singleton, 128)) == list(range(129))
        pair = make_fixed(root / "pair.npz", np.arange(2), [0, 2], 2)
        assert regroup_fixed_validation_batches(pair, 128) == [[0, 1]]
        odd = make_fixed(root / "odd.npz", np.arange(3), [0, 3], 3)
        try:
            regroup_fixed_validation_batches(odd, 2)
        except ValueError:
            pass
        else:
            raise AssertionError("An impossible singleton tail must be rejected")

        noise = asdict(NoiseConfig())
        for split in ("train", "val"):
            config = {
                "noise_augmentation": {
                    **noise,
                    "application_stage": "after_clean_batch_collation",
                    "mode": "random_per_batch" if split == "train" else "fixed_per_sample_id",
                    "validation_times": [level / 10 for level in range(10)],
                    "seed": 42,
                }
            }
            (root / f"config/{split}.json").write_text(json.dumps(config))
            np.savez(root / f"index/{split}.npz", indices=np.arange(4))
        (root / "config/cpm.json").write_text("{}")
        for name in (
            "tactile_scale.json",
            "action_norm_stats.json",
            "entry_config.json",
            "splits.json",
        ):
            (root / "meta" / name).write_text("{}")

        train_set = DummyDataset("train", 256)
        val_set = DummyDataset("val", 4372)

        def configured_loader(_root, split, **_kwargs):
            if split == "train":
                return CPMBatchLoader(train_set, EpochIndexBatchSampler(256, 64))
            return CPMBatchLoader(val_set, saved)

        args = SimpleNamespace(
            root=root,
            batch_size=128,
            val_batch_size=128,
            seed=42,
            num_workers=0,
            epochs=200,
            lr=1e-5,
            weight_decay=1e-4,
            warmup_ratio=0.05,
            min_lr_ratio=0.1,
            grad_clip=1.0,
            precision="bf16",
        )
        with patch.object(CPMBatchLoader, "from_config", side_effect=configured_loader):
            single_train, single_val, _ = train.build_loaders(args)
            assert list(single_val.index_batches) == regrouped
            assert len(single_val) == 35
            distributed_groups = {}
            for rank in range(16):
                context = DistContext(rank, 16, rank, torch.device("cpu"))
                _, val, _ = train_distributed.build_loaders(args, context)
                for local_index, group in enumerate(val.index_batches):
                    global_index = rank + local_index * 16
                    distributed_groups[global_index] = group
            assert [distributed_groups[index] for index in range(35)] == regrouped

            old_args = SimpleNamespace(**{**vars(args), "val_batch_size": None})
            old_train, old_val, _ = train.build_loaders(old_args)
            assert [len(group) for group in old_val.index_batches] == [64] * 68 + [20]
            old_contract = train.make_contract(old_args, old_train)
            assert "validation_batch_size" not in old_contract
            new_contract = train.make_contract(args, single_train)
            assert new_contract.pop("validation_batch_size") == 128
            assert old_contract == new_contract

    print("Fixed validation 128 regrouping, 16-rank partition, and old contract compatibility: PASS")


if __name__ == "__main__":
    main()
