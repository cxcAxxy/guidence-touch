"""Batch aligned CPM examples for diagonal, symmetric contrastive matching.

This module does not construct scores, losses, noisy actions or a model. All
action columns retain their stored values in their own observation reference.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Iterator, Sequence
from functools import partial
import hashlib
import json
import operator
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from openpi.touchguide.data import CPMDataset


def collate_matched_samples(
    samples: Sequence[dict[str, Any]], *, include_auxiliary: bool = False
) -> dict[str, Any]:
    """Stack examples once; diagonal identities define both matching directions."""
    size = len(samples)
    if size < 2:
        raise ValueError("Contrastive batches require at least two matched samples")
    ids = [sample["id"] for sample in samples]
    metadata = [dict(sample["metadata"]) for sample in samples]
    identities = [
        (
            item["split"],
            item["episode_index"],
            item["frame_index"],
            item.get("pair_kind", 0),
            item.get("variant_index", -1),
        )
        for item in metadata
    ]
    if len(set(ids)) != size or len(set(identities)) != size:
        raise ValueError("Repeated samples cannot be assigned negative pairing labels")
    if len({item["split"] for item in metadata}) != 1:
        raise ValueError("A batch must not mix training and validation samples")
    if any(
        "negative_actions" in sample or "negative_ids" in sample for sample in samples
    ):
        raise ValueError(
            "Legacy offline negatives are not supported by paper-style batching"
        )
    image_presence = [
        {key for key in ("head", "wrist") if key in sample} for sample in samples
    ]
    if any(keys != image_presence[0] for keys in image_presence) or image_presence[
        0
    ] not in (set(), {"head", "wrist"}):
        raise ValueError(
            "All samples must consistently contain both RGB views or neither"
        )
    shapes = {"tactile_map_raw": (5, 3, 7, 5), "actions": (30, 27)}
    if image_presence[0]:
        shapes.update(head=(3, 224, 224), wrist=(3, 224, 224))
    if include_auxiliary:
        shapes.update(state_raw=(27,), contact_fingers=(5,))
    batch = {}
    for key, shape in shapes.items():
        values = [sample[key] for sample in samples]
        if any(
            not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
            for value in values
        ):
            raise ValueError(f"Invalid sample tensor shape for {key}, expected {shape}")
        batch[key] = torch.stack(values)
    positive = torch.eye(size, dtype=torch.bool)
    batch.update(
        targets=torch.arange(size, dtype=torch.long),
        positive_mask=positive,
        negative_mask=~positive,
        sample_ids=ids,
        metadata=metadata,
    )
    return batch


def iter_index_batches(
    indices: Iterable[int], batch_size: int, *, drop_last: bool = False
) -> Iterator[list[int]]:
    """Group distinct indices; a singleton tail cannot supply contrastive negatives."""
    batch_size = operator.index(batch_size)
    if batch_size < 2:
        raise ValueError("batch_size must be at least two")
    values = [operator.index(index) for index in indices]
    if any(index < 0 for index in values) or len(set(values)) != len(values):
        raise ValueError("Batch indices must be nonnegative and unique")
    remainder = len(values) % batch_size
    if not drop_last and remainder == 1:
        raise ValueError(
            "A singleton tail has no negative pairs; change batch_size or use drop_last=True"
        )
    end = len(values) - remainder if drop_last else len(values)
    for start in range(0, end, batch_size):
        yield values[start : min(start + batch_size, end)]


class EpochIndexBatchSampler:
    """One reproducible permutation per epoch, with no replacement."""

    def __init__(
        self,
        size: int,
        batch_size: int,
        *,
        seed: int = 42,
        epoch: int = 0,
        drop_last: bool = True,
    ):
        self.size, self.batch_size = operator.index(size), operator.index(batch_size)
        self.seed, self.epoch = operator.index(seed), operator.index(epoch)
        self.drop_last = bool(drop_last)
        if self.size < 2 or self.batch_size < 2 or self.seed < 0 or self.epoch < 0:
            raise ValueError("Invalid epoch sampler settings")
        if self.drop_last and self.size < self.batch_size:
            raise ValueError("drop_last would produce no training batches")
        if not self.drop_last and self.size % self.batch_size == 1:
            raise ValueError("Epoch sampler would produce a singleton tail")

    def set_epoch(self, epoch: int) -> None:
        if operator.index(epoch) < 0:
            raise ValueError("epoch must be nonnegative")
        self.epoch = operator.index(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch]))
        return iter_index_batches(
            rng.permutation(self.size), self.batch_size, drop_last=self.drop_last
        )

    def __len__(self) -> int:
        return (
            self.size // self.batch_size
            if self.drop_last
            else (self.size + self.batch_size - 1) // self.batch_size
        )


class FixedIndexBatchSampler:
    """Read the saved validation permutation and its fixed batch boundaries."""

    def __init__(self, path: str | Path, size: int):
        self.path = Path(path)
        with np.load(self.path, allow_pickle=False) as saved:
            self.indices = saved["indices"].copy()
            self.offsets = saved["batch_offsets"].copy()
            self.batch_size = int(saved["batch_size"])
            self.seed = int(saved["seed"])
        if (
            self.indices.shape != (size,)
            or not np.issubdtype(self.indices.dtype, np.integer)
            or not np.array_equal(np.sort(self.indices), np.arange(size))
        ):
            raise ValueError(
                "Fixed validation indices must cover each sample exactly once"
            )
        if (
            self.offsets.ndim != 1
            or not np.issubdtype(self.offsets.dtype, np.integer)
            or len(self.offsets) < 2
            or self.offsets[0] != 0
            or self.offsets[-1] != size
        ):
            raise ValueError("Invalid fixed validation batch boundaries")
        lengths = np.diff(self.offsets)
        if (
            self.batch_size < 2
            or np.any(lengths < 2)
            or np.any(lengths > self.batch_size)
        ):
            raise ValueError(
                "Fixed validation batches require between 2 and batch_size samples"
            )

    def __iter__(self) -> Iterator[list[int]]:
        for start, end in zip(self.offsets[:-1], self.offsets[1:], strict=True):
            yield self.indices[start:end].tolist()

    def __len__(self) -> int:
        return len(self.offsets) - 1


def regroup_fixed_validation_batches(
    sampler: FixedIndexBatchSampler, batch_size: int
) -> list[list[int]]:
    """Regroup one saved validation permutation without changing its sample order.

    The prepared index file remains untouched. A singleton remainder borrows
    one sample from the preceding group so every group has a negative pair.
    """
    if not isinstance(sampler, FixedIndexBatchSampler):
        raise TypeError("Validation regrouping requires a saved fixed index sampler")
    batch_size = operator.index(batch_size)
    if batch_size < 2:
        raise ValueError("Validation batch_size must be at least two")
    indices = sampler.indices.tolist()
    batches = list(iter_index_batches(indices, batch_size, drop_last=True))
    tail = indices[len(batches) * batch_size :]
    if len(tail) == 1:
        if not batches or len(batches[-1]) < 3:
            raise ValueError(
                "Cannot regroup a singleton validation tail into batches of at least two"
            )
        tail.insert(0, batches[-1].pop())
    if tail:
        batches.append(tail)
    return batches


class _CheckedIndexBatches:
    def __init__(self, batches: Iterable[Sequence[int]], size: int):
        self.batches, self.size = batches, size

    def __iter__(self) -> Iterator[list[int]]:
        for batch in self.batches:
            values = [operator.index(index) for index in batch]
            if len(values) < 2 or len(set(values)) != len(values):
                raise ValueError(
                    "Each batch must contain at least two distinct indices"
                )
            if any(index < 0 or index >= self.size for index in values):
                raise ValueError("A batch index is outside its dataset split")
            yield values

    def __len__(self) -> int:
        return len(self.batches)


class CPMBatchLoader(DataLoader):
    """A CPU DataLoader using only matched positives and diagonal identity labels."""

    def __init__(
        self,
        dataset: CPMDataset,
        index_batches: Iterable[Sequence[int]],
        *,
        include_auxiliary: bool = False,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        if dataset.include_negatives:
            raise ValueError("CPMBatchLoader requires include_negatives=False")
        self.index_batches = index_batches
        self.include_auxiliary = include_auxiliary
        super().__init__(
            dataset,
            batch_sampler=_CheckedIndexBatches(index_batches, len(dataset)),
            collate_fn=partial(
                collate_matched_samples, include_auxiliary=include_auxiliary
            ),
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    def set_epoch(self, epoch: int) -> None:
        """Change only the training permutation; fixed validation never changes."""
        if not isinstance(self.index_batches, EpochIndexBatchSampler):
            raise ValueError("Only the training epoch sampler supports set_epoch")
        self.index_batches.set_epoch(epoch)

    @classmethod
    def from_config(
        cls,
        root: str | Path,
        split: str = "train",
        *,
        epoch: int = 0,
        include_images: bool | None = None,
        include_auxiliary: bool | None = None,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        if split not in ("train", "val"):
            raise ValueError("split must be train or val")
        root = Path(root).expanduser().resolve()
        config = json.loads((root / f"config/{split}.json").read_text())
        kwargs = dict(config["reader_kwargs"])
        if kwargs["include_negatives"] or config["noise_on_read"]:
            raise ValueError("This loader supports clean matching only")
        if Path(kwargs["root"]).resolve() != root or kwargs["split"] != split:
            raise ValueError("Reader configuration must refer to this entry and split")
        if include_images is not None:
            kwargs["include_images"] = include_images
        dataset = CPMDataset(**kwargs)
        try:
            settings = config["batching"]
            if settings["sampler"] == "epoch_permutation":
                sampler = EpochIndexBatchSampler(
                    len(dataset),
                    settings["batch_size"],
                    seed=settings["seed"],
                    epoch=epoch,
                    drop_last=settings["drop_last"],
                )
            elif settings["sampler"] == "fixed_index_file":
                file = root / settings["index_file"]
                digest = hashlib.sha256(
                    (root / f"index/{split}.npz").read_bytes()
                ).hexdigest()
                if digest != settings["source_index_sha256"]:
                    raise ValueError(
                        "The split index changed after fixed validation batches were created"
                    )
                sampler = FixedIndexBatchSampler(file, len(dataset))
                if sampler.batch_size != settings["batch_size"]:
                    raise ValueError(
                        "Configured batch_size differs from saved validation groups"
                    )
            else:
                raise ValueError("Unsupported batch sampler")
            return cls(
                dataset,
                sampler,
                include_auxiliary=settings["include_auxiliary"]
                if include_auxiliary is None
                else include_auxiliary,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
        except BaseException:
            dataset.close()
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--preview-size", type=int, default=3)
    parser.add_argument("--no-images", action="store_true")
    args = parser.parse_args()
    configured = CPMBatchLoader.from_config(
        args.root, args.split, epoch=args.epoch, include_images=not args.no_images
    )
    try:
        indices = next(iter(configured.index_batches))
        if not 2 <= args.preview_size <= len(indices):
            raise ValueError(
                "preview-size must be between 2 and the first batch length"
            )
        preview = CPMBatchLoader(configured.dataset, [indices[: args.preview_size]])
        batch = next(iter(preview))
        print(
            json.dumps(
                {
                    "split": args.split,
                    "sample_ids": batch["sample_ids"],
                    "tensors": {
                        key: list(value.shape)
                        for key, value in batch.items()
                        if isinstance(value, torch.Tensor)
                    },
                    "targets": batch["targets"].tolist(),
                    "positive_mask": batch["positive_mask"].tolist(),
                    "noise_added": False,
                    "model_loaded": False,
                    "training_started": False,
                },
                indent=2,
            )
        )
    finally:
        configured.dataset.close()


if __name__ == "__main__":
    main()
