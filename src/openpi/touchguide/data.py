"""Read the poker TouchGuide CPM dataset without building or loading a model.

CPM inputs are the two current RGB images, the current ``tactile_map_raw``
array, and candidate action chunks. There is no tactile history, F6 feature,
or independent CPM state feature. ``state_raw`` is auxiliary provenance and
is used only to rebase negative arm actions to the anchor observation.

Positive actions are the stored 20k-normalized arm-delta-7/hand-absolute-20
chunks. Negative actions are rebuilt from donor absolute targets relative to
the anchor state, then normalized with the exact same 20k quantile statistics.
Dataset reads never add noise; training may call ``sample_noisy_pairs``.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from collections.abc import Sequence
import json
import math
import operator
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

DEFAULT_ROOT = Path(
    "/path/to/datasets/sim/poker-draw/pi05_trex/touchguide/"
    "cpm_datasets/poker_0920_200_v1"
)
ACTION_HORIZON = 30
ACTION_DIM = 27
NEGATIVE_POOL_SIZE = 8


def _decode_video_frame(path: str, frame_index: int, fps: float) -> np.ndarray:
    """Read the nearest frame by episode-relative video time using bounded RAM."""
    import av  # noqa: PLC0415

    query_s = frame_index / fps
    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        if stream.average_rate is not None:
            video_fps = float(stream.average_rate)
            if not math.isclose(video_fps, fps, rel_tol=1e-3, abs_tol=1e-3):
                raise ValueError(
                    f"Video FPS {video_fps} disagrees with dataset FPS {fps}: {path}"
                )
        if stream.time_base is None:
            raise ValueError(f"Video has no time base: {path}")
        seek_s = max(0.0, query_s - 1.0)
        container.seek(
            int(seek_s / stream.time_base),
            stream=stream,
            backward=True,
            any_frame=False,
        )
        best_frame = None
        best_distance = math.inf
        best_time = None
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            time_base = (
                frame.time_base if frame.time_base is not None else stream.time_base
            )
            frame_s = float(frame.pts * time_base)
            distance = abs(frame_s - query_s)
            if distance < best_distance:
                best_frame, best_distance, best_time = frame, distance, frame_s
            if frame_s >= query_s:
                break
        tolerance_s = 0.5 / fps + 1e-6
        if best_frame is None or best_distance > tolerance_s:
            raise ValueError(
                f"Cannot match frame {frame_index} at {query_s:.9f}s in {path}: "
                f"nearest={best_time}, difference={best_distance}, tolerance={tolerance_s}"
            )
        return best_frame.to_ndarray(format="rgb24")


def _image_tensor(path: str, frame_index: int, fps: float) -> torch.Tensor:
    # Import only when images are requested, so numerical checks need no decoder
    # or OpenPI installation. This is OpenPI's CPU, bilinear resize-with-padding.
    from openpi_client import image_tools  # noqa: PLC0415

    image = _decode_video_frame(path, frame_index, fps)
    image = image_tools.resize_with_pad(image, 224, 224)
    image = np.ascontiguousarray(image.transpose(2, 0, 1), dtype=np.float32)
    return torch.from_numpy(image).div_(127.5).sub_(1.0)


class CPMDataset(Dataset):
    """Per-sample H5 reader with at most four open episode handles per worker.

    ``include_images=False`` skips all video access. ``include_negatives=False``
    skips loading negative indices and omits the ``negative_actions`` key.
    Training randomly chooses K distinct members of each stored eight-donor
    pool. Set ``randomize_negatives=False`` for a fixed first-K selection;
    validation is always fixed. Seed PyTorch/DataLoader normally to reproduce
    training choices. Returned raw tactile values are never clipped or scaled.

    The ``metadata`` dictionary and ``state_raw`` are auxiliary records, not
    CPM observation features. Each negative ID refers to its split index, with
    the donor episode, H5 row, and frame also returned in metadata.
    """

    def __init__(
        self,
        root: str | Path = DEFAULT_ROOT,
        split: str = "train",
        *,
        include_images: bool = True,
        include_negatives: bool = False,
        num_negatives: int = NEGATIVE_POOL_SIZE,
        randomize_negatives: bool = True,
        max_open_h5: int = 4,
    ) -> None:
        if split not in ("train", "val"):
            raise ValueError("split must be 'train' or 'val'")
        if not 1 <= num_negatives <= NEGATIVE_POOL_SIZE:
            raise ValueError("num_negatives must be between 1 and 8")
        if not 1 <= max_open_h5 <= 4:
            raise ValueError("max_open_h5 must be between 1 and 4")
        self.root = Path(root).expanduser().resolve()
        self.split = split
        self.include_images = bool(include_images)
        self.include_negatives = bool(include_negatives)
        self.num_negatives = operator.index(num_negatives)
        self.randomize_negatives = bool(randomize_negatives) and split == "train"
        self.max_open_h5 = operator.index(max_open_h5)
        self._h5: OrderedDict[int, h5py.File] = OrderedDict()
        self._pid = os.getpid()
        self.manifest = json.loads((self.root / "meta/manifest.json").read_text())
        configuration = self.manifest.get("configuration", {})
        self.fps = float(configuration.get("fps", 30.0))
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("Manifest FPS must be finite and positive")
        for name, expected in (
            ("action_horizon", ACTION_HORIZON),
            ("action_dim", ACTION_DIM),
        ):
            if name in configuration and configuration[name] != expected:
                raise ValueError(
                    f"Unsupported {name}: {configuration[name]} (expected {expected})"
                )
        with np.load(self.root / "index" / f"{split}.npz", allow_pickle=False) as index:
            self.index = {
                name: index[name].copy()
                for name in (
                    "episode_index",
                    "sample_row",
                    "frame_index",
                    "timestamp_s",
                )
            }
        self._length = len(self.index["episode_index"])
        if any(value.shape != (self._length,) for value in self.index.values()):
            raise ValueError(
                "Split index arrays must be one-dimensional with the same length"
            )
        for name in ("episode_index", "sample_row", "frame_index"):
            if not np.issubdtype(self.index[name].dtype, np.integer) or np.any(
                self.index[name] < 0
            ):
                raise ValueError(f"Invalid {name} split index")
        if not np.isfinite(self.index["timestamp_s"]).all():
            raise ValueError("Nonfinite split timestamps")
        self.negative_pool: np.ndarray | None = None
        self._q01: np.ndarray | None = None
        self._q99: np.ndarray | None = None
        if self.include_negatives:
            stats = json.loads((self.root / "meta/action_norm_stats.json").read_text())
            action_stats = stats["norm_stats"]["actions"]
            self._q01 = np.asarray(action_stats["q01"], dtype=np.float32)[
                ..., :ACTION_DIM
            ]
            self._q99 = np.asarray(action_stats["q99"], dtype=np.float32)[
                ..., :ACTION_DIM
            ]
            if self._q01.shape != (ACTION_DIM,) or self._q99.shape != (ACTION_DIM,):
                raise ValueError("Action q01/q99 must contain at least 27 dimensions")
            if (
                not np.isfinite(self._q01).all()
                or not np.isfinite(self._q99).all()
                or np.any(self._q99 < self._q01)
            ):
                raise ValueError("Invalid action quantiles")
            with np.load(
                self.root / "negatives" / f"{split}.npz", allow_pickle=False
            ) as negatives:
                self.negative_pool = negatives["negative_sample_index"].copy()
            if self.negative_pool.shape != (
                self._length,
                NEGATIVE_POOL_SIZE,
            ) or not np.issubdtype(self.negative_pool.dtype, np.integer):
                raise ValueError(
                    "Negative index pool must have shape [split_length, 8] and integer dtype"
                )
            if self.negative_pool.size and (
                self.negative_pool.min() < 0 or self.negative_pool.max() >= self._length
            ):
                raise ValueError("Negative donor indices must refer to the same split")

    def __len__(self) -> int:
        return self._length

    def close(self) -> None:
        for handle in self._h5.values():
            handle.close()
        self._h5.clear()

    def __del__(self) -> None:
        try:  # noqa: SIM105 - best-effort cleanup during interpreter shutdown
            self.close()
        except Exception:
            pass

    def __getstate__(self) -> dict[str, Any]:
        # DataLoader spawn must not pickle h5py objects; fork handles are also
        # dropped below before the first read in each child process.
        state = self.__dict__.copy()
        state["_h5"] = OrderedDict()
        state["_pid"] = None
        return state

    def _episode(self, episode_index: int) -> h5py.File:
        if self._pid != os.getpid():
            self.close()
            self._pid = os.getpid()
        if episode_index in self._h5:
            self._h5.move_to_end(episode_index)
            return self._h5[episode_index]
        if len(self._h5) >= self.max_open_h5:
            _, old = self._h5.popitem(last=False)
            old.close()
        handle = h5py.File(
            self.root / "episodes" / f"episode_{episode_index:06d}.h5", "r"
        )
        self._h5[episode_index] = handle
        return handle

    def _location(self, index: int) -> tuple[int, int, int]:
        return tuple(
            int(self.index[key][index])
            for key in ("episode_index", "sample_row", "frame_index")
        )

    def _normalize_donor(self, absolute: np.ndarray, state: np.ndarray) -> np.ndarray:
        if absolute.shape != (ACTION_HORIZON, ACTION_DIM):
            raise ValueError(f"Invalid donor action shape: {absolute.shape}")
        rebased = np.array(absolute, dtype=np.float32, copy=True)
        rebased[:, :7] -= state[:7]
        # Match OpenPI Normalize(use_quantiles=True), including epsilon and no
        # clipping; only the first seven absolute targets were made deltas.
        return (
            (rebased - self._q01) / (self._q99 - self._q01 + 1e-6) * 2.0 - 1.0
        ).astype(np.float32)

    def __getitem__(self, index: int) -> dict[str, Any]:
        index = operator.index(index)
        if index < 0:
            index += self._length
        if index < 0 or index >= self._length:
            raise IndexError(index)
        episode_index, row, frame_index = self._location(index)
        episode = self._episode(episode_index)
        if int(episode["frame_index"][row]) != frame_index:
            raise ValueError(f"Split index/H5 frame mismatch at {self.split}:{index}")
        state = np.asarray(episode["state_raw"][row], dtype=np.float32)
        actions = np.asarray(episode["actions"][row], dtype=np.float32)
        tactile = np.asarray(episode["tactile_map_raw"][row], dtype=np.float32)
        contacts = np.asarray(episode["contact_fingers"][row], dtype=np.bool_)
        if (
            state.shape != (ACTION_DIM,)
            or actions.shape != (ACTION_HORIZON, ACTION_DIM)
            or tactile.shape != (5, 3, 7, 5)
            or contacts.shape != (5,)
        ):
            raise ValueError(f"Invalid sample shapes at {self.split}:{index}")
        metadata: dict[str, Any] = {
            "split": self.split,
            "sample_index": index,
            "episode_index": episode_index,
            "sample_row": row,
            "frame_index": frame_index,
            "timestamp_s": float(episode["timestamp_s"][row]),
            "source_state_index": int(episode["source_state_index"][row]),
            "source_camera_timestamp_s": float(
                episode["source_camera_timestamp_s"][row]
            ),
        }
        if "pair_kind" in episode:
            metadata["pair_kind"] = int(episode["pair_kind"][row])
            metadata["variant_index"] = int(episode["variant_index"][row])
        sample: dict[str, Any] = {
            "id": f"{self.split}:{index:08d}",
            "actions": torch.from_numpy(actions),
            "tactile_map_raw": torch.from_numpy(tactile),
            "contact_fingers": torch.from_numpy(contacts),
            "state_raw": torch.from_numpy(state),
            "metadata": metadata,
        }
        if self.include_images:
            for key, attr in (("head", "head_video"), ("wrist", "right_wrist_video")):
                video_path = episode.attrs[attr]
                if isinstance(video_path, bytes):
                    video_path = video_path.decode("utf-8")
                sample[key] = _image_tensor(str(video_path), frame_index, self.fps)
        if self.include_negatives:
            donor_indices = self.negative_pool[index]
            if self.randomize_negatives:
                slots = torch.randperm(NEGATIVE_POOL_SIZE)[: self.num_negatives].numpy()
                donor_indices = donor_indices[slots]
            else:
                donor_indices = donor_indices[: self.num_negatives]
            donor_indices = np.asarray(donor_indices, dtype=np.int64)
            if len(set(donor_indices.tolist())) != self.num_negatives:
                raise ValueError(
                    f"Duplicate negative donor indices for {self.split}:{index}"
                )
            negative_actions = []
            locations = []
            for donor_index in donor_indices:
                donor_episode, donor_row, donor_frame = self._location(int(donor_index))
                if (
                    donor_episode == episode_index
                    and abs(donor_frame - frame_index) < ACTION_HORIZON
                ):
                    raise ValueError(
                        f"Negative donor overlaps the anchor window at {self.split}:{index}"
                    )
                absolute = np.asarray(
                    self._episode(donor_episode)["actions_absolute"][donor_row],
                    dtype=np.float32,
                )
                negative_actions.append(self._normalize_donor(absolute, state))
                locations.append((donor_episode, donor_row, donor_frame))
            sample["negative_actions"] = torch.from_numpy(np.stack(negative_actions))
            sample["negative_ids"] = [
                f"{self.split}:{int(donor):08d}" for donor in donor_indices
            ]
            metadata["negative_sample_index"] = torch.from_numpy(donor_indices.copy())
            for slot, name in enumerate(
                (
                    "negative_episode_index",
                    "negative_sample_row",
                    "negative_frame_index",
                )
            ):
                metadata[name] = torch.tensor(
                    [location[slot] for location in locations], dtype=torch.int64
                )
        return sample


def sample_noisy_pairs(
    actions: torch.Tensor,
    negative_actions: torch.Tensor,
    generator: torch.Generator | None = None,
    noise_times: Sequence[float] = (0.0, 0.1, 0.2, 0.3, 0.4),
) -> dict[str, torch.Tensor]:
    """Create flow-noisy positives and negatives using a shared tau per anchor.

    Accept ``[T,D]`` and ``[K,T,D]``, or matching batch prefixes such as
    ``[B,T,D]`` and ``[B,K,T,D]``. Each anchor draws one discrete noise time;
    the positive and all K negatives share it. Epsilon samples are independent
    standard Gaussians with the same dtype/device and distribution. The flow
    convention is ``x_tau = (1-tau)*actions + tau*epsilon``; tau=0 is clean.
    Return keys are ``noisy_actions``, ``noisy_negative_actions`` and ``tau``
    (tau has just the batch-prefix shape, or scalar for an unbatched sample).
    A CPU generator is supported even when input tensors are on a GPU.
    """
    if actions.ndim < 2 or negative_actions.ndim != actions.ndim + 1:
        raise ValueError("Expected actions [...,T,D] and negatives [...,K,T,D]")
    if (
        negative_actions.shape[:-3] != actions.shape[:-2]
        or negative_actions.shape[-2:] != actions.shape[-2:]
        or negative_actions.shape[-3] < 1
    ):
        raise ValueError("Positive and negative action shapes do not match")
    if (
        actions.device != negative_actions.device
        or actions.dtype != negative_actions.dtype
        or not actions.is_floating_point()
    ):
        raise ValueError(
            "Actions and negatives must have the same floating dtype and device"
        )
    levels = tuple(float(level) for level in noise_times)
    if not levels or any(
        not math.isfinite(level) or not 0 <= level <= 1 for level in levels
    ):
        raise ValueError("noise_times must be nonempty finite values between 0 and 1")
    random_device = (
        actions.device if generator is None else torch.device(generator.device)
    )
    batch_shape = actions.shape[:-2]
    choices = torch.randint(
        len(levels), batch_shape, generator=generator, device=random_device
    )
    levels_tensor = torch.tensor(levels, device=random_device, dtype=actions.dtype)
    tau = levels_tensor[choices].to(device=actions.device)
    positive_eps = torch.randn(
        actions.shape, generator=generator, device=random_device, dtype=actions.dtype
    )
    negative_eps = torch.randn(
        negative_actions.shape,
        generator=generator,
        device=random_device,
        dtype=negative_actions.dtype,
    )
    positive_eps = positive_eps.to(device=actions.device)
    negative_eps = negative_eps.to(device=actions.device)
    positive_tau = tau.reshape(*batch_shape, 1, 1)
    negative_tau = tau.reshape(*batch_shape, 1, 1, 1)
    return {
        "noisy_actions": (1.0 - positive_tau) * actions + positive_tau * positive_eps,
        "noisy_negative_actions": (1.0 - negative_tau) * negative_actions
        + negative_tau * negative_eps,
        "tau": tau,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--include-negatives", action="store_true")
    parser.add_argument("--num-negatives", type=int, default=NEGATIVE_POOL_SIZE)
    parser.add_argument("--fixed-negatives", action="store_true")
    parser.add_argument("--no-images", action="store_true")
    args = parser.parse_args()
    dataset = CPMDataset(
        args.root,
        args.split,
        include_images=not args.no_images,
        include_negatives=args.include_negatives,
        num_negatives=args.num_negatives,
        randomize_negatives=not args.fixed_negatives,
    )
    try:
        sample = dataset[args.index]
        result = {
            "root": str(dataset.root),
            "split_length": len(dataset),
            "id": sample["id"],
        }
        result["tensors"] = {
            key: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "min": value.min().item(),
                "max": value.max().item(),
            }
            for key, value in sample.items()
            if isinstance(value, torch.Tensor)
        }
        result["metadata"] = {
            key: value.tolist() if isinstance(value, torch.Tensor) else value
            for key, value in sample["metadata"].items()
        }
        if "negative_ids" in sample:
            result["negative_ids"] = sample["negative_ids"]
        print(json.dumps(result, indent=2))
    finally:
        dataset.close()


if __name__ == "__main__":
    main()
