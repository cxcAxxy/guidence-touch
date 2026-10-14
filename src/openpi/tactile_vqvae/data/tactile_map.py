"""Frame-level three-axis tactile maps from the pi05_trex sidecar."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class TactileMapStats:
    """Global per-axis q01/q99 ranges in normal, tangent-0, tangent-1 order."""

    q01: np.ndarray
    q99: np.ndarray
    eps: float = 1e-6

    def __post_init__(self) -> None:
        q01 = np.asarray(self.q01, dtype=np.float32)
        q99 = np.asarray(self.q99, dtype=np.float32)
        if q01.shape != (3,) or q99.shape != (3,):
            raise ValueError(f"Expected three channel ranges, got {q01.shape}/{q99.shape}")
        object.__setattr__(self, "q01", q01)
        object.__setattr__(self, "q99", q99)

    @classmethod
    def from_root(cls, root: str | Path) -> TactileMapStats:
        with (Path(root) / "meta" / "tactile_stats.json").open() as f:
            payload = json.load(f)["tactile_map"]
        return cls(q01=payload["q01"], q99=payload["q99"])

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize maps shaped [..., 3, 7, 5] into [-1, 1]."""
        x = np.asarray(x, dtype=np.float32)
        if x.shape[-3:] != (3, 7, 5):
            raise ValueError(f"Expected [..., 3, 7, 5], got {x.shape}")
        shape = (1,) * (x.ndim - 3) + (3, 1, 1)
        lo = self.q01.reshape(shape)
        scale = (self.q99 - self.q01).reshape(shape)
        valid = scale > self.eps
        normalized = 2.0 * (x - lo) / np.where(valid, scale, 1.0) - 1.0
        normalized = np.where(valid, normalized, 0.0)
        return np.clip(normalized, -1.0, 1.0).astype(np.float32, copy=False)

    def empty_map(self) -> np.ndarray:
        """Normalized value of a physically zero-valued tactile map."""
        return self.normalize(np.zeros((3, 7, 5), dtype=np.float32))

    def to_dict(self) -> dict:
        return {
            "q01": self.q01.tolist(),
            "q99": self.q99.tolist(),
            "eps": self.eps,
            "channel_order": ["normal", "tangent0", "tangent1"],
            "normalization": "clip(2 * (x-q01)/(q99-q01) - 1); constant channels -> 0",
        }


class TactileMapFrameDataset(Dataset[dict[str, torch.Tensor]]):
    """One frame of five or ten fingertip maps per item."""

    def __init__(
        self,
        root: str | Path,
        *,
        episode_indices: Sequence[int],
        stats: TactileMapStats,
        stride: int = 4,
        hand_mode: str = "right",
    ) -> None:
        self.root = Path(root)
        self.stats = stats
        if stride <= 0:
            raise ValueError("stride must be positive")
        if hand_mode not in {"left", "right", "both"}:
            raise ValueError("hand_mode must be left, right, or both")
        self.hand_mode = hand_mode
        with (self.root / "meta" / "episodes.jsonl").open() as f:
            all_episodes = [json.loads(line) for line in f if line.strip()]
        if any(i < 0 or i >= len(all_episodes) for i in episode_indices):
            raise IndexError("episode_indices contains an out-of-range episode")
        self.episode_indices = list(episode_indices)
        self.episodes = [all_episodes[i] for i in self.episode_indices]
        self.samples = [
            (local_episode, frame)
            for local_episode, episode in enumerate(self.episodes)
            for frame in range(0, int(episode["num_frames"]), stride)
        ]
        if not self.samples:
            raise ValueError("No tactile-map frames found")
        self._cache: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def _load_episode(self, local_episode: int) -> np.ndarray:
        if local_episode in self._cache:
            return self._cache[local_episode]
        import h5py  # noqa: PLC0415

        path = self.root / self.episodes[local_episode]["file"]
        with h5py.File(path, "r") as f:
            maps = np.asarray(f["tactile_map"], dtype=np.float32)
        if maps.ndim != 5 or maps.shape[1:] != (10, 3, 7, 5):
            raise ValueError(f"{path}: expected tactile_map [N, 10, 3, 7, 5], got {maps.shape}")
        self._cache[local_episode] = maps
        return maps

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        local_episode, frame = self.samples[index]
        raw = self._load_episode(local_episode)[frame]
        if self.hand_mode == "left":
            raw = raw[:5]
        elif self.hand_mode == "right":
            raw = raw[5:]
        contact = np.any(raw != 0.0, axis=(1, 2, 3))
        return {
            "maps": torch.from_numpy(self.stats.normalize(raw)),
            "contact": torch.from_numpy(contact),
        }


def build_map_train_val_datasets(
    root: str | Path,
    *,
    stride: int = 4,
    val_ratio: float = 0.1,
    seed: int = 42,
    hand_mode: str = "right",
) -> tuple[TactileMapFrameDataset, TactileMapFrameDataset | None]:
    """Use the same episode-level split rule as the F6 training stage."""
    root = Path(root)
    with (root / "meta" / "episodes.jsonl").open() as f:
        n_episodes = sum(bool(line.strip()) for line in f)
    if n_episodes < 1:
        raise ValueError("No episodes in sidecar")
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must be in [0, 1)")
    order = np.random.default_rng(seed).permutation(n_episodes).tolist()
    n_val = min(round(n_episodes * val_ratio), n_episodes - 1)
    stats = TactileMapStats.from_root(root)
    train = TactileMapFrameDataset(
        root,
        episode_indices=sorted(order[n_val:]),
        stats=stats,
        stride=stride,
        hand_mode=hand_mode,
    )
    val = (
        TactileMapFrameDataset(
            root,
            episode_indices=sorted(order[:n_val]),
            stats=stats,
            stride=stride,
            hand_mode=hand_mode,
        )
        if n_val
        else None
    )
    return train, val
