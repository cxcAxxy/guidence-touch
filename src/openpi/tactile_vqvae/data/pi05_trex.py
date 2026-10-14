"""Dataset adapter for the pi05_trex tactile sidecar.

The sidecar stores one HDF5 file per episode with ``tactile_f6`` shaped
[N, 10, 6] and ``history_indices`` shaped [N, 16].  T-Rex's F6 model is
single-hand, so this adapter selects five fingers before returning a window
shaped [16, 5, 6].
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


def _load_h5py():
    try:
        import h5py  # noqa: PLC0415 - delay optional HDF5 dependency until data access
    except ImportError as exc:  # pragma: no cover - depends on the runtime image
        raise ImportError(
            "The pi05_trex HDF5 loader requires h5py. Install the openpi runtime "
            "dependencies (h5py is already present in the project's uv lockfile)."
        ) from exc
    return h5py


@dataclass(frozen=True)
class Pi05TrexF6Stats:
    """Per-fingertip robust ranges used by the sidecar conversion.

    ``q01`` and ``q99`` have shape [10, 6], matching the sidecar's left-then-
    right fingertip order.  Constant channels are mapped to zero rather than
    to -1; this matters for the unused, all-zero left hand in the current
    poker data.
    """

    q01: np.ndarray
    q99: np.ndarray
    eps: float = 1e-6

    def __post_init__(self) -> None:
        q01 = np.asarray(self.q01, dtype=np.float32)
        q99 = np.asarray(self.q99, dtype=np.float32)
        if q01.shape != (10, 6) or q99.shape != (10, 6):
            raise ValueError(f"Expected q01/q99 shape [10, 6], got {q01.shape}/{q99.shape}")
        object.__setattr__(self, "q01", q01)
        object.__setattr__(self, "q99", q99)

    @classmethod
    def from_json(cls, path: str | Path) -> Pi05TrexF6Stats:
        with Path(path).open() as f:
            payload = json.load(f)
        try:
            f6 = payload["f6"]
            return cls(q01=f6["q01"], q99=f6["q99"])
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Invalid tactile stats file: {path}") from exc

    @classmethod
    def from_dataset_root(cls, root: str | Path) -> Pi05TrexF6Stats:
        return cls.from_json(Path(root) / "meta" / "tactile_stats.json")

    def to_dict(self) -> dict[str, Any]:
        return {
            "q01": self.q01.tolist(),
            "q99": self.q99.tolist(),
            "eps": self.eps,
            "shape": [10, 6],
            "normalization": "clip(2 * (x-q01)/(q99-q01) - 1); constant channels -> 0",
        }

    def _normalize(self, x: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        scale = q99 - q01
        valid = scale > self.eps
        normalized = 2.0 * (x - q01) / np.where(valid, scale, 1.0) - 1.0
        normalized = np.where(valid, normalized, 0.0)
        return np.clip(normalized, -1.0, 1.0).astype(np.float32, copy=False)

    def normalize_hand(self, x: np.ndarray, hand: int) -> np.ndarray:
        """Normalize x with shape [..., 5, 6] for hand 0 (left) or 1 (right)."""
        if hand not in (0, 1):
            raise ValueError(f"hand must be 0 or 1, got {hand}")
        x = np.asarray(x, dtype=np.float32)
        if x.shape[-2:] != (5, 6):
            raise ValueError(f"Expected hand F6 shape [..., 5, 6], got {x.shape}")
        start = hand * 5
        return self._normalize(x, self.q01[start : start + 5], self.q99[start : start + 5])

    def normalize_full(self, x: np.ndarray) -> np.ndarray:
        """Normalize x with shape [..., 10, 6]."""
        x = np.asarray(x, dtype=np.float32)
        if x.shape[-2:] != (10, 6):
            raise ValueError(f"Expected full F6 shape [..., 10, 6], got {x.shape}")
        return self._normalize(x, self.q01, self.q99)


class Pi05TrexF6WindowDataset(Dataset[dict[str, torch.Tensor]]):
    """Lazy F6 window dataset backed by a ``pi05_trex`` sidecar.

    A sample is one hand and one frame.  The returned ``f6`` is normalized and
    has shape [window, 5, 6].  ``magnitude`` is computed from the raw window
    and is used by T-Rex's contact-weighted reconstruction loss.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        stats: Pi05TrexF6Stats | None = None,
        episode_indices: Sequence[int] | None = None,
        window: int = 16,
        stride: int = 4,
        hand_mode: str = "both",
        drop_incomplete: bool = True,
    ) -> None:
        self.root = Path(root)
        self.window = int(window)
        self.stride = int(stride)
        if self.window <= 0 or self.stride <= 0:
            raise ValueError("window and stride must be positive")
        if hand_mode not in {"left", "right", "both"}:
            raise ValueError("hand_mode must be 'left', 'right', or 'both'")
        self.hands = {"left": (0,), "right": (1,), "both": (0, 1)}[hand_mode]
        self.hand_mode = hand_mode
        self.stats = stats or Pi05TrexF6Stats.from_dataset_root(self.root)

        manifest_path = self.root / "meta" / "episodes.jsonl"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing sidecar manifest: {manifest_path}")
        with manifest_path.open() as f:
            all_episodes = [json.loads(line) for line in f if line.strip()]
        selected = list(range(len(all_episodes))) if episode_indices is None else list(episode_indices)
        if any(i < 0 or i >= len(all_episodes) for i in selected):
            raise IndexError("episode_indices contains an out-of-range episode")
        self.episodes = [all_episodes[i] for i in selected]
        self.episode_indices = selected

        self._samples: list[tuple[int, int, int]] = []
        # One episode is small (~0.35 MB); retain episodes reached by a worker
        # so shuffled windows do not reopen the same NAS file for every sample.
        self._cache: dict[int, dict[str, np.ndarray]] = {}
        for local_episode, episode in enumerate(self.episodes):
            n_frames = int(episode["num_frames"])
            first_frame = self.window - 1 if drop_incomplete else 0
            for frame in range(first_frame, n_frames, self.stride):
                self._samples.extend((local_episode, frame, hand) for hand in self.hands)
        if not self._samples:
            raise ValueError("No F6 windows found; check the sidecar and window settings")

    def __len__(self) -> int:
        return len(self._samples)

    def _load_episode(self, local_episode: int) -> dict[str, np.ndarray]:
        if local_episode in self._cache:
            return self._cache[local_episode]
        h5py = _load_h5py()
        path = self.root / self.episodes[local_episode]["file"]
        with h5py.File(path, "r") as f:
            data = {
                "tactile_f6": np.asarray(f["tactile_f6"], dtype=np.float32),
                "history_indices": np.asarray(f["history_indices"], dtype=np.int64),
                "history_valid": np.asarray(f["history_valid"], dtype=np.bool_)
                if "history_valid" in f
                else np.ones((len(f["tactile_f6"]), self.window), dtype=np.bool_),
            }
        if data["tactile_f6"].ndim != 3 or data["tactile_f6"].shape[1:] != (10, 6):
            raise ValueError(f"{path}: tactile_f6 must have shape [N, 10, 6]")
        if data["history_indices"].shape != (len(data["tactile_f6"]), self.window):
            raise ValueError(f"{path}: history_indices has an unexpected shape")
        self._cache[local_episode] = data
        return data

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        local_episode, frame, hand = self._samples[index]
        episode = self._load_episode(local_episode)
        history_indices = episode["history_indices"][frame]
        if np.any(history_indices < 0) or np.any(history_indices >= len(episode["tactile_f6"])):
            raise IndexError(f"Invalid history index in episode {local_episode}, frame {frame}")
        raw_window_full = episode["tactile_f6"][history_indices]  # [T, 10, 6]
        raw_window = raw_window_full[:, hand * 5 : (hand + 1) * 5]
        f6 = self.stats.normalize_hand(raw_window, hand)
        magnitude = float(np.linalg.norm(raw_window.reshape(-1)))
        valid = episode["history_valid"][frame].astype(np.bool_, copy=False)
        return {
            "f6": torch.from_numpy(f6),
            "magnitude": torch.tensor(magnitude, dtype=torch.float32),
            "history_valid": torch.from_numpy(valid.copy()),
            "episode_index": torch.tensor(self.episode_indices[local_episode], dtype=torch.long),
            "frame_index": torch.tensor(frame, dtype=torch.long),
            "hand": torch.tensor(hand, dtype=torch.long),
        }


def build_train_val_datasets(
    root: str | Path,
    *,
    window: int = 16,
    stride: int = 4,
    val_ratio: float = 0.1,
    seed: int = 42,
    hand_mode: str = "both",
) -> tuple[Pi05TrexF6WindowDataset, Pi05TrexF6WindowDataset | None]:
    """Split by episode, avoiding leakage between train and validation windows."""
    root = Path(root)
    manifest_path = root / "meta" / "episodes.jsonl"
    with manifest_path.open() as f:
        n_episodes = sum(bool(line.strip()) for line in f)
    if n_episodes < 1:
        raise ValueError(f"No episodes in {manifest_path}")
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be in [0, 1)")

    rng = np.random.default_rng(seed)
    order = rng.permutation(n_episodes).tolist()
    n_val = round(n_episodes * val_ratio) if n_episodes > 1 else 0
    n_val = min(max(n_val, 0), n_episodes - 1)
    val_indices = sorted(order[:n_val])
    train_indices = sorted(order[n_val:])
    stats = Pi05TrexF6Stats.from_dataset_root(root)
    train = Pi05TrexF6WindowDataset(
        root,
        stats=stats,
        episode_indices=train_indices,
        window=window,
        stride=stride,
        hand_mode=hand_mode,
    )
    val = None
    if val_indices:
        val = Pi05TrexF6WindowDataset(
            root,
            stats=stats,
            episode_indices=val_indices,
            window=window,
            stride=stride,
            hand_mode=hand_mode,
        )
    return train, val
