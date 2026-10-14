"""Pair LeRobot poker frames with the tactile sidecar without timestamp guessing."""

from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.tactile_vqvae.data.pi05_trex import Pi05TrexF6Stats
from openpi.tactile_vqvae.data.tactile_map import TactileMapStats


class PokerTactileDataset(Dataset):
    def __init__(self, pi05_root: str | Path, tactile_root: str | Path,
                 norm_stats_path: str | Path, *, split: str, seed: int = 42,
                 val_ratio: float = 0.1, stride: int = 4) -> None:
        if split not in {"train", "val"} or stride <= 0 or not 0 < val_ratio < 1:
            raise ValueError("Invalid split, stride or validation ratio")
        self.pi05_root = Path(pi05_root)
        self.tactile_root = Path(tactile_root)
        with (self.tactile_root / "meta" / "episodes.jsonl").open() as f:
            self.episodes = [json.loads(line) for line in f if line.strip()]
        if len(self.episodes) != 200:
            raise ValueError(f"Expected 200 paired episodes, got {len(self.episodes)}")
        if [e["episode_index"] for e in self.episodes] != list(range(len(self.episodes))):
            raise ValueError("Tactile episode indices must be consecutive")
        self.offsets = np.cumsum([0] + [int(e["num_frames"]) for e in self.episodes[:-1]]).tolist()
        order = np.random.default_rng(seed).permutation(len(self.episodes)).tolist()
        n_val = round(len(self.episodes) * val_ratio)
        selected = sorted(order[:n_val] if split == "val" else order[n_val:])
        self.samples = [(episode, frame) for episode in selected
                        for frame in range(0, int(self.episodes[episode]["num_frames"]), stride)]
        self.lerobot = LeRobotDataset(
            "poker-draw", root=self.pi05_root,
            delta_timestamps={"action": [i / 30 for i in range(30)]},
            video_backend="pyav",
        )
        if len(self.lerobot) != sum(int(e["num_frames"]) for e in self.episodes):
            raise ValueError("LeRobot and tactile sidecar frame counts differ")
        with Path(norm_stats_path).open() as f:
            payload = json.load(f)["norm_stats"]
        # The original pi05_kaihand data config enables quantile normalization.
        # Its state is discretized into prompt tokens after normalization, so
        # both state and action ranges must match the frozen pi0.5 checkpoint.
        self.state_q01 = np.asarray(payload["state"]["q01"], dtype=np.float32)
        self.state_q99 = np.asarray(payload["state"]["q99"], dtype=np.float32)
        self.action_q01 = np.asarray(payload["actions"]["q01"], dtype=np.float32)
        self.action_q99 = np.asarray(payload["actions"]["q99"], dtype=np.float32)
        if any(a.shape != (27,) for a in (self.state_q01, self.state_q99, self.action_q01, self.action_q99)):
            raise ValueError("Expected 27-D poker state/action normalization stats")
        self.f6_stats = Pi05TrexF6Stats.from_dataset_root(self.tactile_root)
        self.map_stats = TactileMapStats.from_root(self.tactile_root)
        with (self.pi05_root / "meta" / "tasks.jsonl").open() as f:
            tasks = [json.loads(line) for line in f if line.strip()]
        if len(tasks) != 1:
            raise ValueError("Expected the one-task poker dataset")
        self.prompt = tasks[0]["task"]
        self.tokenizer = PaligemmaTokenizer(200)
        self._cache: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()

    def __len__(self) -> int:
        return len(self.samples)

    def _sidecar(self, episode: int) -> dict[str, np.ndarray]:
        if episode in self._cache:
            self._cache.move_to_end(episode)
            return self._cache[episode]
        path = self.tactile_root / self.episodes[episode]["file"]
        with h5py.File(path) as f:
            data = {key: np.asarray(f[key]) for key in
                    ("tactile_f6", "tactile_map", "history_indices", "frame_index",
                     "source_state_index", "source_camera_timestamp_s")}
        n = int(self.episodes[episode]["num_frames"])
        if (data["tactile_f6"].shape != (n, 10, 6) or
                data["tactile_map"].shape != (n, 10, 3, 7, 5) or
                data["history_indices"].shape != (n, 16)):
            raise ValueError(f"Unexpected tactile array shape in {path}")
        self._cache[episode] = data
        while len(self._cache) > 4:
            self._cache.popitem(last=False)
        return data

    @staticmethod
    def _image(value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 3:
            raise ValueError(f"Expected one video frame, got {tuple(value.shape)}")
        if value.shape[0] != 3:
            value = value.permute(2, 0, 1)
        if value.dtype == torch.uint8:
            return value.float() / 127.5 - 1.0
        # LeRobot decodes video frames as [0, 1] float tensors.
        return value.float() * 2.0 - 1.0

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        episode, frame = self.samples[index]
        item = self.lerobot[self.offsets[episode] + frame]
        if int(item["episode_index"]) != episode or int(item["frame_index"]) != frame:
            raise ValueError(f"LeRobot frame mismatch at episode {episode}, frame {frame}")
        sidecar = self._sidecar(episode)
        if int(sidecar["frame_index"][frame]) != frame:
            raise ValueError(f"Tactile frame mismatch at episode {episode}, frame {frame}")
        if (int(sidecar["source_state_index"][frame]) != int(item["provenance.source_state_index"]) or
                abs(float(sidecar["source_camera_timestamp_s"][frame]) -
                    float(item["provenance.source_camera_timestamp_s"])) > 1e-4):
            raise ValueError(f"Tactile source-time mismatch at episode {episode}, frame {frame}")
        history_idx = sidecar["history_indices"][frame].astype(np.int64)
        if np.any(history_idx < 0) or np.any(history_idx > frame):
            raise ValueError("Tactile history contains future frames")
        state_raw = np.asarray(item["observation.state"], dtype=np.float32)
        actions_raw = np.asarray(item["action"], dtype=np.float32)
        if state_raw.shape != (27,) or actions_raw.shape != (30, 27):
            raise ValueError(f"Expected state [27] and action [30,27], got {state_raw.shape}/{actions_raw.shape}")
        # Match the base pi0.5 preprocessing: first seven arm joints are delta
        # actions, remaining 20 right-hand joints stay absolute.
        actions_raw = actions_raw.copy()
        actions_raw[:, :7] -= state_raw[None, :7]
        state = 2.0 * (state_raw - self.state_q01) / (self.state_q99 - self.state_q01 + 1e-6) - 1.0
        actions = 2.0 * (actions_raw - self.action_q01) / (self.action_q99 - self.action_q01 + 1e-6) - 1.0
        prompt_ids, prompt_mask = self.tokenizer.tokenize(self.prompt, state)
        history = self.f6_stats.normalize_hand(sidecar["tactile_f6"][history_idx, 5:], 1)
        current = self.f6_stats.normalize_hand(sidecar["tactile_f6"][frame, 5:], 1)
        maps = self.map_stats.normalize(sidecar["tactile_map"][frame, 5:])
        return {
            "head": self._image(item["observation.images.head"]),
            "wrist": self._image(item["observation.images.right_wrist"]),
            "state": torch.from_numpy(np.pad(state, (0, 5))),
            "actions": torch.from_numpy(np.pad(actions, ((0, 0), (0, 5)))),
            "prompt_ids": torch.from_numpy(prompt_ids.astype(np.int64)),
            "prompt_mask": torch.from_numpy(prompt_mask.astype(np.bool_)),
            "f6_history": torch.from_numpy(history),
            "f6_current": torch.from_numpy(current),
            "tactile_map": torch.from_numpy(maps),
        }
