"""Flow-matching action augmentation for TouchGuide's shared action columns.

The geometric parameter, flow grid and batch-wide time are local settings.
This data module does not construct CPM scores, losses or a training loop.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import operator
from pathlib import Path
from typing import Any

import torch

from openpi.touchguide.batching import CPMBatchLoader


def truncated_geometric_probabilities(num_levels: int = 10, p: float = 0.3) -> torch.Tensor:
    """P(k) proportional to p*(1-p)**k, normalized on k=0..num_levels-1.

    Sampling then clamping an unbounded geometric random variable would have
    different tail mass; it is deliberately not used here.
    """
    num_levels = operator.index(num_levels)
    if num_levels < 1 or not math.isfinite(p) or not 0 < p < 1:
        raise ValueError("num_levels must be positive and p must be strictly between 0 and 1")
    logits = torch.arange(num_levels, dtype=torch.float64) * math.log1p(-p)
    return torch.softmax(logits, dim=0)


@dataclass(frozen=True)
class NoiseConfig:
    num_flow_steps: int = 10
    geometric_p: float = 0.3
    time_scope: str = "batch"
    enabled: bool = True

    def __post_init__(self) -> None:
        if operator.index(self.num_flow_steps) < 2 or self.time_scope != "batch":
            raise ValueError("Use at least two flow steps and batch-wide noise time")
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be boolean")
        truncated_geometric_probabilities(self.num_flow_steps, self.geometric_p)

    @classmethod
    def from_dict(cls, settings: dict[str, Any]) -> NoiseConfig:
        return cls(**{key: settings[key] for key in ("num_flow_steps", "geometric_p", "time_scope", "enabled")})


def _stable_seed(seed: int, identity: str) -> int:
    seed = operator.index(seed)
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    digest = hashlib.blake2b(f"{seed}:{identity}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") & ((1 << 63) - 1)


class FlowActionNoise:
    """Add one epsilon per stored action and reuse it for every observation row."""

    def __init__(self, config: NoiseConfig | None = None) -> None:
        if config is None:
            config = NoiseConfig()
        self.config = config
        self.probabilities = truncated_geometric_probabilities(config.num_flow_steps, config.geometric_p)

    @staticmethod
    def _actions(batch: dict[str, Any]) -> torch.Tensor:
        if "noisy_actions" in batch:
            raise ValueError("Augment a clean batch once; do not apply noise to an augmented batch")
        if "negative_actions" in batch or "noisy_negative_actions" in batch:
            raise ValueError("Legacy separately generated negative actions are not supported")
        actions = batch["actions"]
        if (
            not isinstance(actions, torch.Tensor)
            or actions.ndim != 3
            or tuple(actions.shape[1:]) != (30, 27)
            or actions.shape[0] < 2
            or not actions.is_floating_point()
        ):
            raise ValueError("Expected a floating action batch [B>=2,30,27]")
        if not torch.isfinite(actions).all():
            raise ValueError("Actions must be finite before augmentation")
        return actions

    @staticmethod
    def _epsilon(actions: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
        device = actions.device if generator is None else torch.device(generator.device)
        return torch.randn(actions.shape, dtype=actions.dtype, device=device, generator=generator).to(actions.device)

    def _attach(
        self,
        batch: dict[str, Any],
        epsilon: torch.Tensor,
        flow_time: float | torch.Tensor,
        noise_level: int | torch.Tensor,
    ) -> dict[str, Any]:
        actions = batch["actions"]
        time = torch.as_tensor(flow_time, dtype=actions.dtype, device=actions.device).reshape(())
        level = torch.as_tensor(noise_level, dtype=torch.long, device=actions.device).reshape(())
        result = dict(batch)
        result.update(
            noisy_actions=(1 - time) * actions + time * epsilon,
            flow_time=time,
            noise_level=level,
            noise_epsilon=epsilon,
        )
        return result

    def _disabled(self, batch: dict[str, Any], actions: torch.Tensor) -> dict[str, Any]:
        result = self._attach(batch, torch.zeros_like(actions), 0.0, 0)
        result["noisy_actions"] = actions
        return result

    def augment(self, batch: dict[str, Any], *, generator: torch.Generator | None = None) -> dict[str, Any]:
        actions = self._actions(batch)
        if not self.config.enabled:
            return self._disabled(batch, actions)
        random_device = actions.device if generator is None else torch.device(generator.device)
        level = torch.multinomial(self.probabilities.to(random_device), 1, generator=generator).reshape(())
        epsilon = self._epsilon(actions, generator)
        return self._attach(batch, epsilon, level.float() / self.config.num_flow_steps, level)

    def augment_at_time(
        self, batch: dict[str, Any], flow_time: float, *, generator: torch.Generator | None = None
    ) -> dict[str, Any]:
        """Force a time for diagnostics; t=1 is allowed here, never randomly sampled."""
        actions = self._actions(batch)
        if not math.isfinite(flow_time) or not 0 <= flow_time <= 1:
            raise ValueError("flow_time must be between 0 and 1")
        if not self.config.enabled:
            return self._disabled(batch, actions)
        scaled = flow_time * self.config.num_flow_steps
        level = round(scaled) if math.isclose(scaled, round(scaled), abs_tol=1e-6) else -1
        return self._attach(batch, self._epsilon(actions, generator), flow_time, level)

    def augment_validation_at_time(self, batch: dict[str, Any], flow_time: float, *, seed: int = 42) -> dict[str, Any]:
        """Stable per-ID CPU epsilon, shared across diagnostic times and batching."""
        actions = self._actions(batch)
        if not math.isfinite(flow_time) or not 0 <= flow_time <= 1:
            raise ValueError("flow_time must be between 0 and 1")
        if not self.config.enabled:
            return self._disabled(batch, actions)
        ids = batch["sample_ids"]
        if len(ids) != len(actions) or len(set(ids)) != len(ids):
            raise ValueError("Validation noise requires one distinct ID per action")
        values = []
        for identity, action in zip(ids, actions, strict=True):
            generator = torch.Generator(device="cpu").manual_seed(_stable_seed(seed, identity))
            values.append(torch.randn(action.shape, dtype=actions.dtype, generator=generator))
        epsilon = torch.stack(values).to(actions.device)
        scaled = flow_time * self.config.num_flow_steps
        level = round(scaled) if math.isclose(scaled, round(scaled), abs_tol=1e-6) else -1
        return self._attach(batch, epsilon, flow_time, level)


class NoiseAugmentedBatchLoader:
    """Apply action augmentation after clean collation, with explicit split modes."""

    def __init__(
        self,
        clean_loader: CPMBatchLoader,
        augmenter: FlowActionNoise,
        *,
        split: str,
        seed: int = 42,
        epoch: int = 0,
        validation_time: float = 0.0,
    ):
        if split not in ("train", "val") or clean_loader.dataset.split != split:
            raise ValueError("Noise loader split differs from its clean dataset")
        if epoch < 0 or not math.isfinite(validation_time) or not 0 <= validation_time <= 1:
            raise ValueError("Invalid epoch or validation time")
        _stable_seed(seed, "seed_check")
        self.clean_loader, self.augmenter = clean_loader, augmenter
        self.split, self.seed, self.epoch = split, seed, epoch
        self.validation_time = validation_time
        if split == "train":
            self.clean_loader.set_epoch(epoch)

    @property
    def dataset(self):
        return self.clean_loader.dataset

    def set_epoch(self, epoch: int) -> None:
        if self.split != "train":
            raise ValueError("Validation noise and groups do not have training epochs")
        self.clean_loader.set_epoch(epoch)
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.clean_loader)

    def __iter__(self):
        if self.split == "train":
            generator = torch.Generator(device="cpu").manual_seed(_stable_seed(self.seed, f"epoch:{self.epoch}"))
            for batch in self.clean_loader:
                yield self.augmenter.augment(batch, generator=generator)
        else:
            for batch in self.clean_loader:
                yield self.augmenter.augment_validation_at_time(batch, self.validation_time, seed=self.seed)

    @classmethod
    def from_config(
        cls,
        root: str | Path,
        split: str = "train",
        *,
        epoch: int = 0,
        validation_time: float = 0.0,
        include_images: bool | None = None,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        if split not in ("train", "val"):
            raise ValueError("split must be train or val")
        root = Path(root).expanduser().resolve()
        settings = json.loads((root / f"config/{split}.json").read_text())["noise_augmentation"]
        if settings["application_stage"] != "after_clean_batch_collation":
            raise ValueError("Noise must be applied after clean batch collation")
        augmenter = FlowActionNoise(NoiseConfig.from_dict(settings))
        clean = CPMBatchLoader.from_config(
            root, split, epoch=epoch, include_images=include_images, num_workers=num_workers, pin_memory=pin_memory
        )
        try:
            return cls(
                clean, augmenter, split=split, seed=settings["seed"], epoch=epoch, validation_time=validation_time
            )
        except BaseException:
            clean.dataset.close()
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--preview-size", type=int, default=3)
    parser.add_argument("--flow-time", type=float)
    parser.add_argument("--no-images", action="store_true")
    args = parser.parse_args()
    loader = NoiseAugmentedBatchLoader.from_config(args.root, args.split, include_images=not args.no_images)
    try:
        indices = next(iter(loader.clean_loader.index_batches))
        if not 2 <= args.preview_size <= len(indices):
            raise ValueError("preview-size must be between 2 and the first batch length")
        clean = next(iter(CPMBatchLoader(loader.dataset, [indices[: args.preview_size]])))
        if args.split == "val":
            time = 0.0 if args.flow_time is None else args.flow_time
            noisy = loader.augmenter.augment_validation_at_time(clean, time, seed=loader.seed)
        else:
            generator = torch.Generator().manual_seed(_stable_seed(loader.seed, "epoch:0"))
            noisy = (
                loader.augmenter.augment(clean, generator=generator)
                if args.flow_time is None
                else loader.augmenter.augment_at_time(clean, args.flow_time, generator=generator)
            )
        print(
            json.dumps(
                {
                    "sample_ids": noisy["sample_ids"],
                    "flow_time": float(noisy["flow_time"]),
                    "noise_level": int(noisy["noise_level"]),
                    "action_shape": list(noisy["noisy_actions"].shape),
                    "targets": noisy["targets"].tolist(),
                    "clean_actions_preserved": True,
                    "model_loaded": False,
                    "training_started": False,
                },
                indent=2,
            )
        )
    finally:
        loader.dataset.close()


if __name__ == "__main__":
    main()
