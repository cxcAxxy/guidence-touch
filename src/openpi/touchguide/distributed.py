"""DDP helpers for global TouchGuide observation/action matching.

Each rank computes the same complete global symmetric InfoNCE after an
autograd-preserving embedding gather. Gather backward sums embedding gradients
and DDP averages parameter gradients: these operations recover the single-global-
batch mean exactly. Do not multiply this loss by world_size.

Batch shards retain global permutation order. Positives are diagonal identities;
o off-diagonal pair is labeled as an experimentally verified physical failure.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
import operator
import os
import random
from typing import Any

import numpy as np
import torch
from torch import distributed as dist
from torch import nn
from torch.distributed.nn import functional as distributed_functional
from torch.nn.parallel import DistributedDataParallel

from openpi.touchguide.batching import EpochIndexBatchSampler
from openpi.touchguide.noise import FlowActionNoise


@dataclass(frozen=True)
class DistContext:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device

    @property
    def is_primary(self) -> bool:
        return self.rank == 0

    @property
    def distributed(self) -> bool:
        return self.world_size > 1


def _rank_world() -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def initialize() -> DistContext:
    """Use torchrun rank variables and NCCL; single-process execution uses cuda:0."""
    try:
        rank = int(os.environ.get("RANK", "0"))
        world = int(os.environ.get("WORLD_SIZE", "1"))
        local = int(os.environ.get("LOCAL_RANK", "0")) if world > 1 else 0
    except ValueError as error:
        raise ValueError("RANK, WORLD_SIZE and LOCAL_RANK must be integers") from error
    if world < 1 or not 0 <= rank < world or local < 0:
        raise ValueError("Invalid torchrun rank/world-size configuration")
    if not torch.cuda.is_available() or local >= torch.cuda.device_count():
        raise RuntimeError(
            f"CPM training requires CUDA device {local}; available devices: {torch.cuda.device_count()}"
        )
    device = torch.device("cuda", local)
    torch.cuda.set_device(device)
    if world > 1:
        if not dist.is_available() or not dist.is_nccl_available():
            raise RuntimeError(
                "Multi-GPU CPM training requires torch.distributed with NCCL"
            )
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://")
        if dist.get_backend() != "nccl" or _rank_world() != (rank, world):
            raise RuntimeError(
                "Existing process group does not match the torchrun NCCL configuration"
            )
    return DistContext(rank, world, local, device)


def is_primary() -> bool:
    return _rank_world()[0] == 0


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


class GlobalBatchShardSampler(EpochIndexBatchSampler):
    """Split each global no-replacement batch contiguously in process-rank order.

    Subclassing preserves CPMBatchLoader.set_epoch's existing type contract.
    batch_size remains the global size; local_batch_size describes yielded lists.
    Every rank has the same number of updates as the wrapped global sampler.
    """

    def __init__(
        self,
        global_sampler: EpochIndexBatchSampler,
        *,
        rank: int | None = None,
        world_size: int | None = None,
    ) -> None:
        actual_rank, actual_world = _rank_world()
        rank = actual_rank if rank is None else operator.index(rank)
        world_size = actual_world if world_size is None else operator.index(world_size)
        if not isinstance(global_sampler, EpochIndexBatchSampler) or isinstance(
            global_sampler, GlobalBatchShardSampler
        ):
            raise ValueError(
                "Wrap one global EpochIndexBatchSampler, not an already sharded sampler"
            )
        if not global_sampler.drop_last:
            raise ValueError("Distributed training requires global drop_last=True")
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError("Invalid batch-shard rank/world size")
        if (
            global_sampler.batch_size % world_size
            or global_sampler.batch_size // world_size < 2
        ):
            raise ValueError(
                "Global batch size must be divisible by world size and supply at least two samples per rank"
            )
        super().__init__(
            global_sampler.size,
            global_sampler.batch_size,
            seed=global_sampler.seed,
            epoch=global_sampler.epoch,
            drop_last=True,
        )
        self.global_sampler = global_sampler
        self.rank, self.world_size = rank, world_size
        self.local_batch_size = self.batch_size // world_size

    def __iter__(self) -> Iterator[list[int]]:
        first = self.rank * self.local_batch_size
        stop = first + self.local_batch_size
        for global_batch in self.global_sampler:
            if len(global_batch) != self.batch_size:
                raise ValueError("Global sampler yielded an incomplete training batch")
            yield global_batch[first:stop]

    def __len__(self) -> int:
        return len(self.global_sampler)

    def set_epoch(self, epoch: int) -> None:
        self.global_sampler.set_epoch(epoch)
        self.epoch = self.global_sampler.epoch


def gather_model_output(local_output: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Gather differentiable embeddings and return global scores plus identity masks.

    The return value can serve as both arguments to cpm_contrastive_loss, since
    it contains the five model output keys and global targets/positive_mask/
    negative_mask. Its temperature retains the local DDP-forward parameter graph.
    Rank-local batches must have equal shapes, as enforced by the global sampler.
    """
    observation = local_output["observation_embedding"]
    action = local_output["action_embedding"]
    temperature = local_output["temperature"]
    if (
        not isinstance(observation, torch.Tensor)
        or not isinstance(action, torch.Tensor)
        or observation.ndim != 2
        or observation.shape[0] < 2
        or observation.shape[1] < 1
        or observation.shape != action.shape
        or observation.device != action.device
        or not observation.is_floating_point()
        or not action.is_floating_point()
    ):
        raise ValueError(
            "Expected matching local floating observation/action embeddings [B>=2,D]"
        )
    if (
        not isinstance(temperature, torch.Tensor)
        or temperature.numel() != 1
        or not temperature.is_floating_point()
    ):
        raise ValueError("Expected one local differentiable temperature scalar")
    _, world = _rank_world()
    if world > 1:
        shape = torch.tensor(
            observation.shape, dtype=torch.long, device=observation.device
        )
        shapes = [torch.empty_like(shape) for _ in range(world)]
        dist.all_gather(shapes, shape)
        if any(not torch.equal(value, shape) for value in shapes):
            raise ValueError(
                "Every process must contribute equally sized embedding shards"
            )
        observation = torch.cat(
            distributed_functional.all_gather(observation.contiguous()), dim=0
        )
        action = torch.cat(
            distributed_functional.all_gather(action.contiguous()), dim=0
        )
    matching_actions = local_output.get("matching_actions")
    if matching_actions is not None:
        if world > 1:
            gathered = [torch.empty_like(matching_actions) for _ in range(world)]
            dist.all_gather(gathered, matching_actions.contiguous())
            matching_actions = torch.cat(gathered)
        from openpi.touchguide.negative_pairing import negative_weights  # noqa: PLC0415

        pair_weights = negative_weights(
            matching_actions, local_output["negative_policy"]
        )
    else:
        pair_weights = None
    # The cosine and temperature arithmetic stays FP32 under outer BF16 autocast.
    with torch.autocast(device_type=observation.device.type, enabled=False):
        observation, action, temperature = (
            observation.float(),
            action.float(),
            temperature.float().reshape(()),
        )
        scores = observation @ action.T
        logits = scores / temperature
    size = len(observation)
    positive = torch.eye(size, dtype=torch.bool, device=observation.device)
    return {
        "observation_embedding": observation,
        "action_embedding": action,
        "temperature": temperature,
        "cosine_scores": scores,
        "logits": logits,
        "targets": torch.arange(size, dtype=torch.long, device=observation.device),
        "positive_mask": positive,
        "negative_mask": ~positive,
        **({"negative_weights": pair_weights} if pair_weights is not None else {}),
    }


class BroadcastFlowActionNoise(FlowActionNoise):
    """Keep per-column epsilon local, and broadcast rank zero's training noise time.

    Give each rank a distinct training generator/loader seed; identical generators
    would otherwise produce repeated epsilon across different shards. Validation
    methods inherit the parent per-ID behavior and perform no collectives.
    """

    def augment(
        self, batch: dict[str, Any], *, generator: torch.Generator | None = None
    ) -> dict[str, Any]:
        result = super().augment(batch, generator=generator)
        if _rank_world()[1] == 1 or not self.config.enabled:
            return result
        # The clean loader yields CPU actions, while NCCL requires CUDA tensors.
        collective_device = (
            torch.device("cuda", torch.cuda.current_device())
            if dist.get_backend() == "nccl"
            else torch.device("cpu")
        )
        time = result["flow_time"].detach().to(collective_device).clone()
        level = result["noise_level"].detach().to(collective_device).clone()
        dist.broadcast(time, src=0)
        dist.broadcast(level, src=0)
        actions = result["actions"]
        result["flow_time"] = time.to(device=actions.device, dtype=actions.dtype)
        result["noise_level"] = level.to(device=actions.device, dtype=torch.long)
        result["noisy_actions"] = (1 - result["flow_time"]) * actions + result[
            "flow_time"
        ] * result["noise_epsilon"]
        return result


def _rng_device(device: torch.device | str | None) -> torch.device:
    if device is None:
        return (
            torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_initialized()
            else torch.device("cpu")
        )
    result = torch.device(device)
    if result.type not in ("cpu", "cuda"):
        raise ValueError("Local RNG capture supports CPU or the current CUDA device")
    if (
        result.type == "cuda"
        and result.index is not None
        and result.index != torch.cuda.current_device()
    ):
        raise ValueError("Capture/restore only this process's current CUDA device")
    return result


def capture_local_rng_state(device: torch.device | str | None = None) -> dict[str, Any]:
    """Capture Python/NumPy/CPU torch RNG and at most the current CUDA generator."""
    device = _rng_device(device)
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (numpy_state[0], numpy_state[1].copy(), *numpy_state[2:]),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": [torch.cuda.get_rng_state(device).clone()]
        if device.type == "cuda"
        else [],
    }


def restore_local_rng_state(
    state: dict[str, Any], device: torch.device | str | None = None
) -> None:
    device = _rng_device(device)
    if not isinstance(state, dict) or set(state) != {
        "python",
        "numpy",
        "torch_cpu",
        "torch_cuda",
    }:
        raise ValueError("Invalid local RNG state keys")
    cuda_states = state["torch_cuda"]
    expected = 1 if device.type == "cuda" else 0
    if not isinstance(cuda_states, list) or len(cuda_states) != expected:
        raise ValueError(f"Expected {expected} current-device CUDA RNG states")
    # Check CPU structures before changing global state.
    random.Random().setstate(state["python"])
    np.random.RandomState().set_state(state["numpy"])
    torch.Generator(device="cpu").set_state(state["torch_cpu"])
    if any(
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.uint8
        or value.ndim != 1
        for value in cuda_states
    ):
        raise ValueError("Invalid current-device CUDA RNG state")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if cuda_states:
        torch.cuda.set_rng_state(cuda_states[0], device=device)


def capture_distributed_rng(
    device: torch.device | str | None = None,
) -> list[dict[str, Any]]:
    """Collect local states in global-rank order; return the complete list on every rank."""
    local = capture_local_rng_state(device)
    _, world = _rank_world()
    if world == 1:
        return [local]
    states: list[Any] = [None] * world
    dist.all_gather_object(states, local)
    return states


def restore_distributed_rng(
    states: list[dict[str, Any]], device: torch.device | str | None = None
) -> None:
    rank, world = _rank_world()
    if not isinstance(states, list) or len(states) != world:
        raise ValueError(
            f"Checkpoint must contain one local RNG state for each of {world} ranks"
        )
    restore_local_rng_state(states[rank], device)
