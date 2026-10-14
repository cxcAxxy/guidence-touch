"""CPU-only Gloo regression for global CPM DDP helpers; no real model/training."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import timedelta
import hashlib
import json
from pathlib import Path
import random
import tempfile
from unittest.mock import patch

import numpy as np
import torch
from torch import distributed as dist
from torch import multiprocessing as mp
from torch import nn
from torch.nn import functional
from torch.nn.parallel import DistributedDataParallel

from openpi.touchguide.batching import EpochIndexBatchSampler
from openpi.touchguide.distributed import BroadcastFlowActionNoise
from openpi.touchguide.distributed import GlobalBatchShardSampler
from openpi.touchguide.distributed import capture_distributed_rng
from openpi.touchguide.distributed import capture_local_rng_state
from openpi.touchguide.distributed import gather_model_output
from openpi.touchguide.distributed import restore_distributed_rng
from openpi.touchguide.distributed import restore_local_rng_state
from openpi.touchguide.distributed import unwrap_model
from openpi.touchguide.losses import cpm_contrastive_loss
from openpi.touchguide.noise import FlowActionNoise
from openpi.touchguide.noise import NoiseConfig


class TinyCPM(nn.Module):
    def __init__(self):
        super().__init__()
        self.observation = nn.Linear(4, 5)
        self.action = nn.Linear(3, 5)
        self.logit_scale = nn.Parameter(torch.tensor(1.25))

    def forward(self, observation, action):
        obs = functional.normalize(self.observation(observation), dim=-1)
        act = functional.normalize(self.action(action), dim=-1)
        temperature = self.logit_scale.exp().reciprocal()
        cosine = obs @ act.T
        return {
            "observation_embedding": obs,
            "action_embedding": act,
            "temperature": temperature,
            "cosine_scores": cosine,
            "logits": cosine / temperature,
        }


def labels(size):
    positive = torch.eye(size, dtype=torch.bool)
    return {"targets": torch.arange(size), "positive_mask": positive, "negative_mask": ~positive}


def draws():
    return random.random(), float(np.random.random()), torch.rand(5)


def equal_draws(left, right):
    assert left[:2] == right[:2]
    torch.testing.assert_close(left[2], right[2], rtol=0, atol=0)


def noise_seed(probabilities, target):
    for seed in range(1000):
        if int(torch.multinomial(probabilities, 1, generator=torch.Generator().manual_seed(seed))) == target:
            return seed
    raise AssertionError("Could not find fixture seed")


def worker(rank, world, init_method, report_path):
    global_batch = 3 * world
    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=world, init_method=init_method, timeout=timedelta(seconds=90))
    try:
        torch.manual_seed(321)
        model = DistributedDataParallel(TinyCPM())
        reference = TinyCPM()
        reference.load_state_dict(deepcopy(unwrap_model(model).state_dict()))
        generator = torch.Generator().manual_seed(928)
        observation = torch.randn(global_batch, 4, generator=generator)
        action = torch.randn(global_batch, 3, generator=generator)
        reference_output = reference(observation, action)
        reference_metrics = cpm_contrastive_loss(reference_output, labels(global_batch))
        reference_metrics["loss"].backward()
        start = rank * 3
        output = gather_model_output(model(observation[start : start + 3], action[start : start + 3]))
        assert output["logits"].shape == (global_batch, global_batch)
        assert torch.equal(output["targets"], torch.arange(global_batch))
        assert torch.equal(output["positive_mask"], torch.eye(global_batch, dtype=torch.bool))
        assert torch.equal(output["negative_mask"], ~torch.eye(global_batch, dtype=torch.bool))
        torch.testing.assert_close(output["logits"], reference_output["logits"], rtol=1e-6, atol=2e-6)
        metrics = cpm_contrastive_loss(output, output)
        torch.testing.assert_close(metrics["loss"], reference_metrics["loss"], rtol=1e-6, atol=1e-6)
        metrics["loss"].backward()
        differences = {}
        for (name, param), (reference_name, reference_param) in zip(
            unwrap_model(model).named_parameters(), reference.named_parameters(), strict=True
        ):
            assert name == reference_name
            assert param.grad is not None
            assert reference_param.grad is not None
            torch.testing.assert_close(param.grad, reference_param.grad, rtol=2e-5, atol=2e-6)
            differences[name] = float((param.grad - reference_param.grad).abs().max())
        sampler = GlobalBatchShardSampler(
            EpochIndexBatchSampler(3 * global_batch, global_batch, seed=42), rank=rank, world_size=world
        )
        assert isinstance(sampler, EpochIndexBatchSampler)
        assert sampler.batch_size == global_batch
        assert sampler.local_batch_size == 3
        assert len(sampler) == 3
        coverage = []
        for epoch in (0, 2):
            sampler.set_epoch(epoch)
            local = list(sampler)
            shards = [None] * world
            dist.all_gather_object(shards, local)
            combined = [[index for shard in shards for index in shard[batch]] for batch in range(len(sampler))]
            expected = list(EpochIndexBatchSampler(3 * global_batch, global_batch, seed=42, epoch=epoch))
            assert combined == expected
            flat = [index for batch in combined for index in batch]
            assert len(flat) == 3 * global_batch
            assert len(set(flat)) == 3 * global_batch
            coverage.append({"epoch": epoch, "global_batches": combined})
        batch = {
            "actions": torch.arange(3 * 30 * 27, dtype=torch.float32).reshape(3, 30, 27) / 1000,
            "sample_ids": [f"fixture:{rank}:{i}" for i in range(3)],
            "metadata": [{"rank": rank}] * 3,
            **labels(3),
        }
        noise = BroadcastFlowActionNoise(NoiseConfig())
        seed = noise_seed(noise.probabilities, rank + 2)
        raw = FlowActionNoise(noise.config).augment(batch, generator=torch.Generator().manual_seed(seed))
        augmented = noise.augment(batch, generator=torch.Generator().manual_seed(seed))
        assert int(raw["noise_level"]) == rank + 2
        assert int(augmented["noise_level"]) == 2
        torch.testing.assert_close(augmented["flow_time"], torch.tensor(0.2), rtol=0, atol=0)
        assert augmented["actions"] is batch["actions"]
        assert augmented["metadata"] is batch["metadata"]
        torch.testing.assert_close(augmented["noise_epsilon"], raw["noise_epsilon"], rtol=0, atol=0)
        expected = (1 - augmented["flow_time"]) * batch["actions"] + augmented["flow_time"] * raw["noise_epsilon"]
        torch.testing.assert_close(augmented["noisy_actions"], expected, rtol=0, atol=0)
        epsilons = [None] * world
        dist.all_gather_object(epsilons, augmented["noise_epsilon"])
        assert len({hashlib.sha256(value.numpy().tobytes()).hexdigest() for value in epsilons}) == world
        # Rank zero alone invokes validation; rank one waits at a barrier.
        # Inherited validation augmentation must not perform a collective.
        if rank == 0:
            validation = noise.augment_validation_at_time(batch, 0.4, seed=42)
            parent_validation = FlowActionNoise(noise.config).augment_validation_at_time(batch, 0.4, seed=42)
            torch.testing.assert_close(validation["noise_epsilon"], parent_validation["noise_epsilon"], rtol=0, atol=0)
        dist.barrier()
        random.seed(500 + rank)
        np.random.seed(500 + rank)
        torch.manual_seed(500 + rank)
        local_rng = capture_local_rng_state("cpu")
        expected_draws = draws()
        restore_local_rng_state(local_rng, "cpu")
        equal_draws(draws(), expected_draws)
        states = capture_distributed_rng("cpu")
        assert len(states) == world
        assert not torch.equal(states[0]["torch_cpu"], states[1]["torch_cpu"])
        expected_draws = draws()
        restore_distributed_rng(states, "cpu")
        equal_draws(draws(), expected_draws)
        assert all(value["torch_cuda"] == [] for value in states)
        payload = {
            "rank": rank,
            "gradient_max_abs_differences": differences,
            "loss": float(metrics["loss"].detach()),
            "shard_coverage": coverage,
            "broadcast_time": float(augmented["flow_time"]),
            "original_local_level": int(raw["noise_level"]),
            "rng_rank_order_and_restore": "passed",
        }
        reports = [None] * world
        dist.all_gather_object(reports, payload)
        if rank == 0:
            Path(report_path).write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "backend": "gloo",
                        "world_size": world,
                        "global_batch": global_batch,
                        "ranks": reports,
                        "loss_world_size_multiplier": False,
                        "real_cpm_loaded": False,
                        "gpu_used": False,
                        "optimizer_step": False,
                    },
                    indent=2,
                )
                + "\n"
            )
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-size", type=int, choices=(2, 8), default=2)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    # The sampler must reject singleton/uneven shards before reading data.
    for size, world in [(6, 4), (8, 8)]:
        try:
            GlobalBatchShardSampler(EpochIndexBatchSampler(20, size), rank=0, world_size=world)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid shard size accepted")
    before = capture_local_rng_state("cpu")
    fake_cuda = torch.arange(32, dtype=torch.uint8)
    recorded = []
    with (
        patch("torch.cuda.current_device", return_value=0),
        patch("torch.cuda.get_rng_state", return_value=fake_cuda) as getter,
        patch("torch.cuda.get_rng_state_all", side_effect=AssertionError("Touched every GPU")),
        patch("torch.cuda.set_rng_state", side_effect=lambda value, device=None: recorded.append((value, device))),
    ):
        state = capture_local_rng_state("cuda:0")
        getter.assert_called_once_with(torch.device("cuda:0"))
        restore_local_rng_state(state, "cuda:0")
        assert len(recorded) == 1
        assert recorded[0][1] == torch.device("cuda:0")
        torch.testing.assert_close(recorded[0][0], fake_cuda, rtol=0, atol=0)
    restore_local_rng_state(before, "cpu")
    report = str(
        args.report
        or Path(
            "/tmp/touchguide_distributed_validation.json"
            if args.world_size == 2
            else "/tmp/touchguide_distributed_validation_8.json"
        )
    )
    with tempfile.TemporaryDirectory(prefix="touchguide_gloo_fixture_") as folder:
        mp.spawn(
            worker, args=(args.world_size, f"file://{folder}/rendezvous", report), nprocs=args.world_size, join=True
        )
    payload = json.loads(Path(report).read_text())
    print(
        json.dumps(
            {
                "status": payload["status"],
                "world_size": args.world_size,
                "global_batch": payload["global_batch"],
                "maximum_gradient_abs_difference": max(
                    max(rank["gradient_max_abs_differences"].values()) for rank in payload["ranks"]
                ),
                "report": report,
                "gpu_used": False,
                "optimizer_step": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
