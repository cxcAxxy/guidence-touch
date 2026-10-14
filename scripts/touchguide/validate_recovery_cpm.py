"""CPU checks of real recovery data, compact DINO model, and weighted gradients."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from openpi.touchguide.batching import CPMBatchLoader
from openpi.touchguide.batching import collate_matched_samples
from openpi.touchguide.data import CPMDataset
from openpi.touchguide.losses import cpm_contrastive_loss
from openpi.touchguide.losses import symmetric_infonce
from openpi.touchguide.losses import weighted_symmetric_infonce
from openpi.touchguide.model import load_cpm_model
from openpi.touchguide.negative_pairing import action_distances
from openpi.touchguide.negative_pairing import negative_weights
from openpi.touchguide.noise import FlowActionNoise

ROOT = Path(
    "/path/to/datasets/touchguidence/augmentation/poker_fingertip_recovery_rules_v1_random10"
)


def run(args):
    torch.set_num_threads(4)
    torch.manual_seed(42)
    report = {
        "status": "passed",
        "gpu_available": torch.cuda.is_available(),
        "large_training_started": False,
        "views": {},
    }
    for ratio in (1, 5, 10):
        view = args.root / f"ratio_1_{ratio}"
        configured = CPMBatchLoader.from_config(
            view, "train", include_images=False, include_auxiliary=True
        )
        dataset = configured.dataset
        episode = int(dataset.index["episode_index"][0])
        locations = np.flatnonzero(dataset.index["episode_index"] == episode)
        clean_count = len(
            h5py.File(args.root / "clean/episodes" / f"episode_{episode:06d}.h5")[
                "state_raw"
            ]
        )
        wanted = [0, clean_count, clean_count + 1] if ratio > 1 else [0, clean_count, 1]
        idx = [
            int(
                locations[
                    np.flatnonzero(dataset.index["sample_row"][locations] == row)[0]
                ]
            )
            for row in wanted
        ]
        samples = [dataset[i] for i in idx]
        batch = collate_matched_samples(samples, include_auxiliary=True)
        assert len(set(batch["sample_ids"])) == 3
        # Intentional clean + recovery from the same original frame is legal.
        assert (
            batch["metadata"][0]["frame_index"] == batch["metadata"][1]["frame_index"]
        )
        a = FlowActionNoise().augment_validation_at_time(batch, 0.2, seed=42)
        b = FlowActionNoise().augment_validation_at_time(batch, 0.2, seed=42)
        torch.testing.assert_close(
            a["noisy_actions"], b["noisy_actions"], rtol=0, atol=0
        )
        report["views"][f"1:{ratio}"] = {
            "train_samples": len(dataset),
            "paired_same_frame_readable": True,
            "validation_noise_reproducible": True,
        }
        dataset.close()
    policy = json.loads((args.root / "ratio_1_1/meta/negative_policy.json").read_text())
    full = policy["full_weight_distance"]
    x = torch.zeros(3, 30, 27)
    x[2, :, 15:19] = full * 2
    weights = negative_weights(x, policy)
    assert weights[0, 1] == 0
    assert weights[0, 2] == 1
    torch.testing.assert_close(weights, weights.T, rtol=0, atol=0)
    logits = torch.randn(3, 3, requires_grad=True)
    base = symmetric_infonce(logits)["loss"]
    identical = weighted_symmetric_infonce(logits, torch.ones_like(logits))["loss"]
    torch.testing.assert_close(base, identical)
    loss = weighted_symmetric_infonce(logits, weights)["loss"]
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0, 1] == 0
    assert logits.grad[1, 0] == 0
    assert logits.grad[0, 2] > 0
    x = torch.zeros(2, 30, 27)
    x[1, 0, 15:19] = 1
    # The first five steps distinguish early disagreement even if endings match.
    expected = 1 / (5**0.5)
    torch.testing.assert_close(action_distances(x)[0, 1], torch.tensor(expected))
    report["weighted_loss"] = {
        "all_one_equals_original_CE": True,
        "ignored_pair_gradient_zero": True,
        "distinct_pair_gradient_positive": True,
        "first_five_steps_sensitive": True,
    }
    if args.skip_model:
        print(json.dumps(report, indent=2))
        return
    root = args.root / "ratio_1_1"
    dataset = CPMDataset(root, "train", include_images=True)
    episode = int(dataset.index["episode_index"][0])
    with h5py.File(args.root / "clean/episodes" / f"episode_{episode:06d}.h5") as f:
        clean_count = len(f["state_raw"])
    indices = [
        0,
        int(
            np.flatnonzero(
                (dataset.index["episode_index"] == episode)
                & (dataset.index["sample_row"] == clean_count)
            )[0]
        ),
        1000,
    ]
    batch = FlowActionNoise().augment_validation_at_time(
        collate_matched_samples([dataset[i] for i in indices], include_auxiliary=True),
        0.2,
        seed=42,
    )
    dataset.close()
    model = load_cpm_model(root, "recovery-state-cpm")
    model.eval()
    report["parameters"] = model.parameter_counts()
    tokens = model.observation_tokens(
        batch["head"], batch["wrist"], batch["tactile_map_raw"]
    )
    assert tokens.shape == (3, 120, 1024)
    cached = model.encode_observation_batch(batch).detach()
    candidate = batch["noisy_actions"].detach().requires_grad_()
    scores = model.paired_scores(cached, candidate, batch["flow_time"])
    gradient = torch.autograd.grad(scores.sum(), candidate)[0]
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0
    report["guidance_gradient_norm"] = float(gradient.norm())
    # Optimize the trainable network on one deterministic real mini-batch only.
    # Frozen DINO remains frozen; eval disables dropout for a meaningful descent check.
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-6
    )
    losses = []
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        output = model(batch)
        metrics = cpm_contrastive_loss(output, batch)
        losses.append(float(metrics["loss"].detach()))
        metrics["loss"].backward()
        assert all(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in model.parameters()
            if p.requires_grad
        )
        assert all(p.grad is None for p in model.vision.parameters())
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        optimizer.step()
        print(
            f"Real CPU mini-batch update {step + 1}: loss {losses[-1]:.6f}", flush=True
        )
    with torch.no_grad():
        losses.append(float(cpm_contrastive_loss(model(batch), batch)["loss"]))
    assert losses[-1] < losses[0]
    state_pi = model.action_encoder.coordinates
    raw = batch["state_raw"]
    q0 = state_pi.state_original_low
    qwidth = state_pi.state_original_width
    old_state = 2 * (raw - q0) / qwidth - 1
    with torch.no_grad():
        left = model.encode_observation(
            batch["head"], batch["wrist"], batch["tactile_map_raw"], state_raw=raw
        )
        right = model.encode_observation(
            batch["head"],
            batch["wrist"],
            batch["tactile_map_raw"],
            state_normalized=old_state,
        )
    torch.testing.assert_close(left, right, rtol=0, atol=2e-6)
    report["real_model"] = {
        "observation_tokens": 127,
        "camera_and_touch_tokens": 120,
        "state_tokens": 7,
        "all_trainable_parameters_have_finite_gradients": True,
        "frozen_DINO_no_gradients": True,
        "mini_batch_losses": losses,
        "raw_and_original_normalized_state_equivalent": True,
        "gpu_BF16_NCCL_tested": False,
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--report", type=Path)
    p.add_argument("--skip-model", action="store_true")
    run(p.parse_args())
