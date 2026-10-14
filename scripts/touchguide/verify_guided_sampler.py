#!/usr/bin/env python3
"""Offline real-checkpoint validation of TouchGuide sampling; no simulation or training."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import h5py
import numpy as np
import torch

from openpi.models import model as model_api
from openpi.touchguide.inference import load_touchguide
from openpi.touchguide.sampler import GuidanceConfig
from openpi.touchguide.sampler import TouchGuideSampler


def load_fixed_inputs(inputs: Path, device: torch.device):
    """Reuse baseline observations/noise, attach nearest same-time measured right taxels."""
    manifest = json.loads(inputs.with_name("fixed_inputs_manifest.json").read_text())
    if hashlib.sha256(inputs.read_bytes()).hexdigest() != manifest["files"]["fixed_inputs"]["sha256"]:
        raise ValueError("Fixed baseline inputs changed since preparation")
    with np.load(inputs, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    camera_keys = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    observation = model_api.Observation(
        images={
            key: torch.from_numpy(arrays["image_" + key]).permute(0, 3, 1, 2).contiguous().to(device)
            for key in camera_keys
        },
        image_masks={key: torch.from_numpy(arrays["mask_" + key]).to(device) for key in camera_keys},
        state=torch.from_numpy(arrays["state"]).to(device),
        tokenized_prompt=torch.from_numpy(arrays["tokenized_prompt"]).long().to(device),
        tokenized_prompt_mask=torch.from_numpy(arrays["tokenized_prompt_mask"]).to(device),
    )
    names = [
        "hand_r_thumb_link6",
        "hand_r_index_link4",
        "hand_r_middle_link4",
        "hand_r_ring_link4",
        "hand_r_pinky_link4",
    ]
    maps, alignment = [], []
    source = manifest["files"]["source_h5"]["path"]
    with h5py.File(source, "r") as episode:
        tactile = episode["tactile_contact_force"]
        links = tactile["link_names"].asstr()[:].tolist()
        right_indices = [links.index(name) for name in names]
        timestamps = np.asarray(tactile["timestamp"])
        tolerance = float(np.median(np.diff(timestamps))) * 0.51 + 1e-6
        for frame in manifest["frames"]:
            query = frame["sim_time_s"]
            index = int(np.argmin(np.abs(timestamps - query)))
            offset = float(timestamps[index] - query)
            if abs(offset) > tolerance:
                raise ValueError("Recorded tactile cannot be aligned to this baseline observation")
            normal = np.asarray(tactile["normal_taxel_force_n"][index])[right_indices]
            tangent = np.asarray(tactile["tangent_taxel_force_n"][index])[right_indices]
            maps.append(np.stack([normal, tangent[..., 0], tangent[..., 1]], axis=1))
            alignment.append({"sim_time_s": query, "tactile_row": index, "offset_s": offset})
    raw_maps = torch.from_numpy(np.stack(maps).astype(np.float32)).to(device)
    noise = torch.from_numpy(arrays["noise"]).to(device)
    state_raw = torch.from_numpy(arrays["raw_state"]).to(device)
    return observation, raw_maps, noise, state_raw, manifest, {"source_h5": source, "frames": alignment}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--cpm-checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--guidance-steps", type=int, default=3)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--precision", choices=("bfloat16", "float32"), default="bfloat16")
    args = parser.parse_args()
    guidance = GuidanceConfig(args.num_steps, args.guidance_steps, args.guidance_scale)
    if not guidance.enabled:
        raise ValueError("This check needs positive guidance strength and at least one guided step")
    torch.set_num_threads(8)
    torch.manual_seed(42)
    started = time.monotonic()
    print("Loading frozen pi0.5 + CPM and checking their inference dependencies", flush=True)
    loaded = load_touchguide(
        base_checkpoint=args.base_checkpoint,
        cpm_checkpoint=args.cpm_checkpoint,
        data_root=args.data_root,
        guidance=guidance,
        precision=args.precision,
    )
    print("Models loaded:", json.dumps(loaded.metadata), flush=True)
    sampler = loaded.sampler
    device = next(sampler.base.parameters()).device
    obs, maps, noise, state_raw, manifest, alignment = load_fixed_inputs(args.inputs, device)
    if Path(manifest["checkpoint"]).resolve() != Path(loaded.metadata["base_source"]):
        raise ValueError("Fixed observations were transformed for another pi0.5 checkpoint")
    original_state = loaded.coordinates.normalize_state(state_raw)
    torch.testing.assert_close(original_state, obs.state[..., :27], rtol=1e-5, atol=1e-5)
    print("Checking stock sampler against guidance scale=0", flush=True)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.precision == "bfloat16"):
        stock = sampler.base.sample_actions(device, obs, noise=noise.clone(), num_steps=guidance.num_steps)
    disabled = TouchGuideSampler(
        sampler.base,
        sampler.cpm,
        GuidanceConfig(guidance.num_steps, guidance.guidance_steps, 0.0),
        precision=args.precision,
    )
    baseline = disabled.sample(obs, noise=noise, record_trace=True)
    zero_difference = float((baseline.actions - stock).abs().max())
    if zero_difference != 0:
        raise AssertionError(f"Disabled guidance changed the original pi0.5 sampler: {zero_difference}")
    print("Scale=0 exactly matches stock. Checking actual CPM action gradients", flush=True)
    torch.cuda.synchronize()
    guided_started = time.monotonic()
    guided = sampler.sample(obs, maps, noise=noise, record_trace=True)
    torch.cuda.synchronize()
    guided_seconds = time.monotonic() - guided_started
    active = [row for row in guided.diagnostics if row["guided"]]
    expected_steps = list(range(guidance.num_steps - guidance.guidance_steps, guidance.num_steps))
    if [row["step"] for row in active] != expected_steps:
        raise AssertionError("Guidance used the wrong denoising steps")
    if not torch.equal(guided.trajectory[: expected_steps[0] + 1], baseline.trajectory[: expected_steps[0] + 1]):
        raise AssertionError("The initial unguided steps changed")
    if any(row["gradient_rms"] <= 0 for row in active):
        raise AssertionError("CPM action gradients must be nonzero")
    if torch.count_nonzero(guided.guidance_updates[..., 27:]):
        raise AssertionError("Guidance directly modified padded action coordinates")
    for model in (sampler.base, sampler.cpm):
        if model.training or any(
            parameter.requires_grad or parameter.grad is not None for parameter in model.parameters()
        ):
            raise AssertionError("Model parameters were updated or a training graph was retained")
    normalized_delta = guided.actions - baseline.actions
    if not torch.any(normalized_delta[..., :27] != 0):
        raise AssertionError("Enabled guidance did not change any physical action coordinates")
    base_targets = loaded.coordinates.action_targets(baseline.actions, state_raw)
    guided_targets = loaded.coordinates.action_targets(guided.actions, state_raw)
    physical_delta = guided_targets - base_targets
    summary = {
        **loaded.metadata,
        "guidance": asdict(guidance),
        "inputs": str(args.inputs),
        "alignment": alignment,
        "guidance_disabled_max_abs": zero_difference,
        "both_models_frozen": True,
        "early_unguided_steps_identical": True,
        "direct_padded_guidance_max_abs": 0.0,
        "normalized_action_delta_rms": float(normalized_delta[..., :27].square().mean().sqrt()),
        "normalized_action_delta_max_abs": float(normalized_delta[..., :27].abs().max()),
        "arm_target_delta_rms_rad": float(physical_delta[..., :7].square().mean().sqrt()),
        "hand_target_delta_rms_rad": float(physical_delta[..., 7:].square().mean().sqrt()),
        "guided_batch_size": len(noise),
        "local_score_improved_pairs": sum(
            after > before
            for row in active
            for before, after in zip(row["score_before"], row["score_after_guidance_only"], strict=True)
        ),
        "local_score_pair_count": len(noise) * len(active),
        "guided_seconds": guided_seconds,
        "total_seconds": time.monotonic() - started,
        "peak_cuda_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "steps": list(guided.diagnostics),
        "scope": "offline sampler checks; these scores do not establish closed-loop task success",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "actions.npz",
        noise=noise.cpu().numpy(),
        tactile_map_raw=maps.cpu().numpy(),
        raw_state=state_raw.cpu().numpy(),
        base_actions=baseline.actions.cpu().numpy(),
        guided_actions=guided.actions.cpu().numpy(),
        base_targets=base_targets.cpu().numpy(),
        guided_targets=guided_targets.cpu().numpy(),
        base_trajectory=baseline.trajectory.cpu().numpy(),
        guided_trajectory=guided.trajectory.cpu().numpy(),
        guidance_updates=guided.guidance_updates.cpu().numpy(),
    )
    (args.output_dir / "report.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print("Offline check saved to", args.output_dir, flush=True)


if __name__ == "__main__":
    main()
