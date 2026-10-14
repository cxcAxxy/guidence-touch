"""Offline real-checkpoint forward-gradient checks, not evidence of task success."""

import argparse
from dataclasses import asdict
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
import torch

from openpi.touchguide.sampler import prepare_pi05_prefix
from scripts.touchguide.verify_guided_sampler import load_fixed_inputs
from tac_forward_guidance.config import add_guidance_arguments
from tac_forward_guidance.config import config_from_args
from tac_forward_guidance.runtime import load_forward_guidance
from tac_forward_guidance.sampler import ForwardGuidanceSampler


def slice_observation(obs, start, stop):
    return replace(
        obs,
        images={k: v[start:stop] for k, v in obs.images.items()},
        image_masks={k: v[start:stop] for k, v in obs.image_masks.items()},
        state=obs.state[start:stop],
        tokenized_prompt=obs.tokenized_prompt[start:stop],
        tokenized_prompt_mask=obs.tokenized_prompt_mask[start:stop],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base-checkpoint", "cpm-checkpoint", "data-root", "inputs", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1, help="Keep full-Jacobian activation memory bounded")
    add_guidance_arguments(parser, verification=True)
    args = parser.parse_args()
    config = config_from_args(args)
    if not config.enabled or args.batch_size < 1:
        parser.error("Verification requires enabled guidance and a positive batch size")
    if not torch.cuda.is_available():
        parser.error("CUDA unavailable: run this check on a GPU node; CPU sampler tests can run separately")
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        parser.error("Use a new output directory; verification never overwrites prior artifacts")
    torch.set_num_threads(8)
    torch.manual_seed(42)
    started = time.monotonic()
    print("Loading frozen pi0.5 + original CPM; no training or weight writes", flush=True)
    loaded = load_forward_guidance(
        base_checkpoint=args.base_checkpoint,
        cpm_checkpoint=args.cpm_checkpoint,
        data_root=args.data_root,
        guidance=config,
        precision=args.precision,
    )
    sampler = loaded.sampler
    print("Models loaded; checking fixed-input stock equivalence and full input gradients", flush=True)
    device = next(sampler.base.parameters()).device
    obs, maps, noise, raw_state, manifest, alignment = load_fixed_inputs(args.inputs, device)
    if Path(manifest["checkpoint"]).resolve() != Path(loaded.metadata["base_source"]):
        raise ValueError("Fixed observations belong to another pi0.5 checkpoint")
    if any(row["offset_s"] > 1e-6 for row in alignment["frames"]):
        raise ValueError("The inherited fixed-input alignment selected future tactile data")
    torch.testing.assert_close(loaded.coordinates.normalize_state(raw_state), obs.state[..., :27], rtol=1e-5, atol=1e-5)
    disabled = ForwardGuidanceSampler(sampler.base, sampler.cpm, replace(config, scale=0.0), precision=args.precision)
    stopped = ForwardGuidanceSampler(
        sampler.base, sampler.cpm, replace(config, gradient_mode="stop_velocity"), precision=args.precision
    )
    base_actions, guided_actions, base_traces, guided_traces, updates, reports = [], [], [], [], [], []
    stock_difference, guided_seconds, jacobian_difference = 0.0, 0.0, []
    torch.cuda.reset_peak_memory_stats()
    for start in range(0, len(noise), args.batch_size):
        stop = min(start + args.batch_size, len(noise))
        current = slice_observation(obs, start, stop)
        epsilon = noise[start:stop]
        with torch.no_grad(), sampler._autocast(device):  # noqa: SLF001 -- shared base inference precision
            stock = sampler.base.sample_actions(device, current, noise=epsilon.clone(), num_steps=config.num_steps)
            prefix = prepare_pi05_prefix(sampler.base, current)
            embedding = sampler.scorer.encode_observation(
                current.images["base_0_rgb"], current.images["right_wrist_0_rgb"], maps[start:stop]
            )
        baseline = disabled.sample_cached(prefix, None, epsilon, record_trace=True)
        difference = float((stock - baseline.actions).abs().max())
        stock_difference = max(stock_difference, difference)
        if difference != 0:
            raise AssertionError(f"Scale=0 differs from stock pi0.5: {difference}")
        torch.cuda.synchronize()
        begin = time.monotonic()
        guided = sampler.sample_cached(prefix, embedding, epsilon, record_trace=True)
        torch.cuda.synchronize()
        elapsed = time.monotonic() - begin
        guided_seconds += elapsed
        active = [row for row in guided.diagnostics if row["guided"]]
        first_step = config.num_steps - config.guidance_steps
        if [row["step"] for row in active] != list(range(first_step, config.num_steps)):
            raise AssertionError("Guidance used the wrong time steps")
        if not torch.equal(baseline.trajectory[: first_step + 1], guided.trajectory[: first_step + 1]):
            raise AssertionError("Early unguided sampling changed")
        if torch.count_nonzero(guided.guidance_updates[..., 27:]):
            raise AssertionError("Guidance directly updated padding")
        if not any(row["gradient_rms"] > 0 for row in active):
            raise AssertionError("All forward gradients were zero")
        if not torch.any((guided.actions - baseline.actions)[..., :27] != 0):
            raise AssertionError("Enabled guidance did not change physical actions")
        if config.gradient_mode == "full":
            time_tensor = torch.tensor(active[0]["flow_time"], device=device, dtype=torch.float32)
            full_gradient = sampler.forward_gradient(prefix, embedding, baseline.trajectory[first_step], time_tensor)
            stop_gradient = stopped.forward_gradient(prefix, embedding, baseline.trajectory[first_step], time_tensor)
            jacobian_difference.append(float((full_gradient.gradient - stop_gradient.gradient).square().mean().sqrt()))
        for model in (sampler.base, sampler.cpm):
            if model.training or any(p.requires_grad or p.grad is not None for p in model.parameters()):
                raise AssertionError("Models must remain frozen with no parameter gradients")
        base_actions.append(baseline.actions.cpu())
        guided_actions.append(guided.actions.cpu())
        base_traces.append(baseline.trajectory.cpu())
        guided_traces.append(guided.trajectory.cpu())
        updates.append(guided.guidance_updates.cpu())
        reports.append({"input_range": [start, stop], "guided_seconds": elapsed, "steps": list(guided.diagnostics)})
        print(f"Verified inputs {start}:{stop}; guided sampling {elapsed:.3f}s", flush=True)
    baseline_actions = torch.cat(base_actions)
    actions = torch.cat(guided_actions)
    raw_cpu = raw_state.cpu()
    base_targets = loaded.coordinates.action_targets(baseline_actions, raw_cpu)
    targets = loaded.coordinates.action_targets(actions, raw_cpu)
    delta = targets - base_targets
    active_rows = [row for report in reports for row in report["steps"] if row["guided"]]
    score_pairs = [
        (before, after)
        for row in active_rows
        if "score_after_guidance_only" in row
        for before, after in zip(row["score_before"], row["score_after_guidance_only"], strict=True)
    ]
    report = {
        **loaded.metadata,
        "guidance": asdict(config),
        "inputs": str(args.inputs),
        "alignment": alignment,
        "guidance_disabled_max_abs": stock_difference,
        "both_models_frozen": True,
        "early_unguided_steps_identical": True,
        "direct_padded_guidance_max_abs": 0.0,
        "normalized_action_delta_rms": float((actions[..., :27] - baseline_actions[..., :27]).square().mean().sqrt()),
        "arm_target_delta_rms_rad": float(delta[..., :7].square().mean().sqrt()),
        "hand_target_delta_rms_rad": float(delta[..., 7:].square().mean().sqrt()),
        "target_delta_max_abs_rad": float(delta.abs().max()),
        "full_vs_stop_velocity_gradient_rms": jacobian_difference,
        "local_score_improved_pairs": sum(after > before for before, after in score_pairs),
        "local_score_pair_count": len(score_pairs),
        "update_cap_count": sum(r["capped_samples"] for r in active_rows),
        "guided_seconds": guided_seconds,
        "total_seconds": time.monotonic() - started,
        "peak_cuda_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "batches": reports,
        "scope": "offline gradient and sampler checks only; not closed-loop task success",
    }
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        output / "actions.npz",
        noise=noise.cpu().numpy(),
        tactile_map_raw=maps.cpu().numpy(),
        base_actions=baseline_actions.numpy(),
        guided_actions=actions.numpy(),
        base_targets=base_targets.numpy(),
        guided_targets=targets.numpy(),
        base_trajectory=torch.cat(base_traces, dim=1).numpy(),
        guided_trajectory=torch.cat(guided_traces, dim=1).numpy(),
        guidance_updates=torch.cat(updates, dim=1).numpy(),
    )
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
