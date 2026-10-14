"""Train the TouchGuide-inspired CPM; this entry does not load a pi0.5 policy."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
from itertools import islice
import json
import logging
import math
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import torch

from openpi.touchguide.batching import CPMBatchLoader
from openpi.touchguide.batching import EpochIndexBatchSampler
from openpi.touchguide.batching import regroup_fixed_validation_batches
from openpi.touchguide.checkpoint import capture_rng_state
from openpi.touchguide.checkpoint import load_checkpoint
from openpi.touchguide.checkpoint import restore_rng_state
from openpi.touchguide.checkpoint import save_checkpoint
from openpi.touchguide.losses import cpm_contrastive_loss
from openpi.touchguide.model import encode_observation_batch
from openpi.touchguide.model import load_cpm_model
from openpi.touchguide.noise import FlowActionNoise
from openpi.touchguide.noise import NoiseAugmentedBatchLoader
from openpi.touchguide.noise import NoiseConfig

LOGGER = logging.getLogger("touchguide.cpm")


class WeightedMetrics:
    """Accumulate batch means with their actual sample counts, including the tail."""

    def __init__(self) -> None:
        self.total: dict[str, float] = {}
        self.count = 0

    def add(self, values: dict[str, Any], size: int) -> None:
        if size < 1:
            raise ValueError("Metric weight must be positive")
        for key, value in values.items():
            number = (
                float(value.detach().item())
                if isinstance(value, torch.Tensor)
                else float(value)
            )
            if not math.isfinite(number):
                raise FloatingPointError(f"Nonfinite metric: {key}")
            self.total[key] = self.total.get(key, 0.0) + number * size
        self.count += size

    def mean(self) -> dict[str, float]:
        if not self.count:
            raise ValueError("No samples accumulated")
        return {key: value / self.count for key, value in self.total.items()}

    def state(self) -> dict[str, Any]:
        return {"total": dict(self.total), "count": self.count}

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> WeightedMetrics:
        result = cls()
        result.total = {key: float(value) for key, value in state["total"].items()}
        result.count = int(state["count"])
        return result


def lr_factor(
    step: int, *, total_steps: int, warmup_steps: int, min_ratio: float
) -> float:
    """Multiplier for the upcoming optimizer update, with step zero being the first."""
    if (
        total_steps < 1
        or not 0 <= warmup_steps < total_steps
        or not 0 <= min_ratio <= 1
        or step < 0
    ):
        raise ValueError("Invalid learning-rate schedule")
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    fraction = (
        1.0
        if total_steps - warmup_steps == 1
        else min(1.0, (step - warmup_steps) / (total_steps - warmup_steps - 1))
    )
    return min_ratio + (1 - min_ratio) * (1 + math.cos(math.pi * fraction)) / 2


def autocast(device: torch.device, precision: str):
    if precision == "bf16":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return nullcontext()


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }


def output_from_embeddings(
    model, observation: torch.Tensor, action: torch.Tensor
) -> dict[str, torch.Tensor]:
    # Autocast would otherwise cast the matrix product back down to BF16.
    with torch.autocast(device_type=observation.device.type, enabled=False):
        cosine = observation.float() @ action.float().T
        temperature = model.temperature.float()
        return {
            "observation_embedding": observation,
            "action_embedding": action,
            "cosine_scores": cosine,
            "logits": cosine / temperature,
            "temperature": temperature,
        }


def validate(
    model,
    clean_loader,
    augmenter: FlowActionNoise,
    settings: dict[str, Any],
    device: torch.device,
    precision: str,
    *,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Fixed validation identities/epsilon, reusing one observation encoding per group."""
    times = settings["validation_times"]
    expected = [
        level / augmenter.config.num_flow_steps
        for level in range(augmenter.config.num_flow_steps)
    ]
    if times != expected or not augmenter.config.enabled:
        raise ValueError(
            "Validation must cover the enabled training noise grid in order"
        )
    meters = [WeightedMetrics() for _ in times]
    previous_mode, rng_state = model.training, capture_rng_state()
    model.eval()
    start = time.monotonic()
    baseline = WeightedMetrics()
    try:
        with torch.no_grad():
            batches = (
                clean_loader
                if max_batches is None
                else islice(clean_loader, max_batches)
            )
            for index, clean in enumerate(batches):
                fixed = augmenter.augment_validation_at_time(
                    clean, 0.0, seed=settings["seed"]
                )
                batch = move_batch(fixed, device)
                with autocast(device, precision):
                    observation = encode_observation_batch(model, batch)
                    for level, flow_time in enumerate(times):
                        noisy = (1 - flow_time) * batch["actions"] + flow_time * batch[
                            "noise_epsilon"
                        ]
                        action = model.action_encoder(noisy, flow_time)
                        output = output_from_embeddings(model, observation, action)
                        from openpi.touchguide.negative_pairing import attach_pairing  # noqa: PLC0415

                        output = attach_pairing(output, model, batch)
                        meters[level].add(
                            cpm_contrastive_loss(output, batch), len(noisy)
                        )
                baseline.add(
                    {
                        "chance_accuracy": 1 / len(batch["actions"]),
                        "uninformed_loss": math.log(len(batch["actions"])),
                        **(
                            {
                                "weighted_uninformed_loss": float(
                                    (
                                        output["negative_weights"].sum(1).log().mean()
                                        + output["negative_weights"].sum(0).log().mean()
                                    )
                                    / 2
                                )
                            }
                            if "negative_weights" in output
                            else {}
                        ),
                    },
                    len(batch["actions"]),
                )
                if index % 10 == 0:
                    LOGGER.info("Validation group %d/%d", index + 1, len(clean_loader))
    finally:
        model.train(previous_mode)
        restore_rng_state(rng_state)
    levels = [meter.mean() for meter in meters]
    weights = augmenter.probabilities.tolist()
    result = {
        f"val/{key}": sum(
            weight * level[key] for weight, level in zip(weights, levels, strict=True)
        )
        for key in levels[0]
    }
    result.update(
        {
            f"val_t{index:02d}/{key}": value
            for index, level in enumerate(levels)
            for key, value in level.items()
        }
    )
    result.update({f"val/{key}": value for key, value in baseline.mean().items()})
    result["val/grid_mean_loss"] = sum(level["loss"] for level in levels) / len(levels)
    late = [
        level["loss"]
        for flow_time, level in zip(times, levels, strict=True)
        if 0 < flow_time <= 0.4
    ]
    if late:
        result["val/late_loss"] = sum(late) / len(late)
    result["val/samples"] = meters[0].count
    result["val/seconds"] = time.monotonic() - start
    return result


def build_loaders(args):
    use_state = getattr(args, "model_variant", "cpm") in (
        "state-cpm",
        "recovery-state-cpm",
    )
    clean = CPMBatchLoader.from_config(
        args.root,
        "train",
        include_images=True,
        include_auxiliary=use_state,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    sampler = EpochIndexBatchSampler(
        len(clean.dataset), args.batch_size, seed=args.seed, drop_last=True
    )
    train_clean = CPMBatchLoader(
        clean.dataset,
        sampler,
        include_auxiliary=use_state,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    settings = json.loads((args.root / "config/train.json").read_text())[
        "noise_augmentation"
    ]
    if (
        settings["application_stage"] != "after_clean_batch_collation"
        or settings["mode"] != "random_per_batch"
    ):
        raise ValueError("Training requires the prepared batch noise configuration")
    augmenter = FlowActionNoise(NoiseConfig.from_dict(settings))
    train = NoiseAugmentedBatchLoader(
        train_clean, augmenter, split="train", seed=args.seed
    )
    prepared_val = CPMBatchLoader.from_config(
        args.root,
        "val",
        include_images=True,
        include_auxiliary=use_state,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    requested_val_batch_size = getattr(args, "val_batch_size", None)
    val = (
        prepared_val
        if requested_val_batch_size is None
        else CPMBatchLoader(
            prepared_val.dataset,
            regroup_fixed_validation_batches(
                prepared_val.index_batches, requested_val_batch_size
            ),
            include_auxiliary=use_state,
            num_workers=args.num_workers,
            pin_memory=True,
        )
    )
    val_settings = json.loads((args.root / "config/val.json").read_text())[
        "noise_augmentation"
    ]
    if (
        NoiseConfig.from_dict(val_settings) != augmenter.config
        or val_settings["mode"] != "fixed_per_sample_id"
    ):
        raise ValueError("Validation and training noise settings differ")
    return train, val, val_settings


def make_contract(args, train) -> dict[str, Any]:
    files = [
        "config/cpm.json",
        "config/train.json",
        "config/val.json",
        "index/train.npz",
        "index/val.npz",
        "index/val_batches.npz",
        "meta/tactile_scale.json",
        "meta/action_norm_stats.json",
        "meta/entry_config.json",
        "meta/splits.json",
    ]
    contract = {
        "dataset_root": str(args.root),
        "data_sha256": {
            name: hashlib.sha256((args.root / name).read_bytes()).hexdigest()
            for name in files
        },
        "batch_size": args.batch_size,
        "seed": args.seed,
        "epochs": args.epochs,
        "steps_per_epoch": len(train),
        "noise": asdict(train.augmenter.config),
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "min_lr_ratio": args.min_lr_ratio,
        "grad_clip": args.grad_clip,
        "precision": args.precision,
        "best_metric": "val/loss",
        "validation_schedule": "complete_grid_at_epoch_end",
    }
    # Omit this key for runs using the prepared validation groups, preserving
    # the exact contract of checkpoints created before --val-batch-size existed.
    if getattr(args, "val_batch_size", None) is not None:
        contract["validation_batch_size"] = args.val_batch_size
    if getattr(args, "model_variant", "cpm") == "state-cpm":
        from openpi.touchguide.state_model import STATE_CPM_SPEC  # noqa: PLC0415

        contract["model_variant"] = "state-cpm"
        contract["state_cpm"] = dict(STATE_CPM_SPEC)
        contract["state_cpm"]["observation_tokens"] = 457
    if getattr(args, "model_variant", "cpm") == "recovery-state-cpm":
        settings = json.loads((args.root / "config/cpm.json").read_text())
        contract["model_variant"] = "recovery-state-cpm"
        contract["recovery_state_cpm"] = settings["recovery"]
        for name in ("meta/cpm_normalization_v1.json", "meta/negative_policy.json"):
            contract["data_sha256"][name] = hashlib.sha256(
                (args.root / name).read_bytes()
            ).hexdigest()
    return contract


def check_gpu(args, model, train, val, val_settings, device) -> dict[str, Any]:
    """One real CUDA forward/backward; no optimizer update, W&B run or checkpoint."""
    torch.cuda.reset_peak_memory_stats(device)
    start = time.monotonic()
    model.train()
    LOGGER.info("GPU check: reading %d actual training samples", args.batch_size)
    batch = move_batch(next(iter(train)), device)
    torch.cuda.synchronize()
    read_seconds = time.monotonic() - start
    start = time.monotonic()
    with autocast(device, args.precision):
        metrics = cpm_contrastive_loss(model(batch), batch)
    metrics["loss"].backward()
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if any(parameter.grad is None for parameter in parameters):
        raise RuntimeError("A trainable parameter has no gradient")
    norm = torch.nn.utils.clip_grad_norm_(
        parameters, args.grad_clip, error_if_nonfinite=True
    )
    if any(not torch.isfinite(parameter.grad).all() for parameter in parameters):
        raise FloatingPointError("A trainable gradient is not finite")
    if any(parameter.grad is not None for parameter in model.vision.parameters()):
        raise RuntimeError("Frozen DINO received a gradient")
    torch.cuda.synchronize()
    forward_backward_seconds = time.monotonic() - start
    train_values = {key: float(value.detach().item()) for key, value in metrics.items()}
    model.zero_grad(set_to_none=True)
    model.eval()
    contact_index = batch["tactile_map_raw"].abs().flatten(1).amax(1).argmax().item()
    selection = slice(contact_index, contact_index + 1)
    selected = {
        key: value[selection]
        for key, value in batch.items()
        if isinstance(value, torch.Tensor)
        and key in ("head", "wrist", "tactile_map_raw", "state_raw")
    }
    with torch.no_grad(), autocast(device, args.precision):
        observation = encode_observation_batch(model, selected)
        empty_observation = encode_observation_batch(
            model,
            {
                **selected,
                "tactile_map_raw": torch.zeros_like(selected["tactile_map_raw"]),
            },
        )
    candidate = (
        batch["noisy_actions"][selection].detach().requires_grad_(requires_grad=True)
    )
    with autocast(device, args.precision):
        score = model.paired_scores(observation, candidate, batch["flow_time"])
    action_gradient = torch.autograd.grad(score.sum(), candidate)[0]
    if (
        not torch.isfinite(action_gradient).all()
        or action_gradient.abs().max().item() == 0
    ):
        raise RuntimeError(
            "The raw CPM score must have a finite, nonzero action gradient"
        )
    guidance = {
        "raw_score": float(score.detach().item()),
        "action_gradient_norm": float(action_gradient.norm().item()),
        "maximum_force_n": float(
            batch["tactile_map_raw"][selection].abs().max().item()
        ),
        "observation_change_without_touch": float(
            (observation - empty_observation).norm().item()
        ),
    }
    if (
        guidance["maximum_force_n"] > 0
        and guidance["observation_change_without_touch"] == 0
    ):
        raise RuntimeError(
            "Removing nonzero tactile input did not change the observation encoding"
        )
    del (
        batch,
        metrics,
        observation,
        empty_observation,
        candidate,
        score,
        action_gradient,
    )
    validation = validate(
        model, val, train.augmenter, val_settings, device, args.precision, max_batches=1
    )
    result = {
        "schema": "touchguide_cpm_gpu_check_v2",
        "gpu": torch.cuda.get_device_name(device),
        "batch_size": args.batch_size,
        "precision": args.precision,
        "parameters": model.parameter_counts(),
        "train_metrics": train_values,
        "guidance_interface": guidance,
        "gradient_norm_before_clip": float(norm),
        "validation": validation,
        "read_seconds": read_seconds,
        "forward_backward_seconds": forward_backward_seconds,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
        "optimizer_state_estimated_mib": model.parameter_counts()["trainable"]
        * 8
        / 2**20,
        "optimizer_step": False,
        "wandb_run_created": False,
        "training_started": False,
        "validation_groups_checked": 1,
        "note": "Backward check; AdamW state has not been allocated.",
    }
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + "\n")
    LOGGER.info(
        "GPU check passed: allocated peak %.3f GiB, reserved peak %.3f GiB",
        result["peak_allocated_gib"],
        result["peak_reserved_gib"],
    )
    print(json.dumps(result, indent=2))
    return result


class Recorder:
    def __init__(self, output: Path, wandb_run):
        self.path, self.run = output / "metrics.jsonl", wandb_run

    def log(self, step: int, values: dict[str, Any]) -> None:
        payload = {"step": step, **values}
        with self.path.open("a") as file:
            file.write(json.dumps(payload, allow_nan=False) + "\n")
        if self.run is not None:
            self.run.log(payload)


def init_wandb(args, output: Path, contract, counts, previous_run):
    if args.wandb_mode == "disabled":
        return None, previous_run
    import wandb  # noqa: PLC0415 -- optional dependency; diagnostic creates no run.

    previous_run = previous_run or {}
    entity, project = args.wandb_entity, args.wandb_project
    if previous_run and (
        previous_run["entity"] != entity or previous_run["project"] != project
    ):
        raise ValueError("Resume must keep the original W&B entity and project")
    run = wandb.init(
        entity=entity,
        project=project,
        name=previous_run.get("name", args.run_name),
        id=previous_run.get("id"),
        resume="must" if previous_run and args.wandb_mode == "online" else None,
        mode=args.wandb_mode,
        dir=str(output),
        config={**contract, "parameters": counts},
        tags=["recovery-state-cpm"]
        if getattr(args, "model_variant", "cpm") == "recovery-state-cpm"
        else (
            ["state-CPM"]
            if getattr(args, "model_variant", "cpm") == "state-cpm"
            else None
        ),
    )
    run.define_metric("step")
    run.define_metric("*", step_metric="step")
    return run, {"id": run.id, "name": run.name, "entity": entity, "project": project}


def train_model(args, model, train, val, val_settings, device) -> None:
    contract = make_contract(args, train)
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        parameters, lr=args.lr, weight_decay=args.weight_decay
    )
    total_steps = args.epochs * len(train)
    warmup = min(total_steps - 1, round(total_steps * args.warmup_ratio))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: lr_factor(
            step,
            total_steps=total_steps,
            warmup_steps=warmup,
            min_ratio=args.min_lr_ratio,
        ),
    )
    progress = {"epoch": 0, "batch_in_epoch": 0, "global_step": 0, "best_val": None}
    resumed, run_info, rng_to_restore = None, None, None
    if args.resume is not None:
        resumed = load_checkpoint(
            args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_contract=contract,
        )
        progress, run_info, rng_to_restore = (
            resumed["progress"],
            resumed["run"],
            resumed["rng_state"],
        )
        if (
            not 0 <= progress["epoch"] <= args.epochs
            or not 0 <= progress["batch_in_epoch"] <= len(train)
            or progress["global_step"]
            != progress["epoch"] * len(train) + progress["batch_in_epoch"]
        ):
            raise ValueError("Checkpoint progress differs from the training sampler")
        if args.resume.parent.resolve() != args.output_dir.resolve():
            raise ValueError("Resume into the original checkpoint output directory")
    elif args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            "Fresh training requires an empty or new output directory"
        )
    if progress["epoch"] == args.epochs and progress["batch_in_epoch"] == 0:
        raise ValueError(
            "This run has already completed its configured epochs and final validation"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.addHandler(logging.FileHandler(args.output_dir / "training.log"))
    if resumed is None:
        (args.output_dir / "run_config.json").write_text(
            json.dumps(
                {
                    "contract": contract,
                    "arguments": {
                        key: str(value) if isinstance(value, Path) else value
                        for key, value in vars(args).items()
                    },
                    "parameters": model.parameter_counts(),
                    "warmup_steps": warmup,
                    "total_steps": total_steps,
                    "validation_times": val_settings["validation_times"],
                },
                indent=2,
            )
            + "\n"
        )
    run, run_info = init_wandb(
        args, args.output_dir, contract, model.parameter_counts(), run_info
    )
    recorder = Recorder(args.output_dir, run)
    interval = WeightedMetrics.from_state(
        progress.get("interval_metrics", {"total": {}, "count": 0})
    )
    epoch_meter = WeightedMetrics.from_state(
        progress.get("epoch_metrics", {"total": {}, "count": 0})
    )
    start_epoch, skip = progress["epoch"], progress["batch_in_epoch"]
    success = False

    def save(path: Path):
        progress["interval_metrics"] = interval.state()
        progress["epoch_metrics"] = epoch_meter.state()
        save_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            progress=progress,
            contract=contract,
            run=run_info,
        )

    LOGGER.info(
        "CPM: train=%d samples, val=%d samples, %d batches/epoch, %d updates; output=%s",
        len(train.dataset),
        len(val.dataset),
        len(train),
        total_steps,
        args.output_dir,
    )
    try:
        for epoch in range(start_epoch, args.epochs):
            model.train()
            train.set_epoch(epoch)
            epoch_start = time.monotonic()
            # Epoch-boundary checkpoints precede iterator construction. Preserve
            # its normal RNG consumption; only mid-epoch resumes replay a prefix.
            if rng_to_restore is not None and skip == 0:
                restore_rng_state(rng_to_restore)
                rng_to_restore = None
            for batch_index, cpu_batch in enumerate(train):
                if epoch == start_epoch and batch_index < skip:
                    continue
                # Iterator creation / prefix replay may consume global CPU RNG.
                # Noise uses an independent seeded generator, replayed with the prefix.
                if rng_to_restore is not None:
                    restore_rng_state(rng_to_restore)
                    rng_to_restore = None
                batch = move_batch(cpu_batch, device)
                optimizer.zero_grad(set_to_none=True)
                used_lr = optimizer.param_groups[0]["lr"]
                with autocast(device, args.precision):
                    metrics = cpm_contrastive_loss(model(batch), batch)
                metrics["loss"].backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    parameters, args.grad_clip, error_if_nonfinite=True
                )
                optimizer.step()
                scheduler.step()
                size = len(batch["actions"])
                values = {
                    key: float(value.detach().item()) for key, value in metrics.items()
                }
                values.update(lr=used_lr, grad_norm=float(norm))
                interval.add(values, size)
                epoch_meter.add(values, size)
                progress.update(
                    epoch=epoch,
                    batch_in_epoch=batch_index + 1,
                    global_step=progress["global_step"] + 1,
                )
                step = progress["global_step"]
                if step % args.log_interval == 0:
                    logged = {
                        f"train/{key}": value for key, value in interval.mean().items()
                    }
                    logged.update(
                        epoch=epoch + 1,
                        **{
                            "train/interval_samples": interval.count,
                            "train/last_noise_time": float(batch["flow_time"]),
                        },
                    )
                    recorder.log(step, logged)
                    LOGGER.info(
                        "epoch %d/%d step %d/%d loss %.5f obs→action %.3f gap %.4f lr %.2g",
                        epoch + 1,
                        args.epochs,
                        step,
                        total_steps,
                        logged["train/loss"],
                        logged["train/accuracy_obs_to_action"],
                        logged["train/score_gap"],
                        used_lr,
                    )
                    interval = WeightedMetrics()
                if args.save_interval > 0 and step % args.save_interval == 0:
                    save(args.output_dir / f"step_{step:06d}.pt")
                    save(args.output_dir / "last.pt")
                del batch, metrics
            if rng_to_restore is not None:
                restore_rng_state(rng_to_restore)
                rng_to_restore = None
            validation = validate(
                model, val, train.augmenter, val_settings, device, args.precision
            )
            stats = {
                f"train_epoch/{key}": value for key, value in epoch_meter.mean().items()
            }
            stats.update(validation)
            stats.update(
                epoch=epoch + 1,
                **{
                    "train_epoch/samples": epoch_meter.count,
                    "train_epoch/seconds": time.monotonic() - epoch_start,
                },
            )
            if interval.count:
                stats.update(
                    {f"train/{key}": value for key, value in interval.mean().items()}
                )
                stats["train/interval_samples"] = interval.count
                interval = WeightedMetrics()
            recorder.log(progress["global_step"], stats)
            save_epoch = (epoch + 1) % getattr(
                args, "save_every_epochs", 1
            ) == 0 or epoch + 1 == args.epochs
            improved = save_epoch and (
                progress["best_val"] is None
                or validation["val/loss"] < progress["best_val"]
            )
            if improved:
                progress["best_val"] = validation["val/loss"]
                progress["best_step"] = progress["global_step"]
            LOGGER.info(
                "epoch %d validation loss %.5f accuracy %.3f best saved %s",
                epoch + 1,
                validation["val/loss"],
                validation["val/accuracy_obs_to_action"],
                f"{progress['best_val']:.5f}"
                if progress["best_val"] is not None
                else "pending",
            )
            progress.update(epoch=epoch + 1, batch_in_epoch=0)
            epoch_meter = WeightedMetrics()
            if improved:
                save(args.output_dir / "best.pt")
            if save_epoch:
                save(args.output_dir / "last.pt")
            if save_epoch and run is not None and progress["best_val"] is not None:
                run.summary.update(
                    {
                        "best_val_loss": progress["best_val"],
                        "best_step": progress.get("best_step"),
                        "best_checkpoint": str(args.output_dir / "best.pt"),
                    }
                )
        success = True
    except KeyboardInterrupt:
        LOGGER.info(
            "Interrupted: retaining checkpoints saved after completed optimizer steps"
        )
        # Interrupt can occur halfway through a forward/update; never label that state resumable.
        raise
    finally:
        if run is not None:
            run.finish(exit_code=0 if success else 1)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--run-name", default="touchguide_cpm")
    parser.add_argument(
        "--model-variant",
        choices=("cpm", "state-cpm", "recovery-state-cpm"),
        default="cpm",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--save-interval", type=int, default=0)
    parser.add_argument("--save-every-epochs", type=int, default=20)
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"), default="online"
    )
    parser.add_argument(
        "--wandb-entity", default="xianchichen8-hunan-university-library"
    )
    parser.add_argument("--wandb-project", default="poker-draw-touchguide-cpm")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if (
        args.batch_size < 2
        or (args.val_batch_size is not None and args.val_batch_size < 2)
        or args.epochs < 1
        or args.num_workers < 0
        or args.seed < 0
        or args.log_interval < 1
        or args.save_interval < 0
        or args.save_every_epochs < 1
        or not args.lr > 0
        or not args.weight_decay >= 0
        or not args.grad_clip > 0
        or not 0 <= args.warmup_ratio < 1
        or not 0 <= args.min_lr_ratio <= 1
        or not all(
            math.isfinite(value)
            for value in (args.lr, args.weight_decay, args.grad_clip)
        )
    ):
        parser.error("Invalid training settings")
    if not args.check_only and args.output_dir is None:
        parser.error("Training requires --output-dir")
    if args.check_only and args.resume is not None:
        parser.error("GPU check starts fresh and cannot resume training")
    args.root = args.root.expanduser().resolve(strict=True)
    if args.output_dir is not None:
        args.output_dir = args.output_dir.expanduser().resolve()
    if args.resume is not None:
        args.resume = args.resume.expanduser().resolve(strict=True)
    return args


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout
    )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CPM training/check requires a CUDA GPU; use the vtla Python environment"
        )
    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This GPU does not support BF16; select --precision fp32")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:0")
    LOGGER.info(
        "Loading verified local DINOv2 and CPM on %s",
        torch.cuda.get_device_name(device),
    )
    LOGGER.info("Model variant: %s", args.model_variant)
    model = load_cpm_model(args.root, args.model_variant).to(device)
    train, val, val_settings = build_loaders(args)
    try:
        if args.check_only:
            check_gpu(args, model, train, val, val_settings, device)
        else:
            train_model(args, model, train, val, val_settings, device)
    finally:
        train.dataset.close()
        val.dataset.close()


if __name__ == "__main__":
    main()
