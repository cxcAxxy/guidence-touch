"""DDP CPM training with global contrastive pairs and NAS rank-zero checkpoints."""

from __future__ import annotations

from itertools import islice
import json
import logging
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from openpi.touchguide import train as base
from openpi.touchguide.batching import CPMBatchLoader
from openpi.touchguide.batching import EpochIndexBatchSampler
from openpi.touchguide.batching import regroup_fixed_validation_batches
from openpi.touchguide.checkpoint import load_checkpoint
from openpi.touchguide.checkpoint import save_checkpoint
from openpi.touchguide.distributed import BroadcastFlowActionNoise
from openpi.touchguide.distributed import DistContext
from openpi.touchguide.distributed import GlobalBatchShardSampler
from openpi.touchguide.distributed import capture_distributed_rng
from openpi.touchguide.distributed import capture_local_rng_state
from openpi.touchguide.distributed import gather_model_output
from openpi.touchguide.distributed import initialize
from openpi.touchguide.distributed import restore_distributed_rng
from openpi.touchguide.distributed import restore_local_rng_state
from openpi.touchguide.losses import cpm_contrastive_loss
from openpi.touchguide.model import encode_observation_batch
from openpi.touchguide.model import load_cpm_model
from openpi.touchguide.noise import NoiseAugmentedBatchLoader
from openpi.touchguide.noise import NoiseConfig

LOGGER = logging.getLogger("touchguide.cpm.ddp")
METRIC_KEYS = (
    "loss",
    "loss_obs_to_action",
    "loss_action_to_obs",
    "accuracy_obs_to_action",
    "accuracy_action_to_obs",
    "positive_score",
    "negative_score",
    "score_gap",
    "temperature",
)


def primary_action(context: DistContext, function):
    """Propagate rank-zero IO errors before other ranks enter another collective."""
    payload = [None, None]
    if context.is_primary:
        try:
            payload[1] = function()
        except Exception as error:
            payload[0] = f"{type(error).__name__}: {error}"
    if context.world_size > 1:
        dist.broadcast_object_list(payload, src=0)
    if payload[0] is not None:
        raise RuntimeError(f"Primary rank operation failed: {payload[0]}")
    return payload[1]


def build_loaders(args, context: DistContext):
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
    global_sampler = EpochIndexBatchSampler(
        len(clean.dataset), args.batch_size, seed=args.seed, drop_last=True
    )
    sampler = GlobalBatchShardSampler(
        global_sampler, rank=context.rank, world_size=context.world_size
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
        raise ValueError("Use the prepared clean training batch and noise settings")
    augmenter = BroadcastFlowActionNoise(NoiseConfig.from_dict(settings))
    train = NoiseAugmentedBatchLoader(
        train_clean, augmenter, split="train", seed=args.seed + context.rank
    )
    complete_val = CPMBatchLoader.from_config(
        args.root,
        "val",
        include_images=True,
        include_auxiliary=use_state,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    requested_val_batch_size = getattr(args, "val_batch_size", None)
    fixed_groups = (
        complete_val.index_batches
        if requested_val_batch_size is None
        else regroup_fixed_validation_batches(
            complete_val.index_batches, requested_val_batch_size
        )
    )
    groups = [
        group
        for index, group in enumerate(fixed_groups)
        if index % context.world_size == context.rank
    ]
    val = CPMBatchLoader(
        complete_val.dataset,
        groups,
        include_auxiliary=use_state,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    if args.num_workers:
        # NCCL must not be inherited by forked DataLoader workers.
        train_clean.multiprocessing_context = "spawn"
        val.multiprocessing_context = "spawn"
    val_settings = json.loads((args.root / "config/val.json").read_text())[
        "noise_augmentation"
    ]
    if (
        NoiseConfig.from_dict(val_settings) != augmenter.config
        or val_settings["mode"] != "fixed_per_sample_id"
    ):
        raise ValueError("Validation and training noise settings differ")
    return train, val, val_settings


def reduce_meter(meter: base.WeightedMetrics, keys, context: DistContext):
    values = torch.tensor(
        [meter.total.get(key, 0.0) for key in keys] + [meter.count],
        dtype=torch.float64,
        device=context.device,
    )
    if context.world_size > 1:
        dist.all_reduce(values)
    count = int(values[-1].item())
    if count < 1:
        raise ValueError("Distributed validation has no samples")
    return {
        key: float(values[index].item()) / count for index, key in enumerate(keys)
    }, count


def validate(
    model,
    clean_loader,
    augmenter,
    settings,
    context: DistContext,
    precision: str,
    *,
    max_batches: int | None = None,
):
    times = settings["validation_times"]
    if times != [
        level / augmenter.config.num_flow_steps
        for level in range(augmenter.config.num_flow_steps)
    ]:
        raise ValueError("Validation must cover the complete noise grid")
    meters = [base.WeightedMetrics() for _ in times]
    baseline = base.WeightedMetrics()
    previous_mode, rng = model.training, capture_local_rng_state(context.device)
    model.eval()
    start = time.monotonic()
    error_text = None
    try:
        with torch.no_grad():
            batches = (
                clean_loader
                if max_batches is None
                else islice(clean_loader, max_batches)
            )
            for clean in batches:
                batch = base.move_batch(
                    augmenter.augment_validation_at_time(
                        clean, 0.0, seed=settings["seed"]
                    ),
                    context.device,
                )
                with base.autocast(context.device, precision):
                    observation = encode_observation_batch(model, batch)
                    for level, flow_time in enumerate(times):
                        noisy = (1 - flow_time) * batch["actions"] + flow_time * batch[
                            "noise_epsilon"
                        ]
                        action = model.action_encoder(noisy, flow_time)
                        output = base.output_from_embeddings(model, observation, action)
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
    except Exception as error:
        error_text = f"rank {context.rank}: {type(error).__name__}: {error}"
    finally:
        model.train(previous_mode)
        restore_local_rng_state(rng, context.device)
    errors = [error_text]
    if context.world_size > 1:
        errors = [None] * context.world_size
        dist.all_gather_object(errors, error_text)
    if any(errors):
        raise RuntimeError(
            "Distributed validation failed: "
            + "; ".join(error for error in errors if error)
        )
    keys = METRIC_KEYS + (
        (
            "unweighted_loss",
            "negative_weight_mean",
            "ignored_negative_fraction",
            "downweighted_negative_fraction",
            "effective_negatives_per_observation",
        )
        if getattr(model, "uses_recovery_normalization", False)
        else ()
    )
    levels_and_counts = [reduce_meter(meter, keys, context) for meter in meters]
    levels = [item[0] for item in levels_and_counts]
    weights = augmenter.probabilities.tolist()
    result = {
        f"val/{key}": sum(
            weight * level[key] for weight, level in zip(weights, levels, strict=True)
        )
        for key in keys
    }
    result.update(
        {
            f"val_t{level:02d}/{key}": value
            for level, metrics in enumerate(levels)
            for key, value in metrics.items()
        }
    )
    baseline_values, _ = reduce_meter(
        baseline,
        (
            ("chance_accuracy", "uninformed_loss", "weighted_uninformed_loss")
            if getattr(model, "uses_recovery_normalization", False)
            else ("chance_accuracy", "uninformed_loss")
        ),
        context,
    )
    result.update({f"val/{key}": value for key, value in baseline_values.items()})
    result["val/grid_mean_loss"] = sum(level["loss"] for level in levels) / len(levels)
    late = [
        level["loss"]
        for flow_time, level in zip(times, levels, strict=True)
        if 0 < flow_time <= 0.4
    ]
    if late:
        result["val/late_loss"] = sum(late) / len(late)
    result["val/samples"] = levels_and_counts[0][1]
    result["val/seconds"] = time.monotonic() - start
    return result


def train_model(args, model, train, val, val_settings, context: DistContext):
    contract = base.make_contract(args, train)
    contract.update(
        world_size=context.world_size,
        global_batch_size=args.batch_size,
        local_batch_size=args.batch_size // context.world_size,
        trainer="global_contrastive_ddp_v1",
    )
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
        lambda step: base.lr_factor(
            step,
            total_steps=total_steps,
            warmup_steps=warmup,
            min_ratio=args.min_lr_ratio,
        ),
    )
    progress = {"epoch": 0, "batch_in_epoch": 0, "global_step": 0, "best_val": None}
    run_info, rng_to_restore = None, None
    if args.resume is not None:
        restored = load_checkpoint(
            args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_contract=contract,
        )
        progress, run_info = restored["progress"], restored["run"]
        rng_to_restore = progress.pop("distributed_rng_states")
        if (
            len(rng_to_restore) != context.world_size
            or not 0 <= progress["epoch"] <= args.epochs
            or not 0 <= progress["batch_in_epoch"] <= len(train)
            or progress["global_step"]
            != progress["epoch"] * len(train) + progress["batch_in_epoch"]
        ):
            raise ValueError(
                "Checkpoint progress/world size differs from the distributed sampler"
            )

    def prepare_output():
        if (
            args.resume is not None
            and args.resume.parent.resolve() != args.output_dir.resolve()
        ):
            raise ValueError("Resume into the original checkpoint directory")
        if (
            args.resume is None
            and args.output_dir.exists()
            and any(args.output_dir.iterdir())
        ):
            raise FileExistsError(
                "Fresh training requires a new or empty output directory"
            )
        if progress["epoch"] == args.epochs and progress["batch_in_epoch"] == 0:
            raise ValueError(
                "Training already completed all configured epochs and validation"
            )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.addHandler(logging.FileHandler(args.output_dir / "training.log"))
        if args.resume is None:
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

    primary_action(context, prepare_output)
    run_holder = []

    def start_wandb():
        run, info = base.init_wandb(
            args, args.output_dir, contract, model.parameter_counts(), run_info
        )
        run_holder.append(run)
        return info

    run_info = primary_action(context, start_wandb)
    run = run_holder[0] if context.is_primary else None
    recorder = base.Recorder(args.output_dir, run) if context.is_primary else None
    wrapped = (
        DistributedDataParallel(
            model,
            device_ids=[context.local_rank] if context.device.type == "cuda" else None,
            broadcast_buffers=False,
        )
        if context.world_size > 1
        else model
    )
    interval = base.WeightedMetrics.from_state(
        progress.get("interval_metrics", {"total": {}, "count": 0})
    )
    epoch_meter = base.WeightedMetrics.from_state(
        progress.get("epoch_metrics", {"total": {}, "count": 0})
    )
    start_epoch, skip = progress["epoch"], progress["batch_in_epoch"]

    def save(path):
        states = capture_distributed_rng(context.device)
        progress.update(
            interval_metrics=interval.state(),
            epoch_metrics=epoch_meter.state(),
            distributed_rng_states=states,
        )
        primary_action(
            context,
            lambda: save_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                progress=progress,
                contract=contract,
                run=run_info,
                rng_state=states[0],
            ),
        )

    def record(step, values):
        primary_action(context, lambda: recorder.log(step, values))

    success = False
    try:
        for epoch in range(start_epoch, args.epochs):
            model.train()
            train.set_epoch(epoch)
            epoch_start = time.monotonic()
            if rng_to_restore is not None and skip == 0:
                restore_distributed_rng(rng_to_restore, context.device)
                rng_to_restore = None
            for batch_index, clean in enumerate(train):
                if epoch == start_epoch and batch_index < skip:
                    continue
                if rng_to_restore is not None:
                    restore_distributed_rng(rng_to_restore, context.device)
                    rng_to_restore = None
                batch = base.move_batch(clean, context.device)
                optimizer.zero_grad(set_to_none=True)
                used_lr = optimizer.param_groups[0]["lr"]
                with base.autocast(context.device, args.precision):
                    output = gather_model_output(wrapped(batch))
                    metrics = cpm_contrastive_loss(output, output)
                metrics["loss"].backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    parameters, args.grad_clip, error_if_nonfinite=True
                )
                optimizer.step()
                scheduler.step()
                values = {
                    key: float(value.detach().item()) for key, value in metrics.items()
                }
                values.update(lr=used_lr, grad_norm=float(norm))
                interval.add(values, args.batch_size)
                epoch_meter.add(values, args.batch_size)
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
                    record(step, logged)
                    if context.is_primary:
                        LOGGER.info(
                            "epoch %d/%d step %d/%d global loss %.5f lr %.2g",
                            epoch + 1,
                            args.epochs,
                            step,
                            total_steps,
                            logged["train/loss"],
                            used_lr,
                        )
                    interval = base.WeightedMetrics()
                if args.save_interval > 0 and step % args.save_interval == 0:
                    save(args.output_dir / f"step_{step:06d}.pt")
                    save(args.output_dir / "last.pt")
                del batch, output, metrics
            if rng_to_restore is not None:
                restore_distributed_rng(rng_to_restore, context.device)
                rng_to_restore = None
            validation = validate(
                model, val, train.augmenter, val_settings, context, args.precision
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
                interval = base.WeightedMetrics()
            record(progress["global_step"], stats)
            checkpoint_due = (epoch + 1) % getattr(
                args, "save_every_epochs", 1
            ) == 0 or epoch + 1 == args.epochs
            improved = checkpoint_due and (
                progress["best_val"] is None
                or validation["val/loss"] < progress["best_val"]
            )
            if improved:
                progress.update(
                    best_val=validation["val/loss"], best_step=progress["global_step"]
                )
            progress.update(epoch=epoch + 1, batch_in_epoch=0)
            epoch_meter = base.WeightedMetrics()
            if checkpoint_due:
                if improved:
                    save(args.output_dir / "best.pt")
                save(args.output_dir / "last.pt")
            if context.is_primary:
                best_text = (
                    f"{progress['best_val']:.5f}"
                    if progress["best_val"] is not None
                    else "pending"
                )
                LOGGER.info(
                    "epoch %d validation %.5f best %s",
                    epoch + 1,
                    validation["val/loss"],
                    best_text,
                )
                if (
                    checkpoint_due
                    and run is not None
                    and progress["best_val"] is not None
                ):
                    try:
                        run.summary.update(
                            {
                                "best_val_loss": progress["best_val"],
                                "best_step": progress.get("best_step"),
                                "best_checkpoint": str(args.output_dir / "best.pt"),
                            }
                        )
                    except Exception:
                        LOGGER.exception(
                            "W&B summary update failed after saving checkpoint; continuing training"
                        )
        success = True
    finally:
        if run is not None:
            run.finish(exit_code=0 if success else 1)


def one_batch_loader(loader: CPMBatchLoader) -> CPMBatchLoader:
    """Exhaust a finite diagnostic loader before shutting down its workers."""
    preview = CPMBatchLoader(
        loader.dataset,
        list(islice(loader.index_batches, 1)),
        include_auxiliary=getattr(loader, "include_auxiliary", False),
        num_workers=loader.num_workers,
        pin_memory=loader.pin_memory,
    )
    if preview.num_workers:
        preview.multiprocessing_context = "spawn"
    return preview


def check_gpu(args, model, train, val, val_settings, context):
    torch.cuda.reset_peak_memory_stats(context.device)
    model.train()
    wrapped = (
        DistributedDataParallel(
            model, device_ids=[context.local_rank], broadcast_buffers=False
        )
        if context.world_size > 1
        else model
    )
    (clean_batch,) = list(one_batch_loader(train.clean_loader))
    generator = torch.Generator(device="cpu").manual_seed(args.seed + context.rank)
    batch = base.move_batch(
        train.augmenter.augment(clean_batch, generator=generator), context.device
    )
    with base.autocast(context.device, args.precision):
        output = gather_model_output(wrapped(batch))
        metrics = cpm_contrastive_loss(output, output)
    metrics["loss"].backward()
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if any(
        parameter.grad is None or not torch.isfinite(parameter.grad).all()
        for parameter in params
    ):
        raise RuntimeError("A trainable parameter has a missing/nonfinite gradient")
    if any(parameter.grad is not None for parameter in model.vision.parameters()):
        raise RuntimeError("Frozen DINO received gradients")
    norm = torch.nn.utils.clip_grad_norm_(
        params, args.grad_clip, error_if_nonfinite=True
    )
    train_values = {key: float(value.detach().item()) for key, value in metrics.items()}
    model.zero_grad(set_to_none=True)
    del batch, output, metrics
    validation = validate(
        model,
        one_batch_loader(val),
        train.augmenter,
        val_settings,
        context,
        args.precision,
    )
    peaks = torch.tensor(
        [
            torch.cuda.max_memory_allocated(context.device),
            torch.cuda.max_memory_reserved(context.device),
        ],
        dtype=torch.float64,
        device=context.device,
    )
    if context.world_size > 1:
        dist.all_reduce(peaks, op=dist.ReduceOp.MAX)
    report = {
        "status": "passed",
        "schema": "touchguide_cpm_ddp_gpu_check_v1",
        "world_size": context.world_size,
        "global_batch_size": args.batch_size,
        "local_batch_size": args.batch_size // context.world_size,
        "parameters": model.parameter_counts(),
        "train_metrics": train_values,
        "validation": validation,
        "gradient_norm": float(norm),
        "max_rank_peak_allocated_gib": float(peaks[0]) / 2**30,
        "max_rank_peak_reserved_gib": float(peaks[1]) / 2**30,
        "optimizer_step": False,
        "training_started": False,
        "wandb_run_created": False,
        "validation_groups_checked_per_rank": 1,
    }

    def publish():
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))

    primary_action(context, publish)


def main():
    args = base.parse_args()
    context = initialize()
    if (
        args.batch_size % context.world_size
        or args.batch_size // context.world_size < 2
    ):
        raise ValueError(
            "Global batch size must divide by world size and provide at least two samples per rank"
        )
    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This GPU does not support BF16")
    logging.basicConfig(
        level=logging.INFO if context.is_primary else logging.WARNING,
        format="%(asctime)s %(message)s",
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    logging.info("Model variant: %s", args.model_variant)
    model = load_cpm_model(args.root, args.model_variant).to(context.device)
    train, val, val_settings = build_loaders(args, context)
    random.seed(args.seed + context.rank)
    np.random.seed(args.seed + context.rank)
    torch.manual_seed(args.seed + context.rank)
    try:
        if args.check_only:
            check_gpu(args, model, train, val, val_settings, context)
        else:
            train_model(args, model, train, val, val_settings, context)
    finally:
        train.dataset.close()
        val.dataset.close()
        if context.world_size > 1:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
