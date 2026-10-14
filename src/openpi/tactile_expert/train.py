"""Train the poker tactile velocity expert while freezing pi0.5 and both encoders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import safetensors.torch
import torch
from torch.utils.data import DataLoader, Subset

from openpi.models.model import Observation
from openpi.models.pi0_config import Pi0Config
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from .data import PokerTactileDataset
from .model import TactilePI05, TactileVelocityExpert


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pi05-root", type=Path, required=True)
    p.add_argument("--tactile-root", type=Path, required=True)
    p.add_argument("--norm-stats", type=Path, required=True)
    p.add_argument("--base-pytorch", type=Path, required=True)
    p.add_argument("--f6-checkpoint", type=Path, required=True)
    p.add_argument("--map-checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--resume-checkpoint", type=Path)
    p.add_argument("--checkpoint-interval", type=int, default=0,
                   help="Also keep a numbered checkpoint every N global steps; 0 disables this")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--val-stride", type=int, default=16)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--log-interval", type=int, default=20)
    p.add_argument("--eval-interval", type=int, default=250)
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb-mode", choices=("disabled", "online", "offline"), default="online")
    p.add_argument("--wandb-entity", default="xianchichen8-hunan-university-library")
    p.add_argument("--wandb-project", default="poker-draw-tactile-expert")
    p.add_argument("--wandb-run-name", default=None)
    return p.parse_args()


def make_observation(batch: dict[str, torch.Tensor], device: torch.device) -> Observation:
    head = batch["head"].to(device, non_blocking=True)
    wrist = batch["wrist"].to(device, non_blocking=True)
    n = head.shape[0]
    yes = torch.ones(n, device=device, dtype=torch.bool)
    no = torch.zeros(n, device=device, dtype=torch.bool)
    return Observation(
        images={"base_0_rgb": head, "left_wrist_0_rgb": torch.zeros_like(head),
                "right_wrist_0_rgb": wrist},
        image_masks={"base_0_rgb": yes, "left_wrist_0_rgb": no,
                     "right_wrist_0_rgb": yes},
        state=batch["state"].to(device, non_blocking=True),
        tokenized_prompt=batch["prompt_ids"].to(device, non_blocking=True),
        tokenized_prompt_mask=batch["prompt_mask"].to(device, non_blocking=True),
    )


def forward_batch(policy: TactilePI05, batch: dict[str, torch.Tensor], device: torch.device):
    observation = make_observation(batch, device)
    return policy.late_flow_loss(
        observation,
        batch["actions"].to(device, non_blocking=True),
        batch["f6_history"].to(device, non_blocking=True),
        batch["f6_current"].to(device, non_blocking=True),
        batch["tactile_map"].to(device, non_blocking=True),
    )


def save_checkpoint(path: Path, policy: TactilePI05, optimizer: torch.optim.Optimizer,
                    args: argparse.Namespace, step: int, best_val: float) -> None:
    payload = {
        "format": "poker_pi05_tactile_velocity_v1",
        "step": step,
        "best_val_mse": best_val,
        "tactile_state": policy.tactile.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "paths": {key: str(getattr(args, key)) for key in
                  ("base_pytorch", "f6_checkpoint", "map_checkpoint", "norm_stats", "pi05_root", "tactile_root")},
        "settings": vars(args),
        "data_normalization": "pi05_quantile_q01_q99",
        "rng_state": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all(),
        },
    }
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp)
    temp.replace(path)


def restore_checkpoint(path: Path, policy: TactilePI05,
                       optimizer: torch.optim.Optimizer, args: argparse.Namespace) -> tuple[int, float]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "poker_pi05_tactile_velocity_v1":
        raise ValueError(f"Unsupported tactile expert checkpoint: {path}")
    if checkpoint.get("data_normalization") != "pi05_quantile_q01_q99":
        raise ValueError("Resume checkpoint does not use the poker pi0.5 quantile normalization")
    for key in ("base_pytorch", "f6_checkpoint", "map_checkpoint", "norm_stats", "pi05_root", "tactile_root"):
        saved = checkpoint["paths"][key]
        current = getattr(args, key)
        if Path(saved).resolve() != current.resolve():
            raise ValueError(f"Resume checkpoint {key} differs: {saved} != {current}")
    for key in ("batch_size", "stride", "val_stride", "val_ratio", "seed", "lr", "weight_decay"):
        saved = checkpoint["settings"][key]
        current = getattr(args, key)
        if saved != current:
            raise ValueError(f"Resume checkpoint {key} differs: {saved} != {current}")
    step = int(checkpoint["step"])
    if step < 0 or step >= args.steps:
        raise ValueError(f"Resume step {step} must be less than target steps {args.steps}")
    policy.tactile.load_state_dict(checkpoint["tactile_state"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    rng_state = checkpoint.get("rng_state")
    if rng_state is not None:
        torch.set_rng_state(rng_state["torch"])
        torch.cuda.set_rng_state_all(rng_state["cuda"])
    else:
        # Older checkpoints did not save RNG state. Avoid repeating the first
        # run's shuffled batches and noise by starting a different sequence.
        torch.manual_seed(args.seed + step)
        torch.cuda.manual_seed_all(args.seed + step)
    return step, float(checkpoint["best_val_mse"])


@torch.no_grad()
def evaluate(policy: TactilePI05, loader: DataLoader, device: torch.device, max_batches: int) -> dict[str, float]:
    policy.eval()
    total = {"mse": 0.0, "base_mse": 0.0, "correction_mse": 0.0}
    count = 0
    # Fixed flow noise/time makes validation comparable between checkpoints.
    with torch.random.fork_rng(devices=[torch.cuda.current_device()] if device.type == "cuda" else []):
        torch.manual_seed(12345)
        for batch in loader:
            _, metrics = forward_batch(policy, batch, device)
            for key in total:
                total[key] += float(metrics[key])
            count += 1
            if count >= max_batches:
                break
    return {key: value / count for key, value in total.items()}


def main() -> None:
    args = parse_args()
    if min(args.steps, args.batch_size, args.stride, args.val_stride,
           args.log_interval, args.eval_interval, args.eval_batches) <= 0:
        raise ValueError("All counts and intervals must be positive")
    if args.checkpoint_interval < 0:
        raise ValueError("checkpoint-interval must be nonnegative")
    if not torch.cuda.is_available():
        raise RuntimeError("This training entry point requires a CUDA GPU")
    device = torch.device("cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    base_config = Pi0Config(action_dim=32, action_horizon=30, max_token_len=200,
                            pi05=True, dtype="bfloat16", pytorch_compile_mode=None)
    with (args.base_pytorch / "config.json").open() as f:
        converted_config = json.load(f)
    if converted_config["action_dim"] != 32 or converted_config["action_horizon"] != 30:
        raise ValueError("Converted pi0.5 checkpoint must have action_dim=32 and action_horizon=30")
    print("Loading frozen pi0.5 weights...", flush=True)
    base = PI0Pytorch(base_config)
    safetensors.torch.load_model(base, args.base_pytorch / "model.safetensors", strict=True)
    base = base.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)

    f6_blob = torch.load(args.f6_checkpoint, map_location="cpu", weights_only=False)
    map_blob = torch.load(args.map_checkpoint, map_location="cpu", weights_only=False)
    if f6_blob["config"]["window"] != 16 or f6_blob["config"]["granularity"] != "hand":
        raise ValueError("Expected the pretrained 16-frame per-hand F6 VQ-VAE")
    tactile = TactileVelocityExpert(
        f6_blob["config"], f6_blob["config"]["codebook_size"],
        width=1024, layers=2, action_dim=32,
    )
    tactile.load_encoder_weights(str(args.f6_checkpoint), str(args.map_checkpoint))
    policy = TactilePI05(base, tactile.to(device)).train()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in policy.tactile.parameters() if parameter.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )

    print("Pairing tactile and LeRobot episodes...", flush=True)
    train_ds = PokerTactileDataset(args.pi05_root, args.tactile_root, args.norm_stats,
                                   split="train", seed=args.seed,
                                   val_ratio=args.val_ratio, stride=args.stride)
    val_ds = PokerTactileDataset(args.pi05_root, args.tactile_root, args.norm_stats,
                                 split="val", seed=args.seed,
                                 val_ratio=args.val_ratio, stride=args.val_stride)
    if not np.allclose(train_ds.f6_stats.q01, np.asarray(f6_blob["stats"]["q01"])) or not np.allclose(
            train_ds.f6_stats.q99, np.asarray(f6_blob["stats"]["q99"])):
        raise ValueError("F6 normalization differs from the pretrained checkpoint")
    if not np.allclose(train_ds.map_stats.q01, np.asarray(map_blob["stats"]["q01"])) or not np.allclose(
            train_ds.map_stats.q99, np.asarray(map_blob["stats"]["q99"])):
        raise ValueError("Map normalization differs from the pretrained checkpoint")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              drop_last=True, num_workers=args.num_workers, pin_memory=True)
    # Cover the whole episode-level holdout instead of validating only on
    # the earliest episodes when eval_batches limits validation cost.
    val_count = min(len(val_ds), args.batch_size * args.eval_batches)
    val_indices = np.linspace(0, len(val_ds) - 1, val_count, dtype=np.int64).tolist()
    val_loader = DataLoader(Subset(val_ds, val_indices), batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    print(f"Train frames: {len(train_ds)}, validation frames: {len(val_ds)} "
          f"({val_count} spread across the holdout per evaluation); "
          "normalization: pi0.5 q01/q99", flush=True)

    run = None
    if args.wandb_mode != "disabled":
        import wandb
        run = wandb.init(entity=args.wandb_entity, project=args.wandb_project,
                         name=args.wandb_run_name, dir=str(args.output_dir),
                         mode=args.wandb_mode,
                         config={key: str(value) if isinstance(value, Path) else value
                                 for key, value in vars(args).items()})
    best_val = float("inf")
    step = 0
    if args.resume_checkpoint is not None:
        step, best_val = restore_checkpoint(args.resume_checkpoint, policy, optimizer, args)
        print(f"Resumed tactile expert from step={step}; target={args.steps}; best_val_mse={best_val:.6f}", flush=True)
    try:
        while step < args.steps:
            for batch in train_loader:
                policy.train()
                optimizer.zero_grad(set_to_none=True)
                loss, metrics = forward_batch(policy, batch, device)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.tactile.parameters(), args.grad_clip)
                optimizer.step()
                step += 1
                if step == 1 or step % args.log_interval == 0:
                    report = {f"train/{key}": float(value) for key, value in metrics.items()}
                    report["train/loss"] = float(loss.detach())
                    print(f"step={step} mse={report['train/mse']:.6f} "
                          f"base_mse={report['train/base_mse']:.6f}", flush=True)
                    if run is not None:
                        run.log(report, step=step)
                if step % args.eval_interval == 0 or step == args.steps:
                    val = evaluate(policy, val_loader, device, args.eval_batches)
                    print(f"step={step} val_mse={val['mse']:.6f} "
                          f"val_base_mse={val['base_mse']:.6f}", flush=True)
                    if run is not None:
                        run.log({f"val/{key}": value for key, value in val.items()}, step=step)
                    if val["mse"] < best_val:
                        best_val = val["mse"]
                        save_checkpoint(args.output_dir / "best.pt", policy, optimizer, args, step, best_val)
                    save_checkpoint(args.output_dir / "latest.pt", policy, optimizer, args, step, best_val)
                    if run is not None:
                        run.summary["best_val_mse"] = best_val
                        run.summary["best_checkpoint"] = str(args.output_dir / "best.pt")
                if args.checkpoint_interval and step % args.checkpoint_interval == 0:
                    save_checkpoint(args.output_dir / f"step_{step:06d}.pt",
                                    policy, optimizer, args, step, best_val)
                if step >= args.steps:
                    break
    finally:
        if run is not None:
            run.finish()


if __name__ == "__main__":
    main()
