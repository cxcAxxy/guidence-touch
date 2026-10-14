"""Pretrain the poker three-axis tactile-map encoder by denoising reconstruction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data.tactile_map import build_map_train_val_datasets
from .models.tactile_map_autoencoder import TactileMapAutoencoder


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hand-mode", choices=("left", "right", "both"), default="right")
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--token-dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=128, help="Frames per batch; right hand gives 5 maps/frame")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--mask-prob", type=float, default=0.1, help="Taxel masking during training only")
    parser.add_argument("--contact-weight", type=float, default=3.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--wandb-mode", choices=("disabled", "online", "offline"), default="disabled")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-project", default="poker-draw-tactile-map-ae")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-log-interval", type=int, default=20)
    return parser.parse_args()


def _save_checkpoint(
    path: Path,
    model: TactileMapAutoencoder,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    stats: dict,
    epoch: int,
    step: int,
    train_metrics: dict,
    val_metrics: dict | None,
    best_val_loss: float,
    wandb_run_id: str | None,
) -> None:
    payload = {
        "model_state": model.state_dict(),
        "encoder_state": model.encoder.state_dict(),
        "optim_state": optimizer.state_dict(),
        "config": {"token_dim": args.token_dim, "hand_mode": args.hand_mode},
        "stats": stats,
        "epoch": epoch,
        "step": step,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "best_val_loss": best_val_loss,
        "wandb_run_id": wandb_run_id,
        "format": "openpi_trex_tactile_map_ae_v1",
    }
    temp = path.with_name(path.name + ".tmp")
    torch.save(payload, temp)
    temp.replace(path)


def _run_epoch(
    model: TactileMapAutoencoder,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    *,
    device: torch.device,
    empty_map: torch.Tensor,
    mask_prob: float,
    contact_weight: float,
    grad_clip: float,
    step: int,
    wandb_run: Any | None,
    log_interval: int,
) -> tuple[dict[str, float], int]:
    training = optimizer is not None
    model.train(training)
    weighted_error = 0.0
    total_weight = 0.0
    contact_error = 0.0
    contact_count = 0
    free_error = 0.0
    free_count = 0
    for batch in loader:
        target = batch["maps"].to(device, non_blocking=True)
        contact = batch["contact"].to(device, non_blocking=True)
        input_maps = target
        if training and mask_prob > 0:
            masked = torch.rand((*target.shape[:2], 1, 7, 5), device=device) < mask_prob
            input_maps = torch.where(masked, empty_map, target)

        with torch.set_grad_enabled(training):
            prediction = model(input_maps)
            per_finger = (prediction - target).square().mean(dim=(2, 3, 4))
            weights = 1.0 + (contact_weight - 1.0) * contact.to(per_finger.dtype)
            loss = (per_finger * weights).sum() / weights.sum()
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        weighted_error += float((per_finger.detach() * weights).sum())
        total_weight += float(weights.sum())
        contact_error += float(per_finger.detach()[contact].sum())
        contact_count += int(contact.sum())
        free_error += float(per_finger.detach()[~contact].sum())
        free_count += int((~contact).sum())
        if training:
            step += 1
            if step == 1 or step % 50 == 0:
                print(
                    f"step={step} loss={loss.item():.6f} "
                    f"contact_mse={per_finger[contact].mean().item() if contact.any() else float('nan'):.6f}",
                    flush=True,
                )
            if wandb_run is not None and (step == 1 or step % log_interval == 0):
                wandb_run.log({"train/loss": float(loss.item())}, step=step, commit=False)
    return {
        "loss": weighted_error / total_weight,
        "contact_mse": contact_error / contact_count if contact_count else float("nan"),
        "free_mse": free_error / free_count if free_count else float("nan"),
        "contact_fraction": contact_count / (contact_count + free_count),
    }, step


def main() -> None:
    args = _parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.stride <= 0 or args.wandb_log_interval <= 0:
        raise ValueError("epochs, batch-size, stride and wandb-log-interval must be positive")
    if not 0 <= args.mask_prob < 1:
        raise ValueError("mask-prob must be in [0, 1)")
    if args.contact_weight < 1:
        raise ValueError("contact-weight must be at least 1")
    if args.wandb_mode != "disabled" and not args.wandb_entity:
        raise ValueError("--wandb-entity is required for W&B logging")
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable to PyTorch")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    train_ds, val_ds = build_map_train_val_datasets(
        args.data_root,
        stride=args.stride,
        val_ratio=args.val_ratio,
        seed=args.seed,
        hand_mode=args.hand_mode,
    )

    def loader(ds, *, shuffle: bool):
        return DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0,
            drop_last=shuffle,
        )

    train_loader = loader(train_ds, shuffle=True)
    val_loader = loader(val_ds, shuffle=False) if val_ds is not None else None
    model = TactileMapAutoencoder(token_dim=args.token_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    empty_map = torch.from_numpy(train_ds.stats.empty_map()).to(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_args = vars(args) | {"data_root": str(args.data_root), "output_dir": str(args.output_dir)}
    config = {
        "args": run_args,
        "train_episodes": len(train_ds.episodes),
        "val_episodes": len(val_ds.episodes) if val_ds is not None else 0,
        "train_frames": len(train_ds),
        "val_frames": len(val_ds) if val_ds is not None else 0,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    with (args.output_dir / "run_config.json").open("w") as f:
        json.dump(config, f, indent=2)
    print(f"device={device} train_frames={len(train_ds)} val_frames={len(val_ds) if val_ds else 0}", flush=True)

    def train(wandb_run: Any | None) -> None:
        step = 0
        best_val_loss = float("inf")
        for epoch in range(1, args.epochs + 1):
            train_metrics, step = _run_epoch(
                model,
                train_loader,
                optimizer,
                device=device,
                empty_map=empty_map,
                mask_prob=args.mask_prob,
                contact_weight=args.contact_weight,
                grad_clip=args.grad_clip,
                step=step,
                wandb_run=wandb_run,
                log_interval=args.wandb_log_interval,
            )
            if val_loader is not None:
                with torch.no_grad():
                    val_metrics, _ = _run_epoch(
                        model,
                        val_loader,
                        None,
                        device=device,
                        empty_map=empty_map,
                        mask_prob=0,
                        contact_weight=args.contact_weight,
                        grad_clip=args.grad_clip,
                        step=step,
                        wandb_run=None,
                        log_interval=args.wandb_log_interval,
                    )
            else:
                val_metrics = None
            print(
                f"epoch={epoch} train_loss={train_metrics['loss']:.6f} "
                f"val_loss={val_metrics['loss'] if val_metrics else None} "
                f"val_contact_mse={val_metrics['contact_mse'] if val_metrics else None}",
                flush=True,
            )
            if wandb_run is not None:
                metrics = {f"train/epoch_{k}": v for k, v in train_metrics.items()}
                if val_metrics is not None:
                    metrics.update({f"val/{k}": v for k, v in val_metrics.items()})
                metrics["trainer/epoch"] = epoch
                wandb_run.log(metrics, step=step, commit=True)
            improved = val_metrics is not None and val_metrics["loss"] < best_val_loss
            if improved:
                best_val_loss = val_metrics["loss"]
            for name in ("latest.pt", f"epoch_{epoch:04d}.pt"):
                _save_checkpoint(
                    args.output_dir / name,
                    model,
                    optimizer,
                    args,
                    train_ds.stats.to_dict(),
                    epoch,
                    step,
                    train_metrics,
                    val_metrics,
                    best_val_loss,
                    wandb_run.id if wandb_run else None,
                )
            if improved:
                _save_checkpoint(
                    args.output_dir / "best.pt",
                    model,
                    optimizer,
                    args,
                    train_ds.stats.to_dict(),
                    epoch,
                    step,
                    train_metrics,
                    val_metrics,
                    best_val_loss,
                    wandb_run.id if wandb_run else None,
                )
            if wandb_run is not None:
                wandb_run.summary["best_val_loss"] = best_val_loss
                wandb_run.summary["best_checkpoint"] = str(args.output_dir / "best.pt")

    if args.wandb_mode == "disabled":
        train(None)
    else:
        import wandb  # noqa: PLC0415

        with wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=args.wandb_run_name or args.output_dir.name,
            dir=str(args.output_dir),
            config=config,
            mode=args.wandb_mode,
            tags=["poker-draw", "tactile-map", "denoising-autoencoder"],
            job_type="map-ae-pretrain",
        ) as run:
            print(f"wandb_run={run.url}", flush=True)
            train(run)


if __name__ == "__main__":
    main()
