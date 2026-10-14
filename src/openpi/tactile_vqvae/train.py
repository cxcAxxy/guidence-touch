"""Train the T-Rex-style F6 VQ-VAE on a pi05_trex tactile sidecar.

This is a standalone tactile representation run.  It leaves the pi0.5 policy
and action expert untouched.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import build_train_val_datasets
from .models import TactileVQVAE
from .models import TactileVQVAEConfig


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--hand-mode", choices=("left", "right", "both"), default="right")
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--bottleneck-channels", type=int, default=256)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--n-strided-blocks", type=int, default=2)
    parser.add_argument("--codebook-size", type=int, default=128)
    parser.add_argument("--commitment-weight", type=float, default=0.25)
    parser.add_argument("--decay", type=float, default=0.99)
    parser.add_argument("--revive-freq", type=int, default=200)
    parser.add_argument("--revive-threshold", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=0, help="0 means no step limit")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    parser.add_argument("--wandb-mode", choices=("disabled", "online", "offline"), default="disabled")
    parser.add_argument("--wandb-entity", default=None, help="W&B username or team entity")
    parser.add_argument("--wandb-project", default="poker-draw-f6-vqvae")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-log-interval", type=int, default=20)
    return parser.parse_args()


def _set_seed(seed: int, *, use_cuda: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)


def _make_loader(dataset, args: argparse.Namespace, *, shuffle: bool, pin_memory: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
        drop_last=shuffle,
    )


def _evaluate(model: TactileVQVAE, loader: DataLoader | None, device: torch.device) -> float | None:
    if loader is None:
        return None
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch in loader:
            result = model(batch["f6"].to(device, non_blocking=True), batch["magnitude"].to(device))
            batch_size = batch["f6"].shape[0]
            total_loss += float(result["total_loss"].item()) * batch_size
            total_samples += batch_size
    return total_loss / total_samples if total_samples else None


def _save_checkpoint(
    path: Path,
    model: TactileVQVAE,
    optimizer: torch.optim.Optimizer,
    cfg: TactileVQVAEConfig,
    stats: dict,
    epoch: int,
    step: int,
    train_loss: float,
    val_loss: float | None,
    wandb_run_id: str | None,
) -> None:
    payload = {
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "config": cfg.to_dict(),
        "stats": stats,
        "epoch": epoch,
        "step": step,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "wandb_run_id": wandb_run_id,
        "format": "openpi_trex_f6_vqvae_v1",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


def _train_epochs(
    args: argparse.Namespace,
    model: TactileVQVAE,
    optimizer: torch.optim.Optimizer,
    cfg: TactileVQVAEConfig,
    stats: dict,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    device: torch.device,
    wandb_run: Any | None,
) -> None:
    step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss_sum = 0.0
        epoch_samples = 0
        for batch in train_loader:
            f6 = batch["f6"].to(device, non_blocking=True)
            magnitude = batch["magnitude"].to(device, non_blocking=True)
            result = model(f6, magnitude)
            optimizer.zero_grad(set_to_none=True)
            result["total_loss"].backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            step += 1
            loss = float(result["total_loss"].item())
            epoch_loss_sum += loss * f6.shape[0]
            epoch_samples += f6.shape[0]
            if step == 1 or step % 50 == 0:
                print(
                    f"epoch={epoch} step={step} loss={loss:.5f} "
                    f"recon={result['recon_loss'].item():.5f} vq={result['vq_loss'].item():.5f} "
                    f"active={int(result['active_codes'].item())} perplexity={result['perplexity'].item():.2f}",
                    flush=True,
                )
            if wandb_run is not None and (step == 1 or step % args.wandb_log_interval == 0):
                wandb_run.log(
                    {
                        "train/loss": loss,
                        "train/recon_loss": float(result["recon_loss"].item()),
                        "train/vq_loss": float(result["vq_loss"].item()),
                        "codebook/active_in_batch": int(result["active_codes"].item()),
                        "codebook/perplexity_in_batch": float(result["perplexity"].item()),
                        "codebook/revived": int(result["revived"].item()),
                    },
                    step=step,
                    commit=False,
                )
            if args.max_steps > 0 and step >= args.max_steps:
                break

        train_loss = epoch_loss_sum / epoch_samples if epoch_samples else float("nan")
        val_loss = _evaluate(model, val_loader, device)
        print(f"epoch={epoch} train_loss={train_loss:.5f} val_loss={val_loss}", flush=True)
        if wandb_run is not None:
            epoch_metrics = {"train/epoch_loss": train_loss, "trainer/epoch": epoch}
            if val_loss is not None:
                epoch_metrics["val/loss"] = val_loss
            wandb_run.log(epoch_metrics, step=step, commit=True)

        for path in (args.output_dir / "latest.pt", args.output_dir / f"epoch_{epoch:04d}.pt"):
            _save_checkpoint(
                path,
                model,
                optimizer,
                cfg,
                stats,
                epoch,
                step,
                train_loss,
                val_loss,
                wandb_run.id if wandb_run is not None else None,
            )
        if wandb_run is not None:
            wandb_run.summary["latest_checkpoint"] = str(args.output_dir / "latest.pt")
        if args.max_steps > 0 and step >= args.max_steps:
            break


def main() -> None:
    args = _parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.wandb_log_interval <= 0:
        raise ValueError("epochs, batch-size, and wandb-log-interval must be positive")
    if args.wandb_mode != "disabled" and not args.wandb_entity:
        raise ValueError("--wandb-entity is required when W&B logging is enabled")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable to PyTorch in this environment; check GPU access before training")
    _set_seed(args.seed, use_cuda=device.type == "cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds = build_train_val_datasets(
        args.data_root,
        window=args.window,
        stride=args.stride,
        val_ratio=args.val_ratio,
        seed=args.seed,
        hand_mode=args.hand_mode,
    )
    train_loader = _make_loader(train_ds, args, shuffle=True, pin_memory=device.type == "cuda")
    val_loader = (
        _make_loader(val_ds, args, shuffle=False, pin_memory=device.type == "cuda") if val_ds is not None else None
    )
    cfg = TactileVQVAEConfig(
        window=args.window,
        in_channels=30,
        hidden_channels=args.hidden_channels,
        bottleneck_channels=args.bottleneck_channels,
        embed_dim=args.embed_dim,
        n_strided_blocks=args.n_strided_blocks,
        codebook_size=args.codebook_size,
        commitment_weight=args.commitment_weight,
        decay=args.decay,
        revive_freq=args.revive_freq,
        revive_threshold=args.revive_threshold,
        granularity="hand",
    )
    model = TactileVQVAE(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    stats = train_ds.stats.to_dict()

    run_args = vars(args) | {"data_root": str(args.data_root), "output_dir": str(args.output_dir)}
    run_config = {
        "args": run_args,
        "model": cfg.to_dict(),
        "train_episodes": len(train_ds.episodes),
        "val_episodes": len(val_ds.episodes) if val_ds is not None else 0,
        "train_windows": len(train_ds),
        "val_windows": len(val_ds) if val_ds is not None else 0,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    with (args.output_dir / "run_config.json").open("w") as f:
        json.dump(run_config, f, indent=2)
    print(
        f"device={device} train_episodes={run_config['train_episodes']} "
        f"val_episodes={run_config['val_episodes']} "
        f"train_windows={run_config['train_windows']} val_windows={run_config['val_windows']}",
        flush=True,
    )
    if args.wandb_mode == "disabled":
        _train_epochs(args, model, optimizer, cfg, stats, train_loader, val_loader, device, None)
    else:
        import wandb  # noqa: PLC0415 - optional experiment logging

        with wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=args.wandb_run_name or args.output_dir.name,
            dir=str(args.output_dir),
            config=run_config,
            mode=args.wandb_mode,
            tags=["poker-draw", "f6-vqvae", "pi05-trex"],
            job_type="vqvae-pretrain",
        ) as wandb_run:
            print(f"wandb_run={wandb_run.url}", flush=True)
            _train_epochs(args, model, optimizer, cfg, stats, train_loader, val_loader, device, wandb_run)


if __name__ == "__main__":
    main()
