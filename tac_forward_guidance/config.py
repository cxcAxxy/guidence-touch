"""Explicit v1 sampling settings; no training configuration is changed."""

import argparse
from dataclasses import dataclass
import math

from openpi.touchguide.sampler import GuidanceConfig


@dataclass(frozen=True)
class ForwardGuidanceConfig(GuidanceConfig):
    scale: float = 0.3
    gradient_mode: str = "full"
    max_update_rms: float | None = 0.01
    local_score_probe: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.gradient_mode not in ("full", "stop_velocity"):
            raise ValueError("gradient_mode must be full or the explicit stop_velocity ablation")
        if self.max_update_rms is not None and (not math.isfinite(self.max_update_rms) or self.max_update_rms <= 0):
            raise ValueError("max_update_rms must be positive and finite, or None")
        if not isinstance(self.local_score_probe, bool):
            raise ValueError("local_score_probe must be boolean")


def add_guidance_arguments(parser, *, verification=False):
    parser.add_argument("--guidance-scale", type=float, default=0.3)
    parser.add_argument("--guidance-steps", type=int, default=3)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--gradient-mode", choices=("full", "stop_velocity"), default="full")
    parser.add_argument(
        "--max-update-rms", type=float, default=0.01, help="Per-sample physical latent-step RMS cap; 0 disables"
    )
    parser.add_argument(
        "--local-score-probe",
        action=argparse.BooleanOptionalAction,
        default=verification,
        help="Extra VLA forwards to measure actual local score change; excluded from production defaults",
    )
    parser.add_argument("--precision", choices=("bfloat16", "float32"), default="bfloat16")


def config_from_args(args):
    return ForwardGuidanceConfig(
        num_steps=args.num_steps,
        guidance_steps=args.guidance_steps,
        scale=args.guidance_scale,
        gradient_mode=args.gradient_mode,
        max_update_rms=None if args.max_update_rms == 0 else args.max_update_rms,
        local_score_probe=args.local_score_probe,
    )
