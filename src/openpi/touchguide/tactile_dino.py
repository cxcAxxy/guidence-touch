"""Local force-to-DINO image adaptation; this is not a published TouchGuide formula.

Current taxels retain normal/tangent-0/tangent-1 channel order. A fixed, signed
asinh map uses train RMS without clipping or per-frame normalization, followed by
aspect-preserving resize, neutral padding, and standard ImageNet normalization.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional


class TactileDinoPreprocessor(nn.Module):
    """Convert [B,5,3,7,5] current forces to [B*5,3,224,224] FP32 pixels.

    ``encode_image`` returns the mapped/padded image before ImageNet normalization.
    It is intentionally allowed outside [0,1] for forces above the reference range.
    ``forward`` returns ImageNet-normalized pixels for a separate DINO entry point
    that accepts tensors directly, without the camera RGB [-1,1] range guard.
    Output order is batch-major, then finger-major. No parameters are trainable.
    """

    def __init__(self, rms: Any, *, reference_multiple: float = 32.0, image_size: int = 224) -> None:
        super().__init__()
        scale = torch.as_tensor(rms, dtype=torch.float32).detach().clone()
        if scale.shape != (3,) or not torch.isfinite(scale).all() or not (scale > 0).all():
            raise ValueError("Tactile RMS must contain three finite positive values in channel order")
        if isinstance(reference_multiple, bool) or not math.isfinite(reference_multiple) or reference_multiple <= 0:
            raise ValueError("reference_multiple must be finite and positive")
        if not isinstance(image_size, int) or isinstance(image_size, bool) or image_size < 7 or image_size % 7:
            raise ValueError("image_size must be a positive multiple of 7 to preserve the 7:5 taxel geometry")
        self.reference_multiple = float(reference_multiple)
        self.image_size = image_size
        self.resized_width = image_size // 7 * 5
        horizontal_padding = image_size - self.resized_width
        self.pad_left = horizontal_padding // 2
        self.pad_right = horizontal_padding - self.pad_left
        self.mapping_denominator = 2.0 * math.asinh(self.reference_multiple)
        self.register_buffer("tactile_rms", scale.reshape(1, 1, 3, 1, 1))
        self.register_buffer("imagenet_mean", torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1))
        self.register_buffer("imagenet_std", torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1))

    def encode_image(self, taxels: torch.Tensor) -> torch.Tensor:
        """Return signed force images with neutral 0.5 padding, without clipping."""
        if (
            not isinstance(taxels, torch.Tensor)
            or not taxels.is_floating_point()
            or taxels.ndim != 5
            or taxels.shape[0] < 1
            or tuple(taxels.shape[1:]) != (5, 3, 7, 5)
        ):
            raise ValueError("Expected floating current taxels [B>=1,5,3,7,5]")
        if not torch.isfinite(taxels).all():
            raise ValueError("Current taxels must be finite")
        # FP64 only on the tiny original taxel grid avoids division/derivative
        # overflow for extreme finite FP32 forces. Resize and DINO pixels are FP32.
        scale = self.tactile_rms.to(device=taxels.device, dtype=torch.float64)
        mapped = (0.5 + torch.asinh(taxels.to(torch.float64) / scale) / self.mapping_denominator).float()
        if not torch.isfinite(mapped).all():
            raise ValueError("Force-to-image mapping produced nonfinite values")
        images = mapped.reshape(-1, 3, 7, 5)
        images = functional.interpolate(
            images, size=(self.image_size, self.resized_width), mode="bilinear", align_corners=False
        )
        return functional.pad(images, (self.pad_left, self.pad_right, 0, 0), mode="constant", value=0.5)

    def forward(self, taxels: torch.Tensor) -> torch.Tensor:
        pixels = self.encode_image(taxels)
        mean = self.imagenet_mean.to(device=pixels.device, dtype=torch.float32)
        std = self.imagenet_std.to(device=pixels.device, dtype=torch.float32)
        return (pixels - mean) / std
