"""Meaningful CPU fixtures for the local tactile-to-DINO mapping; no DINO is loaded."""

from __future__ import annotations

from collections.abc import Callable
import itertools
import json
import math

import torch

from openpi.touchguide.tactile_dino import TactileDinoPreprocessor


def reject(action: Callable[[], object]) -> None:
    try:
        action()
    except ValueError:
        return
    raise AssertionError("Invalid tactile mapping input was accepted")


def main() -> None:
    torch.set_num_threads(1)
    rms = torch.tensor([0.5, 0.25, 0.125])
    preprocessor = TactileDinoPreprocessor(rms)
    scale = rms.reshape(1, 1, 3, 1, 1)
    zero = torch.zeros(1, 5, 3, 7, 5)
    images = preprocessor.encode_image(zero)
    assert images.shape == (5, 3, 224, 224)
    assert images.dtype == torch.float32
    assert torch.equal(images, torch.full_like(images, 0.5))
    expected_background = ((0.5 - preprocessor.imagenet_mean) / preprocessor.imagenet_std).expand_as(images)
    torch.testing.assert_close(preprocessor(zero), expected_background, rtol=0, atol=0)
    assert not list(preprocessor.parameters())
    assert preprocessor.resized_width == 160
    assert preprocessor.pad_left == 32
    assert preprocessor.pad_right == 32

    def uniform(multiple: float) -> torch.Tensor:
        return torch.ones_like(zero) * scale * multiple

    magnitudes = [preprocessor.encode_image(uniform(value))[:, :, :, 32:192].mean() for value in (0, 1, 2, 32, 64)]
    assert all(right > left for left, right in itertools.pairwise(magnitudes))
    torch.testing.assert_close(magnitudes[0], torch.tensor(0.5), rtol=0, atol=0)
    torch.testing.assert_close(magnitudes[3], torch.tensor(1.0), rtol=0, atol=1e-7)
    assert magnitudes[-1] > 1
    torch.testing.assert_close(
        preprocessor.encode_image(uniform(32))[:, :, :, 32:192], torch.ones(5, 3, 224, 160), rtol=0, atol=1e-7
    )
    assert torch.equal(preprocessor.encode_image(uniform(-32))[:, :, :, 32:192], torch.zeros(5, 3, 224, 160))
    positive = preprocessor.encode_image(uniform(3))
    negative = preprocessor.encode_image(uniform(-3))
    torch.testing.assert_close(positive + negative, torch.ones_like(positive), rtol=0, atol=2e-7)
    assert (positive[:, :, :, 32:192] > 0.5).all()
    assert (negative[:, :, :, 32:192] < 0.5).all()
    # Per-channel RMS calibration preserves channel order and signed forces.
    single_channel = zero.clone()
    single_channel[:, :, 1] = rms[1] * -2
    channel_images = preprocessor.encode_image(single_channel)
    assert torch.equal(channel_images[:, 0], torch.full_like(channel_images[:, 0], 0.5))
    assert torch.equal(channel_images[:, 2], torch.full_like(channel_images[:, 2], 0.5))
    assert (channel_images[:, 1, :, 32:192] < 0.5).all()
    for channel in range(3):
        torch.testing.assert_close(positive[:, channel, :, 32:192], positive[:, 0, :, 32:192], rtol=0, atol=0)
    # Fixed neutral padding is identical across fingers, signs, channels and batches.
    assert torch.equal(positive[:, :, :, :32], torch.full_like(positive[:, :, :, :32], 0.5))
    assert torch.equal(negative[:, :, :, 192:], torch.full_like(negative[:, :, :, 192:], 0.5))
    normalized = preprocessor(uniform(3))
    torch.testing.assert_close(normalized[:, :, :, :32], expected_background[:, :, :, :32], rtol=0, atol=0)

    # One finger's representation is independent of the other frames in a batch.
    two_frames = torch.cat([uniform(3), uniform(-7)], dim=0)
    torch.testing.assert_close(preprocessor.encode_image(two_frames)[:5], positive, rtol=0, atol=0)
    torch.testing.assert_close(
        preprocessor.encode_image(two_frames)[5:], preprocessor.encode_image(uniform(-7)), rtol=0, atol=0
    )
    identities = torch.arange(10).reshape(2, 5, 1, 1, 1).float()
    identity_images = preprocessor.encode_image(torch.ones(2, 5, 3, 7, 5) * scale * identities)
    expected_identity = 0.5 + torch.asinh(torch.arange(10).double()) / (2 * math.asinh(32))
    torch.testing.assert_close(identity_images[:, 0, 0, 32], expected_identity.float(), rtol=0, atol=1e-7)
    assert preprocessor.encode_image(uniform(3).bfloat16()).dtype == torch.float32

    # Row and column directions remain distinct; the impulse stays at row2/col3.
    spatial = zero.clone()
    spatial[0, 0, 0] = torch.arange(7).reshape(7, 1).expand(7, 5) * rms[0]
    spatial[0, 0, 1] = -torch.arange(5).reshape(1, 5).expand(7, 5) * rms[1]
    spatial[0, 0, 2, 2, 3] = 4 * rms[2]
    oriented = preprocessor.encode_image(spatial)[0]
    assert oriented[0, -1, 80] > oriented[0, 0, 80]
    torch.testing.assert_close(oriented[0, :, 40], oriented[0, :, 180], rtol=0, atol=2e-7)
    assert oriented[1, 80, 32] > oriented[1, 80, 191]
    torch.testing.assert_close(oriented[1, 20, :], oriented[1, 200, :], rtol=0, atol=2e-7)
    peak = int(oriented[2].argmax())
    assert peak // 224 in (79, 80)
    assert peak % 224 in (143, 144)
    # Constant taxel values demonstrate nonlinearity without saturation.
    small = uniform(1).requires_grad_(requires_grad=True)
    large = uniform(8).requires_grad_(requires_grad=True)
    small_gradient = torch.autograd.grad(preprocessor.encode_image(small).sum(), small)[0]
    large_gradient = torch.autograd.grad(preprocessor.encode_image(large).sum(), large)[0]
    assert torch.isfinite(small_gradient).all()
    assert torch.isfinite(large_gradient).all()
    assert (small_gradient > 0).all()
    assert (large_gradient > 0).all()
    torch.testing.assert_close(
        large_gradient / small_gradient, torch.full_like(small_gradient, math.sqrt(2 / 65)), rtol=1e-5, atol=1e-7
    )
    # A finite difference tests the complete resize + ImageNet gradient path.
    probe = uniform(2).requires_grad_(requires_grad=True)
    target = (0, 2, 1, 3, 2)
    analytic = torch.autograd.grad(preprocessor(probe)[2, 1, 111, 111], probe)[0][target]
    step = float(rms[1]) * 0.001
    plus, minus = probe.detach().clone(), probe.detach().clone()
    plus[target] += step
    minus[target] -= step
    finite_difference = (preprocessor(plus)[2, 1, 111, 111] - preprocessor(minus)[2, 1, 111, 111]) / (2 * step)
    torch.testing.assert_close(analytic, finite_difference, rtol=0.003, atol=1e-4)

    # Even maximal finite FP32 forces keep finite, unclipped images and positive gradients.
    extreme = torch.full_like(zero, torch.finfo(torch.float32).max).requires_grad_(requires_grad=True)
    extreme_images = preprocessor.encode_image(extreme)
    assert torch.isfinite(extreme_images).all()
    assert extreme_images[:, :, :, 32:192].min() > 1
    assert torch.isfinite(preprocessor(-extreme)).all()
    assert preprocessor.encode_image(-extreme)[:, :, :, 32:192].max() < 0
    extreme_gradient = torch.autograd.grad(extreme_images.sum(), extreme)[0]
    assert torch.isfinite(extreme_gradient).all()
    assert (extreme_gradient > 0).all()
    assert torch.equal(extreme_images[:, :, :, :32], torch.full_like(extreme_images[:, :, :, :32], 0.5))
    for invalid_rms in ([0, 1, 1], [-1, 1, 1], [math.inf, 1, 1], [1, 1]):
        reject(lambda values=invalid_rms: TactileDinoPreprocessor(values))
    reject(lambda: TactileDinoPreprocessor(rms, reference_multiple=0))
    reject(lambda: TactileDinoPreprocessor(rms, image_size=225))
    reject(lambda: preprocessor.encode_image(torch.zeros(1, 5, 3, 5, 7)))
    reject(lambda: preprocessor.encode_image(torch.zeros(1, 5, 3, 7, 5, dtype=torch.int64)))
    reject(lambda: preprocessor.encode_image(torch.zeros(0, 5, 3, 7, 5)))
    reject(lambda: preprocessor.encode_image(torch.full_like(zero, math.nan)))
    print(
        json.dumps(
            {
                "status": "passed",
                "scope": "synthetic_cpu_tactile_preprocessor_only_no_dino_or_training",
                "formula_origin": "explicit_local_choice_not_published_touchguide_formula",
                "output_shape": list(images.shape),
                "output_dtype": str(images.dtype),
                "zero_sign_magnitude_channels_padding_verified": True,
                "batch_finger_independence_and_spatial_orientation_verified": True,
                "nonlinear_and_finite_difference_gradients_verified": True,
                "max_finite_fp32_unclipped_images_and_nonzero_gradients_verified": True,
                "finite_difference": float(finite_difference),
                "analytic_gradient": float(analytic),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
