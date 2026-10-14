"""Deterministic fixture checks for shared frozen DINOv2 CPM; no weights or optimizer."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn
from torch.nn import functional

from openpi.touchguide import model as cpm
from openpi.touchguide.losses import cpm_contrastive_loss


class FixtureDinov2(nn.Module):
    """Frozen pixel-responsive surrogate, with the real DINO feature dimensions."""

    def __init__(self, directory: str | Path, *, batch_size: int = 16):
        super().__init__()
        self.feature_dim = 768
        self.batch_size = batch_size
        self.backbone = nn.Linear(3, 768).requires_grad_(requires_grad=False).eval()
        self.register_buffer("rgb_mean", torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1))
        self.register_buffer("rgb_std", torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1))
        self.calls: list[torch.Tensor] = []
        self.source_manifest = {"fixture": True, "directory": str(directory)}

    def train(self, mode: bool = True):  # noqa: FBT001, FBT002 - nn.Module public signature
        super().train(mode)
        self.backbone.eval()
        return self

    def preprocess(self, rgb: torch.Tensor) -> torch.Tensor:
        if rgb.shape[1:] != (3, 224, 224) or not rgb.is_floating_point() or not torch.isfinite(rgb).all():
            raise ValueError("Expected finite fixture RGB [B,3,224,224]")
        if rgb.numel() == 0 or rgb.min() < -1.0001 or rgb.max() > 1.0001:
            raise ValueError("Fixture RGB must be in [-1,1]")
        return ((rgb + 1) * 0.5 - self.rgb_mean) / self.rgb_std

    def features(self, pixels: torch.Tensor) -> torch.Tensor:
        self.backbone.eval()
        with torch.no_grad():
            patches = functional.adaptive_avg_pool2d(pixels, (16, 16)).flatten(2).transpose(1, 2)
            patch_features = self.backbone(patches)
            cls = self.backbone(pixels.mean(dim=(-2, -1))).unsqueeze(1)
            return torch.cat([cls, patch_features], dim=1)

    def forward_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
        if pixels.shape[1:] != (3, 224, 224) or not pixels.is_floating_point() or not torch.isfinite(pixels).all():
            raise ValueError("Expected finite fixture DINO pixels [B,3,224,224]")
        self.calls.append(pixels.detach().clone())
        return self.features(pixels)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        return self.forward_pixels(self.preprocess(rgb))


def expect_value_error(call, label: str) -> None:
    try:
        call()
    except ValueError:
        return
    raise AssertionError(f"Invalid input accepted: {label}")


class PixelBackbone(nn.Module):
    """Expose the Hugging Face forward contract without pretrained data."""

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 768)
        self.calls = []

    def forward(self, *, pixel_values: torch.Tensor) -> SimpleNamespace:
        self.calls.append((len(pixel_values), self.training, torch.is_grad_enabled()))
        patches = functional.adaptive_avg_pool2d(pixel_values, (16, 16)).flatten(2).transpose(1, 2)
        features = self.projection(patches)
        cls = self.projection(pixel_values.mean(dim=(-2, -1))).unsqueeze(1)
        return SimpleNamespace(last_hidden_state=torch.cat([cls, features], dim=1))


def check_real_wrapper_methods() -> dict[str, object]:
    # Run actual FrozenDinov2 train/preprocess/forward methods while skipping its
    # pretrained constructor. This catches wrapper regressions hidden by the mock.
    vision = cpm.FrozenDinov2.__new__(cpm.FrozenDinov2)
    nn.Module.__init__(vision)
    vision.backbone = PixelBackbone().requires_grad_(requires_grad=False).eval()
    vision.feature_dim = 768
    vision.batch_size = 2
    vision.register_buffer("rgb_mean", torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1))
    vision.register_buffer("rgb_std", torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1))
    rgb = torch.stack([torch.full((3, 224, 224), -0.8 + sample * 0.3) for sample in range(5)])
    rgb.requires_grad_(requires_grad=True)
    pixels = vision.preprocess(rgb)
    expected_pixels = ((rgb + 1) * 0.5 - vision.rgb_mean) / vision.rgb_std
    torch.testing.assert_close(pixels, expected_pixels, rtol=0, atol=0)
    vision.train()
    assert not vision.backbone.training
    output = vision(rgb)
    assert output.shape == (5, 257, 768)
    assert not output.requires_grad
    assert vision.backbone.calls == [(2, False, False), (2, False, False), (1, False, False)]
    # The pixel API consumes already-normalized forces, including values beyond
    # the normal RGB range; it must not normalize again or clamp.
    unbounded = torch.full_like(rgb, 100)
    with torch.no_grad():
        expected = vision.backbone(pixel_values=unbounded).last_hidden_state
    actual = vision.forward_pixels(unbounded)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-5)
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in vision.parameters())
    expect_value_error(lambda: vision.preprocess(rgb * 10), "real RGB range")
    expect_value_error(lambda: vision.forward_pixels(unbounded[:, :, :-1]), "real pixel shape")
    expect_value_error(lambda: vision.forward_pixels(torch.full_like(unbounded, float("nan"))), "real pixel nonfinite")
    return {
        "actual_train_preprocess_forward_methods": True,
        "pretrained_constructor_called": False,
        "microbatch_order_preserved": True,
        "pixel_input_unclipped_not_normalized_twice": True,
        "backbone_eval_and_no_grad": True,
    }


def check_pooling() -> dict[str, object]:
    # Distinct batch offsets, CLS values, row/column values and channels make
    # transposition, flattening and cross-example averaging independently visible.
    features = torch.empty(7, 257, 3)
    grid = torch.arange(256, dtype=torch.float32).reshape(16, 16)
    for sample in range(7):
        features[sample, 0] = -1000 - sample * 10 - torch.arange(3)
        features[sample, 1:] = grid.flatten()[:, None] + sample * 1000 + torch.arange(3) / 4
    pooled = cpm.CPMModel.pool_dino_tokens(features)
    expected = features[:, 1:].reshape(7, 16, 16, 3).reshape(7, 8, 2, 8, 2, 3).mean(dim=(2, 4))
    torch.testing.assert_close(pooled[:, 0], features[:, 0], rtol=0, atol=0)
    torch.testing.assert_close(pooled[:, 1:], expected.reshape(7, 64, 3), rtol=0, atol=0)
    assert pooled.shape == (7, 65, 3)
    expect_value_error(lambda: cpm.CPMModel.pool_dino_tokens(features[:, :-1]), "wrong DINO patch count")
    return {"shape": list(pooled.shape), "cls_preserved": True, "exact_spatial_2x2_means": True}


def fixture_batch() -> dict[str, torch.Tensor]:
    size = 3
    head = torch.stack([torch.full((3, 224, 224), -0.8 + sample * 0.4) for sample in range(size)])
    wrist = torch.stack([torch.full((3, 224, 224), 0.7 - sample * 0.3) for sample in range(size)])
    tactile = torch.randn(size, 5, 3, 7, 5) * torch.tensor([0.3, 0.6, 0.9]).reshape(1, 1, 3, 1, 1)
    tactile[:, :, 0] += torch.arange(5).reshape(1, 5, 1, 1) * 0.2
    positive = torch.eye(size, dtype=torch.bool)
    return {
        "head": head,
        "wrist": wrist,
        "tactile_map_raw": tactile,
        "noisy_actions": torch.randn(size, 30, 27),
        "flow_time": torch.tensor(0.4),
        "targets": torch.arange(size),
        "positive_mask": positive,
        "negative_mask": ~positive,
    }


def check_fusion_order(model: cpm.CPMModel, batch: dict[str, torch.Tensor]) -> dict[str, object]:
    captured = []
    handle = model.fusion.register_forward_pre_hook(lambda _module, args: captured.append(args[0].detach().clone()))
    model.vision.calls.clear()
    with torch.no_grad():
        model.encode_observation(batch["head"], batch["wrist"], batch["tactile_map_raw"])
    handle.remove()
    size = len(batch["head"])
    assert len(model.vision.calls) == 1, "Every view and finger should share one concatenated backbone call"
    pixels = model.vision.calls[0]
    assert pixels.shape == (7 * size, 3, 224, 224)
    torch.testing.assert_close(pixels[:size], model.vision.preprocess(batch["head"]), rtol=0, atol=0)
    torch.testing.assert_close(pixels[size : size * 2], model.vision.preprocess(batch["wrist"]), rtol=0, atol=0)
    touch_pixels = model.tactile_preprocessor(batch["tactile_map_raw"])
    torch.testing.assert_close(pixels[size * 2 :], touch_pixels, rtol=0, atol=0)
    features = cpm.CPMModel.pool_dino_tokens(model.vision.features(pixels))
    head = model.vision_projection(features[:size]) + model.modality_embedding[0]
    wrist = model.vision_projection(features[size : size * 2]) + model.modality_embedding[1]
    touch = model.tactile_projection(features[size * 2 :]).reshape(size, 5, 65, model.config.embedding_dim)
    touch = touch + model.modality_embedding[2] + model.finger_embedding.reshape(1, 5, 1, -1)
    expected = torch.cat([model.observation_cls.expand(size, -1, -1), head, wrist, touch.flatten(1, 2)], dim=1)
    assert captured[0].shape == (size, 456, model.config.embedding_dim)
    torch.testing.assert_close(captured[0], expected, rtol=0, atol=0)
    return {"fusion_tokens": 456, "ordered_camera_and_finger_tokens": True, "shared_forward_pixels_calls": 1}


def check_default_meta() -> dict[str, object]:
    config = cpm.CPMConfig()
    assert (config.embedding_dim, config.fusion_layers, config.fusion_heads, config.fusion_ff_dim) == (
        1024,
        8,
        16,
        4096,
    )
    assert config.action_channels == (256, 512, 1024)
    assert config.action_mlp_hidden == 2048
    assert config.action_pool_bins == 6
    assert config.time_embedding_dim == 64
    # Validate the few scalar preprocessing buffers on CPU before moving them
    # to meta; its constructor correctly performs data-dependent RMS validation.
    preprocessor = cpm.TactileDinoPreprocessor((0.3, 0.6, 0.9)).to("meta")
    with mock.patch.object(cpm, "TactileDinoPreprocessor", return_value=preprocessor), torch.device("meta"):
        model = cpm.CPMModel(config, vision_directory="fixture-only", tactile_rms=(0.3, 0.6, 0.9))
    counts = model.parameter_counts()
    assert counts["trainable"] == 122_262_529, counts
    assert all(parameter.device.type == "meta" for parameter in model.parameters())
    assert len([module for module in model.modules() if isinstance(module, FixtureDinov2)]) == 1
    names = set(dict(model.named_parameters()))
    assert not any(part in name for name in names for part in ("tactile_encoder", "tactile_magnitude", "no_contact"))
    convs = [module for module in model.action_encoder.temporal if isinstance(module, nn.Conv1d)]
    assert [(layer.in_channels, layer.out_channels) for layer in convs] == [(27, 256), (256, 512), (512, 1024)]
    mlp = [module for module in model.action_encoder.projection if isinstance(module, nn.Linear)]
    assert [(layer.in_features, layer.out_features) for layer in mlp] == [(6144, 2048), (2048, 1024)]
    assert all(not parameter.requires_grad for parameter in model.vision.parameters())
    return {"parameter_counts_with_fixture_backbone": counts, "default_model_allocated_on_meta_only": True}


def check_gradients(model: cpm.CPMModel, batch: dict[str, torch.Tensor]) -> dict[str, object]:
    model.train()
    assert not model.vision.backbone.training
    assert all(not parameter.requires_grad for parameter in model.vision.backbone.parameters())
    output = model(batch)
    size = len(batch["head"])
    for key in ("observation_embedding", "action_embedding"):
        assert output[key].shape == (size, 64)
        torch.testing.assert_close(output[key].norm(dim=-1), torch.ones(size), rtol=1e-5, atol=1e-5)
    scores = output["observation_embedding"] @ output["action_embedding"].T
    torch.testing.assert_close(output["cosine_scores"], scores, rtol=0, atol=0)
    torch.testing.assert_close(output["logits"], scores / output["temperature"], rtol=0, atol=0)
    metrics = cpm_contrastive_loss(output, batch)
    expected_loss = (
        functional.cross_entropy(output["logits"], batch["targets"])
        + functional.cross_entropy(output["logits"].T, batch["targets"])
    ) / 2
    torch.testing.assert_close(metrics["loss"], expected_loss, rtol=0, atol=0)
    metrics["loss"].backward()
    gradients = defaultdict(float)
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, f"Missing trainable gradient: {name}"
            assert torch.isfinite(parameter.grad).all(), f"Nonfinite gradient: {name}"
            gradients[name.split(".")[0]] += float(parameter.grad.square().sum())
        else:
            assert parameter.grad is None, f"Frozen parameter received gradient: {name}"
    assert all(value > 0 for value in gradients.values()), dict(gradients)
    model.zero_grad(set_to_none=True)
    empty = {**batch, "tactile_map_raw": torch.zeros_like(batch["tactile_map_raw"])}
    empty_output = model(empty)
    assert all(torch.isfinite(value).all() for value in empty_output.values())
    cpm_contrastive_loss(empty_output, empty)["loss"].backward()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    model.zero_grad(set_to_none=True)
    return {
        "loss": float(metrics["loss"].detach()),
        "gradient_squared_norms": dict(gradients),
        "zero_touch_finite": True,
    }


def check_guidance(model: cpm.CPMModel, batch: dict[str, torch.Tensor]) -> dict[str, object]:
    model.eval()
    assert not model.vision.backbone.training
    with torch.no_grad():
        obs = model.encode_observation(batch["head"], batch["wrist"], batch["tactile_map_raw"])
        stronger = model.encode_observation(batch["head"], batch["wrist"], batch["tactile_map_raw"] * 2)
        force_difference = float((obs - stronger).norm(dim=-1).mean())
        assert force_difference > 1e-6
        time_difference = float(
            (model.action_encoder(batch["noisy_actions"], 0.0) - model.action_encoder(batch["noisy_actions"], 0.4))
            .norm(dim=-1)
            .mean()
        )
        assert time_difference > 1e-6
        before = model.score_actions(obs, batch["noisy_actions"], 0.4)
        original_scale = model.logit_scale.detach().clone()
        model.logit_scale.add_(0.5)
        after = model.score_actions(obs, batch["noisy_actions"], 0.4)
        torch.testing.assert_close(before, after, rtol=0, atol=0)
        model.logit_scale.copy_(original_scale)
    model.requires_grad_(requires_grad=False)
    candidate = batch["noisy_actions"].detach().clone().requires_grad_(requires_grad=True)
    scores = model.paired_scores(obs, candidate, 0.4)
    assert scores.shape == (3,)
    gradient = torch.autograd.grad(scores.sum(), candidate)[0]
    assert gradient.shape == (3, 30, 27)
    assert torch.isfinite(gradient).all()
    assert gradient.norm() > 0
    assert all(parameter.grad is None for parameter in model.parameters())
    with torch.no_grad():
        matrix = model.score_actions(obs, candidate, 0.4)
        torch.testing.assert_close(scores, matrix.diagonal(), rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(model.paired_scores(obs[:1], candidate, 0.4), matrix[0], rtol=1e-5, atol=1e-6)
        direction = gradient / gradient.norm()
        plus = model.paired_scores(obs, candidate + 0.01 * direction, 0.4).sum()
        minus = model.paired_scores(obs, candidate - 0.01 * direction, 0.4).sum()
        finite_difference = (plus - minus) / 0.02
        analytic = (gradient * direction).sum()
        torch.testing.assert_close(finite_difference, analytic, rtol=0.03, atol=1e-4)
    return {
        "touch_amplitude_embedding_l2_change": force_difference,
        "flow_time_embedding_l2_change": time_difference,
        "temperature_independent_raw_scores": True,
        "frozen_model_action_gradient_rms": float(gradient.square().mean().sqrt()),
        "directional_finite_difference": float(finite_difference),
        "directional_autograd": float(analytic),
    }


def check_invalid(model: cpm.CPMModel, batch: dict[str, torch.Tensor]) -> dict[str, object]:
    cases = {
        "missing_noisy_actions": lambda: model({key: value for key, value in batch.items() if key != "noisy_actions"}),
        "legacy_negative_actions": lambda: model({**batch, "negative_actions": batch["noisy_actions"]}),
        "unshared_time": lambda: model({**batch, "flow_time": torch.full((3,), 0.4)}),
        "action_shape": lambda: model.action_encoder(batch["noisy_actions"][:, :-1], 0.4),
        "action_integer": lambda: model.action_encoder(batch["noisy_actions"].long(), 0.4),
        "action_nonfinite": lambda: model.action_encoder(torch.full_like(batch["noisy_actions"], float("nan")), 0.4),
        "time_out_of_range": lambda: model.action_encoder(batch["noisy_actions"], 1.1),
        "tactile_shape": lambda: model.tactile_preprocessor(batch["tactile_map_raw"][:, :4]),
        "tactile_nonfinite": lambda: model.tactile_preprocessor(
            torch.full_like(batch["tactile_map_raw"], float("inf"))
        ),
        "rgb_shape": lambda: model.encode_observation(
            batch["head"][:, :, :-1], batch["wrist"], batch["tactile_map_raw"]
        ),
        "rgb_range": lambda: model.encode_observation(batch["head"] * 10, batch["wrist"], batch["tactile_map_raw"]),
        "mismatched_batches": lambda: model.encode_observation(
            batch["head"][:2], batch["wrist"], batch["tactile_map_raw"]
        ),
        "unnormalized_cached_observation": lambda: model.score_actions(torch.ones(3, 64), batch["noisy_actions"], 0.4),
    }
    for label, call in cases.items():
        expect_value_error(call, label)
    with tempfile.TemporaryDirectory(prefix="validate-dino-cpm-", dir="/tmp") as temporary:
        root = Path(temporary)
        (root / "config").mkdir()
        (root / "config/cpm.json").write_text(json.dumps({"schema": "openpi_touchguide_cpm_config_v1"}))
        expect_value_error(lambda: cpm.CPMModel.from_config(root), "old schema")
    for kwargs in ({"embedding_dim": 63}, {"action_channels": (16, 0, 64)}, {"action_mlp_hidden": 0}):
        expect_value_error(lambda kwargs=kwargs: cpm.CPMConfig(**kwargs), "invalid model dimensions")
    return {"rejected_inputs": [*cases, "old_schema", "invalid_model_dimensions"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=Path("/tmp/dino_cpm_validation.json"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    wrapper = check_real_wrapper_methods()
    with mock.patch.object(cpm, "FrozenDinov2", FixtureDinov2):
        default = check_default_meta()
        pooling = check_pooling()
        config = cpm.CPMConfig(
            embedding_dim=64,
            fusion_layers=2,
            fusion_heads=4,
            fusion_ff_dim=256,
            dropout=0.0,
            action_channels=(16, 32, 64),
            action_mlp_hidden=128,
        )
        model = cpm.CPMModel(config, vision_directory="fixture-only", tactile_rms=(0.3, 0.6, 0.9))
        batch = fixture_batch()
        order = check_fusion_order(model, batch)
        gradients = check_gradients(model, batch)
        guidance = check_guidance(model, batch)
        invalid = check_invalid(model, batch)
    report = {
        "status": "passed",
        "scope": "deterministic_cpu_mock_backbone_architecture_and_gradient_fixtures",
        "pretrained_weights_loaded": False,
        "real_dinov2_executed": False,
        "training_started": False,
        "optimizer_created": False,
        "checkpoint_saved": False,
        "pi05_loaded": False,
        "actual_frozen_wrapper_methods": wrapper,
        "default_meta": default,
        "pooling": pooling,
        "token_order": order,
        "gradient_checks": gradients,
        "guidance_checks": guidance,
        "invalid_input_checks": invalid,
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "passed", "report": str(args.report), "default_trainable_parameters": 122_262_529}))


if __name__ == "__main__":
    main()
