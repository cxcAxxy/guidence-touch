"""CPU contracts for state-CPM; no policy/DINO download, actual training or W&B."""

# ruff: noqa: PT009, PT027 -- runnable with stdlib unittest in the training environment

from __future__ import annotations

import unittest
from unittest import mock

import torch
from torch import nn

from openpi.touchguide.batching import collate_matched_samples
from openpi.touchguide.inference import ActionCoordinates
from openpi.touchguide.losses import cpm_contrastive_loss
from openpi.touchguide.model import CPMConfig
from openpi.touchguide.model import CPMModel
from openpi.touchguide.noise import FlowActionNoise
from openpi.touchguide.noise import NoiseConfig
from openpi.touchguide.state_model import StateCPMModel


class FakeDino(nn.Module):
    """Small deterministic spatial features in place of a frozen public backbone."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.feature_dim = 768
        self.scale = nn.Parameter(torch.tensor(1.0), requires_grad=False)

    def preprocess(self, pixels):
        return pixels

    def forward_pixels(self, pixels):
        signal = pixels.mean(dim=(1, 2, 3))[:, None, None]
        channels = torch.linspace(-1, 1, 768, device=pixels.device)[None, None, :]
        positions = torch.linspace(-0.5, 0.5, 257, device=pixels.device)[None, :, None]
        return signal * channels + positions


def make_model(*, state=True):
    config = CPMConfig(
        embedding_dim=32,
        fusion_layers=1,
        fusion_heads=4,
        fusion_ff_dim=64,
        dropout=0,
        action_pool_bins=2,
        time_embedding_dim=8,
        action_channels=(8, 16, 24),
        action_mlp_hidden=32,
    )
    kwargs = {"vision_directory": "/unused", "tactile_rms": (0.1, 0.2, 0.3)}
    with mock.patch("openpi.touchguide.model.FrozenDinov2", FakeDino):
        if state:
            return StateCPMModel(config, **kwargs, state_stats={"q01": [-2.0] * 27, "q99": [2.0] * 27})
        return CPMModel(config, **kwargs)


def make_batch(size=2):
    samples = [
        {
            "id": f"test-{i}",
            "metadata": {"split": "train", "episode_index": 0, "frame_index": i},
            "head": torch.rand(3, 224, 224) * 2 - 1,
            "wrist": torch.rand(3, 224, 224) * 2 - 1,
            "tactile_map_raw": torch.rand(5, 3, 7, 5) * 0.1,
            "state_raw": torch.rand(27),
            "contact_fingers": torch.ones(5, dtype=torch.bool),
            "actions": torch.rand(30, 27),
        }
        for i in range(size)
    ]
    clean = collate_matched_samples(samples, include_auxiliary=True)
    return FlowActionNoise(NoiseConfig()).augment_at_time(clean, 0.3)


class StateCPMTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(42)

    def test_state_uses_original_state_quantiles_without_clipping(self):
        model = make_model()
        stats = {"state": {"q01": [-2.0] * 27, "q99": [2.0] * 27}, "actions": {"q01": [-10.0] * 27, "q99": [10.0] * 27}}
        raw = torch.linspace(-3, 3, 54).reshape(2, 27)
        self.assertTrue(torch.equal(model.normalize_state(raw), ActionCoordinates(stats).normalize_state(raw)))
        self.assertGreater(model.normalize_state(raw).abs().max().item(), 1)

    def test_one_state_token_affects_observation_and_receives_gradients(self):
        model, batch = make_model(), make_batch()
        token_counts = []
        hook = model.fusion.register_forward_pre_hook(lambda module, inputs: token_counts.append(inputs[0].shape[1]))
        output = model(batch)
        cpm_contrastive_loss(output, batch)["loss"].backward()
        hook.remove()
        self.assertEqual(token_counts, [457])
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        self.assertGreater(model.state_encoder[0].weight.grad.norm().item(), 0)
        self.assertGreater(model.state_modality_embedding.grad.norm().item(), 0)
        self.assertIsNone(model.vision.scale.grad)
        model.eval()
        with torch.no_grad():
            before = model.encode_observation_batch(batch)
            after = model.encode_observation_batch({**batch, "state_raw": batch["state_raw"] + 0.5})
        self.assertFalse(torch.equal(before, after))

    def test_raw_and_already_normalized_state_are_equivalent(self):
        model, batch = make_model().eval(), make_batch()
        with torch.no_grad():
            raw = model.encode_observation_batch(batch)
            normalized = model.encode_observation(
                batch["head"],
                batch["wrist"],
                batch["tactile_map_raw"],
                state_normalized=model.normalize_state(batch["state_raw"]),
            )
        self.assertTrue(torch.equal(raw, normalized))
        saved = batch["state_raw"].clone()
        FlowActionNoise(NoiseConfig()).augment_at_time(
            {
                key: value
                for key, value in batch.items()
                if key not in ("noisy_actions", "flow_time", "noise_level", "noise_epsilon")
            },
            0.9,
        )
        self.assertTrue(torch.equal(saved, batch["state_raw"]))

    def test_missing_or_invalid_state_is_rejected(self):
        model, batch = make_model(), make_batch()
        with self.assertRaisesRegex(ValueError, "state_raw"):
            model({key: value for key, value in batch.items() if key != "state_raw"})
        for value in (torch.zeros(2, 32), torch.full((2, 27), float("nan"))):
            with self.assertRaises(ValueError):
                model({**batch, "state_raw": value})

    def test_original_cpm_keeps_456_tokens_and_no_state_parameters(self):
        model, batch = make_model(state=False), make_batch()
        self.assertFalse(model.uses_state)
        self.assertFalse(any(name.startswith("state_") for name in model.state_dict()))
        self.assertEqual(
            model.observation_tokens(batch["head"], batch["wrist"], batch["tactile_map_raw"]).shape[1], 456
        )
        model.eval()
        with torch.no_grad():
            first = model(batch)["cosine_scores"]
            second = model({**batch, "state_raw": batch["state_raw"] + 5})["cosine_scores"]
        self.assertTrue(torch.equal(first, second))


if __name__ == "__main__":
    unittest.main()
