"""Simulation payload contracts, without loading policy weights or using a GPU."""

# ruff: noqa: PT009, PT027 -- stdlib unittest is available in the inference environment

import unittest

import numpy as np
import torch

from openpi.touchguide.policy import ObservationPreprocessor
from openpi.touchguide.policy import noise_seed


class RecordingTokenizer:
    def tokenize(self, prompt, state):
        self.state = state.copy()
        return np.zeros(200, dtype=np.int64), np.ones(200, dtype=bool)


class PolicyInterfaceTest(unittest.TestCase):
    def test_pair_noise_is_independent_of_server_request_history(self):
        before = noise_seed(7, 2)
        for seed in range(100):
            noise_seed(seed, 40)
        self.assertEqual(noise_seed(7, 2), before)
        self.assertNotEqual(before, noise_seed(7, 3))
        self.assertNotEqual(before, noise_seed(8, 2))
        a = torch.randn((1, 30, 32), generator=torch.Generator().manual_seed(before))
        b = torch.randn((1, 30, 32), generator=torch.Generator().manual_seed(noise_seed(7, 2)))
        self.assertTrue(torch.equal(a, b))

    def test_state_is_tokenized_before_padding_without_clipping(self):
        stats = {"state": {"q01": [-1.0] * 27, "q99": [1.0] * 27}}
        tokenizer = RecordingTokenizer()
        preprocessor = ObservationPreprocessor(stats, tokenizer, "cpu")
        rgb = np.full((240, 320, 3), 255, dtype=np.uint8)
        obs, raw = preprocessor.prepare(
            {"state": np.full(27, 2.0, dtype=np.float32), "head_image": rgb, "right_wrist_image": rgb}, "poker"
        )
        self.assertEqual(tokenizer.state.shape, (27,))
        self.assertEqual(tokenizer.state.dtype, np.float64)
        self.assertGreater(float(tokenizer.state.min()), 1.0)
        self.assertEqual(obs.state.shape, (1, 32))
        self.assertEqual(raw.shape, (1, 27))
        self.assertEqual(torch.count_nonzero(obs.state[:, 27:]), 0)
        self.assertEqual(obs.images["base_0_rgb"].shape, (1, 3, 224, 224))
        self.assertTrue(torch.all(obs.images["left_wrist_0_rgb"] == -1))
        self.assertFalse(bool(obs.image_masks["left_wrist_0_rgb"][0]))
        self.assertTrue(bool(obs.image_masks["right_wrist_0_rgb"][0]))

    def test_payload_rejects_full_bilateral_state(self):
        preprocessor = ObservationPreprocessor(
            {"state": {"q01": [-1.0] * 27, "q99": [1.0] * 27}}, RecordingTokenizer(), "cpu"
        )
        with self.assertRaisesRegex(ValueError, "27 finite"):
            preprocessor.prepare({"state": np.zeros(54)}, "poker")


if __name__ == "__main__":
    unittest.main()
