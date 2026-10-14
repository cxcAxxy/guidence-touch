"""Regression checks for state, execution prefix and configuration-only deployment."""

# ruff: noqa: PT009, PT027
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from openpi.touchguide.inference import ActionCoordinates
from openpi.touchguide.inference import LoadedTouchGuide
from openpi.touchguide.inference import load_touchguide
from openpi.touchguide.policy import PokerTouchGuidePolicy
from openpi.touchguide.policy_test import RecordingTokenizer
from openpi.touchguide.sampler import GuidanceConfig
from openpi.touchguide.sampler import PrefixCache
from openpi.touchguide.sampler import TouchGuideSampler
from openpi.touchguide.sampler_test import TinyBase
from openpi.touchguide.sampler_test import TinyCPM


class StateScorer(TinyCPM):
    uses_state = True

    def encode_observation(self, head, wrist, tactile, *, state_raw):
        if state_raw.shape != (1, 27):
            raise ValueError(f"Incorrect state batch shape {state_raw.shape}")
        self.received_state = state_raw.clone()
        return torch.ones(1, 1024) / 32


class DeploymentTest(unittest.TestCase):
    def test_policy_state_shape_and_execution_contract_without_source_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "meta").mkdir()
            (root / "meta/manifest.json").write_text(json.dumps({"sources": {"pi05_root": "/missing-source-data"}}))
            (root / "meta/task_instruction.json").write_text(json.dumps({"task": "poker"}))
            stats = {key: {"q01": [-1.0] * 27, "q99": [1.0] * 27} for key in ("state", "actions")}
            scorer = StateScorer()
            sampler = TouchGuideSampler(TinyBase(), scorer, GuidanceConfig(), precision="float32")
            loaded = LoadedTouchGuide(sampler, ActionCoordinates(stats), {"data_root": str(root)})
            tokenizer_path = root / "tokenizer.model"
            tokenizer_path.touch()
            with patch("openpi.models.tokenizer.PaligemmaTokenizer", return_value=RecordingTokenizer()):
                policy = PokerTouchGuidePolicy(loaded, tokenizer_model=tokenizer_path, execute_steps=5)
            request = {
                "variant": "pi05+touchguidance",
                "state": np.arange(27, dtype=np.float32) / 100,
                "head_image": np.zeros((240, 320, 3), dtype=np.uint8),
                "right_wrist_image": np.zeros((240, 320, 3), dtype=np.uint8),
                "tactile_map_raw": np.zeros((5, 3, 7, 5), dtype=np.float32),
                "episode_seed": 1,
                "request_index": 0,
                "execute_steps": 5,
                "return_trace": True,
                "compare_base": True,
            }

            def prefix(base, obs):
                return PrefixCache(obs.state, torch.ones(1, 1, dtype=torch.bool), None)

            with patch("openpi.touchguide.policy.prepare_pi05_prefix", side_effect=prefix):
                result = policy.infer(request)
            self.assertEqual(result["actions"].shape, (30, 27))
            np.testing.assert_array_equal(scorer.received_state.numpy()[0], request["state"])
            self.assertEqual(policy.metadata["execute_steps"], 5)
            self.assertEqual(policy.metadata["replan_period_sim_s"], 5 / 30)
            self.assertEqual(result["audit"]["guidance_updates"].shape, (10, 30, 32))
            with self.assertRaisesRegex(ValueError, "execute_steps"):
                policy.infer({**request, "execute_steps": 30})

    def test_recovery_metadata_must_match_checkpoint_normalization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            names = [
                "config/cpm.json",
                "meta/tactile_scale.json",
                "meta/action_norm_stats.json",
                "meta/cpm_normalization_v1.json",
                "meta/negative_policy.json",
            ]
            settings = {"recovery": {"normalization": {"file": names[3]}, "negative_policy": {"file": names[4]}}}
            for name in names:
                p = root / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps(settings if name == names[0] else {}))
            checkpoint = root / "test.pt"
            checkpoint.touch()
            payload = {
                "format": "openpi_touchguide_cpm_train_v1",
                "contract": {
                    "model_variant": "recovery-state-cpm",
                    "data_sha256": {n: hashlib.sha256((root / n).read_bytes()).hexdigest() for n in names},
                    "noise": {"num_flow_steps": 10},
                },
            }
            (root / names[3]).write_text('{"tampered": true}')
            with patch("torch.load", return_value=payload), self.assertRaisesRegex(ValueError, "cpm_normalization_v1"):
                load_touchguide(
                    base_checkpoint=root, cpm_checkpoint=checkpoint, data_root=root, device="cpu", precision="float32"
                )

    def test_export_copies_only_configuration_and_preserves_hashes(self):
        path = Path(__file__).resolve().parents[3] / "scripts/touchguide/export_oa_inference_metadata.py"
        spec = importlib.util.spec_from_file_location("oa_metadata_export", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            root.mkdir()
            files = {
                "config/cpm.json": {},
                "meta/tactile_scale.json": {},
                "meta/action_norm_stats.json": {},
                "meta/manifest.json": {"sources": {}},
            }
            for name, value in files.items():
                p = root / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps(value))
            (root / "samples.h5").write_bytes(b"not for deployment")
            out = Path(directory) / "bundle"
            report = module.export(root, out, "poker")
            self.assertFalse(report["contains_samples"])
            self.assertFalse((out / "samples.h5").exists())
            for name in files:
                self.assertEqual((root / name).read_bytes(), (out / name).read_bytes())
            self.assertEqual(json.loads((out / "meta/task_instruction.json").read_text())["task"], "poker")


if __name__ == "__main__":
    unittest.main()
