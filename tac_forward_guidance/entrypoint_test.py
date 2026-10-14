"""Launcher/protocol checks without loading a checkpoint or running simulation."""

# ruff: noqa: PT009, PT027 -- keep tests runnable with stdlib unittest

import argparse
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from openpi.touchguide.inference import ActionCoordinates
from openpi.touchguide.sampler import GuidanceConfig
from openpi.touchguide.sampler import TouchGuideSampler
from openpi.touchguide.sampler_test import TinyBase
from openpi.touchguide.sampler_test import TinyCPM
from tac_forward_guidance import evaluate
from tac_forward_guidance import rollout
from tac_forward_guidance.config import add_guidance_arguments
from tac_forward_guidance.config import config_from_args
from tac_forward_guidance.runtime import FORWARD_VARIANT
from tac_forward_guidance.runtime import POLICY_SCHEMA
from tac_forward_guidance.runtime import PokerForwardGuidancePolicy
from tac_forward_guidance.sampler_test import fixture


class ObservationCPM(TinyCPM):
    def encode_observation(self, head, wrist, maps):
        return maps.mean(dim=(1, 2, 3, 4))[:, None]


class EntrypointTest(unittest.TestCase):
    def test_default_config_and_zero_cap_cli(self):
        parser = argparse.ArgumentParser()
        add_guidance_arguments(parser)
        config = config_from_args(parser.parse_args([]))
        self.assertEqual(config.scale, 0.3)
        self.assertEqual(config.gradient_mode, "full")
        self.assertFalse(config.local_score_probe)
        self.assertIsNone(config_from_args(parser.parse_args(["--max-update-rms", "0"])).max_update_rms)

    def test_launcher_forwards_every_sampling_setting(self):
        args = SimpleNamespace(
            model_python=Path("/model/python"),
            base_checkpoint=Path("/base"),
            cpm_checkpoint=Path("/cpm"),
            data_root=Path("/data"),
            tokenizer_model=Path("/tokenizer"),
            guidance_scale=0.3,
            guidance_steps=3,
            num_steps=10,
            gradient_mode="full",
            max_update_rms=0.01,
            precision="bfloat16",
            local_score_probe=False,
            port=18787,
        )
        command = evaluate.server_command(args, Path("/output"))
        self.assertTrue(command[2].endswith("tac_forward_guidance/serve.py"))
        for flag, expected in (
            ("--gradient-mode", "full"),
            ("--max-update-rms", "0.01"),
            ("--num-steps", "10"),
            ("--precision", "bfloat16"),
        ):
            self.assertEqual(command[command.index(flag) + 1], expected)
        self.assertNotIn("--local-score-probe", command)
        self.assertEqual(command[-1], "/output/server_metadata.json")
        args.local_score_probe = True
        self.assertIn("--local-score-probe", evaluate.server_command(args, Path("/output")))

    def test_worker_environment_includes_repo_root_in_both_environments(self):
        args = SimpleNamespace(gpu_id="0", sim_root=Path("/sim"))
        for model in (True, False):
            env = evaluate.environment(args, model=model)
            self.assertIn(str(evaluate.REPO), env["PYTHONPATH"].split(":"))
            self.assertEqual(env["PYTHONDONTWRITEBYTECODE"], "1")

    def test_rollout_rejects_legacy_method_metadata(self):
        metadata = {
            "schema": POLICY_SCHEMA,
            "task": "poker-draw",
            "method": "tac_forward_guidance",
            "cpm_score_time": 0.0,
            "control_hz": 30,
            "prediction_horizon": 30,
            "execute_steps": 30,
            "model_action_dim": 32,
            "action_dim": 27,
            "model_input_cameras": ["head", "right_wrist"],
            "image_shape_hwc": [240, 320, 3],
            "tactile_shape": [5, 3, 7, 5],
            "tactile_channels": ["normal", "tangent0", "tangent1"],
            "finger_order": ["thumb", "index", "middle", "ring", "pinky"],
            "tactile_unit": "N",
            "tactile_target_geometry": "card_core_geom",
            "tactile_clipping": False,
            "joint_names": [str(i) for i in range(27)],
            "base_source": "/checkpoint/20000",
        }
        rollout.validate_metadata(metadata)
        with self.assertRaises(RuntimeError):
            rollout.validate_metadata({**metadata, "schema": "openpi_poker_touchguide_policy_v1"})
        with self.assertRaises(RuntimeError):
            rollout.validate_metadata({**metadata, "cpm_score_time": 0.3})

    def test_actual_shared_policy_infer_uses_forward_sampler_and_public_labels(self):
        sampler, prefix, _, _ = fixture(size=1, cpm=ObservationCPM())
        policy = object.__new__(PokerForwardGuidancePolicy)
        policy.guided = sampler
        policy.baseline = TouchGuideSampler(TinyBase(), sampler.cpm, GuidanceConfig(scale=0), precision="float32")
        policy.device = torch.device("cpu")
        policy.instruction = "poker"
        stats = {key: {"q01": [-1.0] * 27, "q99": [1.0] * 27} for key in ("state", "actions")}
        policy.loaded = SimpleNamespace(coordinates=ActionCoordinates(stats))
        obs = SimpleNamespace(
            state=prefix.state,
            images={"base_0_rgb": torch.zeros(1, 3, 224, 224), "right_wrist_0_rgb": torch.zeros(1, 3, 224, 224)},
            tokenized_prompt=torch.ones(1, 200, dtype=torch.long),
            tokenized_prompt_mask=torch.ones(1, 200, dtype=torch.bool),
        )
        policy.preprocessor = SimpleNamespace(prepare=lambda request, prompt: (obs, torch.zeros(1, 27)))
        request = {
            "variant": FORWARD_VARIANT,
            "episode_seed": 1,
            "request_index": 0,
            "tactile_map_raw": np.zeros((5, 3, 7, 5), dtype=np.float32),
            "return_trace": True,
            "compare_base": True,
        }
        with patch("openpi.touchguide.policy.prepare_pi05_prefix", return_value=prefix):
            response = policy.infer(request)
            baseline = policy.infer({**request, "variant": "pi05"})
        self.assertEqual(response["actions"].shape, (30, 27))
        self.assertEqual(response["diagnostics"]["variant"], FORWARD_VARIANT)
        self.assertTrue(response["diagnostics"]["base_comparison"])
        self.assertEqual(response["audit"]["trajectory"].shape, (11, 30, 32))
        self.assertEqual(response["audit"]["guidance_updates"][..., 27:].max(), 0)
        self.assertEqual(baseline["diagnostics"]["variant"], "pi05")
        self.assertEqual(request["variant"], FORWARD_VARIANT)
        self.assertTrue(np.array_equal(response["base_actions"], baseline["actions"]))


if __name__ == "__main__":
    unittest.main()
