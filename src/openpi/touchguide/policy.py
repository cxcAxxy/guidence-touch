"""Simulation-facing pi0.5 policy with optional CPM flow guidance."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import numpy as np
from openpi_client import image_tools
import torch

from openpi.touchguide.inference import LoadedTouchGuide
from openpi.touchguide.sampler import GuidanceConfig
from openpi.touchguide.sampler import TouchGuideSampler
from openpi.touchguide.sampler import prepare_pi05_prefix

VARIANTS = ("pi05", "pi05+touchguidance")
RIGHT_JOINT_NAMES = (
    *(f"right_arm_joint{index}" for index in range(1, 8)),
    "hand_r_thumb_joint1",
    "hand_r_thumb_joint2",
    "hand_r_thumb_joint3",
    "hand_r_thumb_joint5",
    *(f"hand_r_{finger}_joint{index}" for finger in ("index", "middle", "ring", "pinky") for index in range(1, 5)),
)


def noise_seed(episode_seed: int, request_index: int) -> int:
    """Identical Gaussian-noise rules for both variants, independent of server history."""
    for value in (episode_seed, request_index):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError("Episode seed and zero-based request index must be nonnegative integers")
    return int(np.random.SeedSequence([episode_seed, request_index]).generate_state(1, dtype=np.uint64)[0])


class ObservationPreprocessor:
    """Match KaiHandInputs/Normalize/Resize/Tokenize/Pad, including NumPy quantile precision."""

    def __init__(self, stats: dict, tokenizer: Any, device: str | torch.device):
        self.low = np.asarray(stats["state"]["q01"], dtype=np.float64)
        self.high = np.asarray(stats["state"]["q99"], dtype=np.float64)
        self.tokenizer = tokenizer
        self.device = torch.device(device)

    def prepare(self, request: dict, prompt: str):
        from openpi.models.model import Observation  # noqa: PLC0415

        raw = np.array(request["state"], dtype=np.float32, copy=True)
        if raw.shape != (27,) or not np.isfinite(raw).all():
            raise ValueError("Use 27 finite right-side measured joint positions")
        state = (raw - self.low) / (self.high - self.low + 1e-6) * 2.0 - 1.0
        tokens, token_mask = self.tokenizer.tokenize(prompt, state)
        # Tokenize the unpadded 27-D state, exactly as the original pi0.5 pipeline.
        state = np.pad(state.astype(np.float32), (0, 5))
        images = {}
        for name, key in (("base_0_rgb", "head_image"), ("right_wrist_0_rgb", "right_wrist_image")):
            image = np.asarray(request[key])
            if image.shape != (240, 320, 3) or image.dtype != np.uint8:
                raise ValueError(f"Expected {key} uint8 RGB [240,320,3]")
            resized = image_tools.resize_with_pad(image, 224, 224)
            pixels = resized.astype(np.float32) / np.float32(127.5) - np.float32(1.0)
            images[name] = torch.from_numpy(pixels).permute(2, 0, 1)[None].contiguous().to(self.device)
        # Original KaiHandInputs uses a black uint8 dummy left view, masked out.
        left = torch.full_like(images["base_0_rgb"], -1.0)
        observation = Observation(
            images={
                "base_0_rgb": images["base_0_rgb"],
                "left_wrist_0_rgb": left,
                "right_wrist_0_rgb": images["right_wrist_0_rgb"],
            },
            image_masks={
                key: torch.tensor([active], device=self.device)
                for key, active in (("base_0_rgb", True), ("left_wrist_0_rgb", False), ("right_wrist_0_rgb", True))
            },
            state=torch.from_numpy(state)[None].to(self.device),
            tokenized_prompt=torch.from_numpy(tokens.astype(np.int64))[None].to(self.device),
            tokenized_prompt_mask=torch.from_numpy(token_mask)[None].to(self.device),
        )
        return observation, torch.from_numpy(raw)[None].to(self.device)


class PokerTouchGuidePolicy:
    """One frozen runtime serves both variants and optional same-input base comparisons."""

    def __init__(self, loaded: LoadedTouchGuide, *, tokenizer_model: str | Path, execute_steps: int = 5):
        from openpi.models.tokenizer import PaligemmaTokenizer  # noqa: PLC0415

        if isinstance(execute_steps, bool) or not isinstance(execute_steps, int) or not 1 <= execute_steps <= 30:
            raise ValueError("execute_steps must be an integer in 1..30")
        self.loaded = loaded
        self.execute_steps = execute_steps
        self.guided = loaded.sampler
        config = self.guided.config
        self.baseline = TouchGuideSampler(
            self.guided.base,
            self.guided.cpm,
            GuidanceConfig(config.num_steps, config.guidance_steps, 0.0),
            precision=self.guided.precision,
        )
        self.device = next(self.guided.base.parameters()).device
        self.preprocessor = ObservationPreprocessor(
            loaded.coordinates.stats, PaligemmaTokenizer(200, model_path=tokenizer_model), self.device
        )
        root = Path(loaded.metadata["data_root"])
        manifest = json.loads((root / "meta/manifest.json").read_text())
        instruction_path = root / "meta/task_instruction.json"
        if instruction_path.exists():
            self.instruction = json.loads(instruction_path.read_text())["task"]
        else:
            tasks_path = Path(manifest["sources"]["pi05_root"]) / "meta/tasks.jsonl"
            tasks = [json.loads(line) for line in tasks_path.read_text().splitlines() if line.strip()]
            if len(tasks) != 1:
                raise ValueError("Expected one frozen poker-draw task instruction")
            self.instruction = tasks[0]["task"]
        if not isinstance(self.instruction, str) or not self.instruction.strip():
            raise ValueError("Task instruction must be nonempty")
        self.metadata = {
            **loaded.metadata,
            "schema": "openpi_poker_touchguide_policy_v1",
            "task": "poker-draw",
            "variants": list(VARIANTS),
            "prediction_horizon": 30,
            "execute_steps": execute_steps,
            "replan_period_sim_s": execute_steps / 30,
            "rtc_enabled": False,
            "control_hz": 30,
            "model_action_dim": 32,
            "action_dim": 27,
            "joint_names": list(RIGHT_JOINT_NAMES),
            "instruction": self.instruction,
            "guidance": asdict(config),
            "image_shape_hwc": [240, 320, 3],
            "model_input_cameras": ["head", "right_wrist"],
            "tactile_shape": [5, 3, 7, 5],
            "tactile_channels": ["normal", "tangent0", "tangent1"],
            "finger_order": ["thumb", "index", "middle", "ring", "pinky"],
            "tactile_unit": "N",
            "tactile_target_geometry": "card_core_geom",
            "tactile_clipping": False,
            "action_output": "absolute right-arm 7 plus absolute right-hand 20 joint targets in radians",
            "observation_timing": "synchronized RGB/state/tactile every executed prefix; simulation paused during inference",
            "noise_rule": "SeedSequence([episode_seed, zero_based_request_index]) -> torch Gaussian [1,30,32]",
            "tokenizer_model": str(Path(tokenizer_model).expanduser().resolve(strict=True)),
        }

    def infer(self, request: dict) -> dict:
        variant = request.get("variant")
        if variant not in VARIANTS:
            raise ValueError(f"Select one of {VARIANTS}")
        prompt = request.get("prompt", self.instruction)
        if prompt != self.instruction:
            raise ValueError("Use the frozen poker task instruction for this comparison")
        if request.get("execute_steps", self.execute_steps) != self.execute_steps:
            raise ValueError("Client execute_steps differs from the server execution contract")
        obs, state_raw = self.preprocessor.prepare(request, prompt)
        seed = noise_seed(request["episode_seed"], request["request_index"])
        generator = torch.Generator(device=self.device).manual_seed(seed)
        noise = torch.randn((1, 30, 32), generator=generator, device=self.device, dtype=torch.float32)
        sampler = self.guided if variant == "pi05+touchguidance" else self.baseline
        trace = bool(request.get("return_trace", True))
        compare = bool(request.get("compare_base", True)) and variant != "pi05"
        with torch.no_grad(), sampler._autocast(self.device):  # noqa: SLF001
            prefix = prepare_pi05_prefix(sampler.base, obs)
            embedding = None
            if sampler.config.enabled:
                maps = np.array(request["tactile_map_raw"], dtype=np.float32, copy=True)
                if maps.shape != (5, 3, 7, 5) or not np.isfinite(maps).all():
                    raise ValueError("Expected current finite right-hand forces [5,3,7,5], in newtons")
                raw_maps = torch.from_numpy(maps)[None].to(self.device)
                state_kwargs = (
                    {"state_raw": torch.as_tensor(state_raw, dtype=torch.float32, device=self.device)}
                    if getattr(sampler.cpm, "uses_state", False)
                    else {}
                )
                embedding = sampler.cpm.encode_observation(
                    obs.images["base_0_rgb"], obs.images["right_wrist_0_rgb"], raw_maps, **state_kwargs
                ).detach()
        result = sampler.sample_cached(prefix, embedding, noise, record_trace=trace)
        base_result = (
            self.baseline.sample_cached(prefix, None, noise, record_diagnostics=False, record_trace=trace)
            if compare
            else (result if variant == "pi05" else None)
        )
        targets = self.loaded.coordinates.action_targets(result.actions, state_raw)
        diagnostics = {
            "variant": variant,
            "noise_seed": seed,
            "guidance": asdict(sampler.config),
            "steps": list(result.diagnostics),
            "base_comparison": base_result is not None,
        }
        output = {"actions": targets[0].cpu().numpy(), "diagnostics": diagnostics}
        if base_result is not None:
            base_targets = self.loaded.coordinates.action_targets(base_result.actions, state_raw)
            delta = targets - base_targets
            diagnostics.update(
                arm_target_delta_rms_rad=float(delta[..., :7].square().mean().sqrt()),
                hand_target_delta_rms_rad=float(delta[..., 7:].square().mean().sqrt()),
                target_delta_max_abs_rad=float(delta.abs().max()),
            )
            output["base_actions"] = base_targets[0].cpu().numpy()
        if trace:
            output["audit"] = {
                "noise": noise[0].cpu().numpy(),
                "normalized_actions": result.actions[0].cpu().numpy(),
                "trajectory": result.trajectory[:, 0].cpu().numpy(),
                "guidance_updates": result.guidance_updates[:, 0].cpu().numpy(),
                "normalized_state": obs.state[0].cpu().numpy(),
                "tokenized_prompt": obs.tokenized_prompt[0].cpu().numpy(),
                "tokenized_prompt_mask": obs.tokenized_prompt_mask[0].cpu().numpy(),
            }
            if base_result is not None:
                output["audit"].update(
                    base_normalized_actions=base_result.actions[0].cpu().numpy(),
                    base_trajectory=base_result.trajectory[:, 0].cpu().numpy(),
                )
        return output
