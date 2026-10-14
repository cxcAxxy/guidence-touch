"""Numerical contracts needed before guiding an actual robot policy."""

# ruff: noqa: PT009, PT027 -- runnable with stdlib unittest in the inference environment

from types import SimpleNamespace
import unittest

import torch

from openpi.touchguide.inference import ActionCoordinates
from openpi.touchguide.sampler import GuidanceConfig
from openpi.touchguide.sampler import PrefixCache
from openpi.touchguide.sampler import TouchGuideSampler


class TinyBase(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.1), requires_grad=False)
        self.config = SimpleNamespace(action_horizon=30, action_dim=32)
        self.eval()

    def denoise_step(self, state, pad, cache, actions, time):
        return self.weight * actions + time[:, None, None] * 0.01


class TinyBF16Base(TinyBase):
    def denoise_step(self, state, pad, cache, actions, time):
        return super().denoise_step(state, pad, cache, actions, time).to(torch.bfloat16)


class TinyCPM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.target = torch.nn.Parameter(torch.tensor(0.3), requires_grad=False)
        self.config = SimpleNamespace(action_horizon=30, action_dim=27)
        self.eval()

    def paired_scores(self, observation, actions, time):
        return -(actions - self.target).square().mean(dim=(1, 2))


class TinyFlatCPM(TinyCPM):
    def paired_scores(self, observation, actions, time):
        # Remain differentiable while returning an exactly zero action gradient.
        return actions.sum(dim=(1, 2)) * 0.0


def fixture(scale=1.0, size=2):
    sampler = TouchGuideSampler(TinyBase(), TinyCPM(), GuidanceConfig(scale=scale), precision="float32")
    prefix = PrefixCache(torch.zeros(size, 32), torch.ones(size, 1, dtype=torch.bool), None)
    noise = torch.linspace(-2, 2, size * 30 * 32).reshape(size, 30, 32)
    return sampler, prefix, noise


class SamplerTest(unittest.TestCase):
    def test_zero_scale_is_exact_euler_and_skips_cpm(self):
        sampler, prefix, noise = fixture(scale=0)

        def unexpected(*args):
            raise AssertionError("Disabled guidance must not evaluate CPM")

        sampler.cpm.paired_scores = unexpected
        actual = sampler.sample_cached(prefix, None, noise).actions
        expected = noise.clone()
        time = torch.tensor(1.0)
        dt = torch.tensor(-0.1)
        while time >= -dt / 2:
            expected = expected + dt * sampler.base.denoise_step(prefix.state, None, None, expected, time.expand(2))
            time += dt
        self.assertTrue(torch.equal(actual, expected))

    def test_enabled_zero_gradient_with_bf16_velocity_is_exact_stock_euler(self):
        sampler, prefix, noise = fixture()
        sampler.base, sampler.cpm = TinyBF16Base(), TinyFlatCPM()
        disabled = TouchGuideSampler(sampler.base, sampler.cpm, GuidanceConfig(scale=0), precision="float32")
        guided = sampler.sample_cached(prefix, torch.ones(2, 1), noise, record_trace=True)
        stock = disabled.sample_cached(prefix, None, noise, record_trace=True)
        self.assertEqual(sum(row["guided"] for row in guided.diagnostics), 3)
        self.assertEqual(torch.count_nonzero(guided.guidance_updates), 0)
        self.assertTrue(torch.equal(guided.trajectory, stock.trajectory))
        self.assertTrue(torch.equal(guided.actions, stock.actions))

    def test_bf16_velocity_keeps_zero_guidance_padding_exact(self):
        sampler, prefix, noise = fixture()
        sampler.base = TinyBF16Base()
        disabled = TouchGuideSampler(sampler.base, sampler.cpm, GuidanceConfig(scale=0), precision="float32")
        guided = sampler.sample_cached(prefix, torch.ones(2, 1), noise, record_trace=True)
        stock = disabled.sample_cached(prefix, None, noise, record_trace=True)
        self.assertEqual(guided.actions.dtype, torch.float32)
        self.assertEqual(guided.guidance_updates.dtype, torch.float32)
        self.assertGreater(torch.count_nonzero(guided.guidance_updates[..., :27]), 0)
        # This toy base treats each coordinate independently, so a padded
        # difference can only come from changed arithmetic, not model coupling.
        self.assertEqual(torch.count_nonzero(guided.guidance_updates[..., 27:]), 0)
        self.assertTrue(torch.equal(guided.trajectory[..., 27:], stock.trajectory[..., 27:]))
        self.assertFalse(torch.equal(guided.actions[..., :27], stock.actions[..., :27]))

    def test_late_steps_increase_score_and_never_guide_padded_coordinates(self):
        sampler, prefix, noise = fixture()
        guided = sampler.sample_cached(prefix, torch.ones(2, 1), noise, record_trace=True)
        disabled, _, _ = fixture(scale=0)
        base = disabled.sample_cached(prefix, None, noise, record_trace=True)
        self.assertEqual([row["step"] for row in guided.diagnostics if row["guided"]], [7, 8, 9])
        self.assertTrue(torch.equal(guided.trajectory[:8], base.trajectory[:8]))
        self.assertEqual(torch.count_nonzero(guided.guidance_updates[..., 27:]), 0)
        self.assertTrue(torch.equal(guided.actions[..., 27:], base.actions[..., 27:]))
        self.assertGreater(torch.count_nonzero(guided.actions[..., 27:]), 0)  # No forced padding reset.
        self.assertFalse(torch.equal(guided.actions[..., :27], base.actions[..., :27]))
        for row in guided.diagnostics[7:]:
            self.assertTrue(
                all(
                    after > before
                    for before, after in zip(row["score_before"], row["score_after_guidance_only"], strict=True)
                )
            )
        self.assertFalse(guided.actions.requires_grad)
        self.assertTrue(
            all(parameter.grad is None for model in (sampler.base, sampler.cpm) for parameter in model.parameters())
        )

    def test_gradient_is_batch_independent_and_works_inside_no_grad(self):
        sampler, _, noise = fixture(size=1)
        with torch.no_grad():
            one, _ = sampler.score_gradient(torch.ones(1, 1), noise, torch.tensor(0.3))
            many, _ = sampler.score_gradient(torch.ones(4, 1), noise.repeat(4, 1, 1), torch.tensor(0.3))
        self.assertTrue(torch.equal(many, one.repeat(4, 1, 1)))
        torch.testing.assert_close(one[..., :27], -2 * (noise[..., :27] - 0.3) / (30 * 27))
        with torch.inference_mode(), self.assertRaisesRegex(RuntimeError, "action gradients"):
            sampler.score_gradient(torch.ones(1, 1), noise, torch.tensor(0.3))

    def test_invalid_or_singular_guidance_is_rejected(self):
        for kwargs in (
            {"scale": -1},
            {"scale": float("nan")},
            {"num_steps": 0},
            {"guidance_steps": 10},
            {"guidance_steps": -1},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                GuidanceConfig(**kwargs)

    def test_normalization_restores_arm_deltas_but_keeps_hand_absolute(self):
        stats = {key: {"q01": [-2.0] * 27, "q99": [2.0] * 27} for key in ("state", "actions")}
        coordinates = ActionCoordinates(stats)
        state = torch.ones(2, 27) * 0.7
        normalized = torch.ones(2, 30, 32) * 0.25
        normalized[..., 27:] = 99.0  # These dimensions must not become actuator commands.
        actual = coordinates.action_targets(normalized, state)
        target = 1.25 / 2 * (4 + 1e-6) - 2
        torch.testing.assert_close(actual[..., :7], torch.full((2, 30, 7), target + 0.7))
        torch.testing.assert_close(actual[..., 7:], torch.full((2, 30, 20), target))
        self.assertEqual(actual.shape, (2, 30, 27))


if __name__ == "__main__":
    unittest.main()
