"""CPU numerical and protocol contracts; no checkpoints, simulator, or GPU needed."""

# ruff: noqa: PT009, PT027 -- stdlib unittest also runs in the inference environment

from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from guidance_common.scorer import CPMScorer
from openpi.touchguide.policy import PokerTouchGuidePolicy
from openpi.touchguide.sampler import GuidanceConfig
from openpi.touchguide.sampler import PrefixCache
from openpi.touchguide.sampler import TouchGuideSampler
from openpi.touchguide.sampler_test import TinyBase
from openpi.touchguide.sampler_test import TinyBF16Base
from openpi.touchguide.sampler_test import TinyCPM
from openpi.touchguide.sampler_test import TinyFlatCPM
from tac_forward_guidance.config import ForwardGuidanceConfig
from tac_forward_guidance.runtime import FORWARD_VARIANT
from tac_forward_guidance.runtime import PokerForwardGuidancePolicy
from tac_forward_guidance.sampler import ForwardGuidanceSampler


class RecordingCPM(TinyCPM):
    def __init__(self):
        super().__init__()
        self.times = []

    def paired_scores(self, observation, actions, time):
        self.times.append(float(time))
        return super().paired_scores(observation, actions, time)


class CoupledBase(TinyBase):
    def denoise_step(self, state, pad, cache, actions, time):
        velocity = super().denoise_step(state, pad, cache, actions, time)
        coupling = actions[..., 27:28].expand(-1, -1, 27) * 0.5
        return torch.cat([velocity[..., :27] + coupling, velocity[..., 27:]], dim=-1)


class DetachedBase(TinyBase):
    def denoise_step(self, state, pad, cache, actions, time):
        return super().denoise_step(state, pad, cache, actions, time).detach()


def fixture(*, size=2, base=None, cpm=None, **config):
    sampler = ForwardGuidanceSampler(
        TinyBase() if base is None else base,
        TinyCPM() if cpm is None else cpm,
        ForwardGuidanceConfig(**config),
        precision="float32",
    )
    prefix = PrefixCache(torch.zeros(size, 32), torch.ones(size, 1, dtype=torch.bool), None)
    noise = torch.linspace(-2, 2, size * 30 * 32).reshape(size, 30, 32)
    return sampler, prefix, noise, torch.ones(size, 1)


class ForwardSamplerTest(unittest.TestCase):
    def test_full_mode_rejects_a_severed_action_expert_graph(self):
        sampler, prefix, noise, embedding = fixture(base=DetachedBase())
        with self.assertRaisesRegex(RuntimeError, "severed its input graph"):
            sampler.forward_gradient(prefix, embedding, noise, 0.3)

    def test_full_gradient_matches_analytic_chain_rule(self):
        cpm = RecordingCPM()
        sampler, prefix, noise, embedding = fixture(cpm=cpm)
        result = sampler.forward_gradient(prefix, embedding, noise, 0.3)
        clean = noise - 0.3 * (0.1 * noise + 0.003)
        expected_clean = -2 * (clean[..., :27] - 0.3) / (30 * 27)
        torch.testing.assert_close(result.clean_actions, clean)
        torch.testing.assert_close(result.clean_gradient[..., :27], expected_clean)
        torch.testing.assert_close(result.gradient[..., :27], (1 - 0.3 * 0.1) * expected_clean)
        self.assertFalse(torch.equal(result.gradient[..., :27], expected_clean))
        self.assertEqual(cpm.times, [0.0])

    def test_stop_velocity_is_an_explicit_distinct_ablation(self):
        full, prefix, noise, embedding = fixture()
        stopped, _, _, _ = fixture(gradient_mode="stop_velocity")
        a = full.forward_gradient(prefix, embedding, noise, 0.3)
        b = stopped.forward_gradient(prefix, embedding, noise, 0.3)
        torch.testing.assert_close(a.clean_actions, b.clean_actions)
        torch.testing.assert_close(a.scores, b.scores)
        torch.testing.assert_close(b.gradient, b.clean_gradient)
        self.assertFalse(torch.equal(a.gradient, b.gradient))

    def test_padding_is_masked_after_the_full_jacobian(self):
        sampler, prefix, noise, embedding = fixture(base=CoupledBase())
        result = sampler.forward_gradient(prefix, embedding, noise, 0.3)
        self.assertGreater(result.raw_padded_gradient_max_abs, 0)
        self.assertEqual(torch.count_nonzero(result.gradient[..., 27:]), 0)
        guided = sampler.sample_cached(prefix, embedding, noise, record_trace=True)
        self.assertEqual(torch.count_nonzero(guided.guidance_updates[..., 27:]), 0)

    def test_shared_scorer_preserves_upstream_action_graph(self):
        scorer = CPMScorer(TinyCPM())
        x = torch.ones(1, 30, 27, requires_grad=True)
        scores = scorer.score(torch.ones(1, 1), x * 2, 0.0)
        (gradient,) = torch.autograd.grad(scores.sum(), x)
        torch.testing.assert_close(gradient, torch.full_like(x, -4 * 1.7 / (30 * 27)))

    def test_zero_scale_and_zero_guidance_steps_skip_scoring_exactly(self):
        for config in ({"scale": 0.0}, {"guidance_steps": 0}):
            sampler, prefix, noise, _ = fixture(**config)
            sampler.cpm.paired_scores = lambda *a: self.fail("Disabled guidance called CPM")
            actual = sampler.sample_cached(prefix, None, noise, record_trace=True)
            baseline = TouchGuideSampler(sampler.base, sampler.cpm, GuidanceConfig(scale=0), precision="float32")
            expected = baseline.sample_cached(prefix, None, noise, record_trace=True)
            self.assertTrue(torch.equal(actual.trajectory, expected.trajectory))

    def test_bf16_zero_gradient_is_exact_stock_even_on_guided_steps(self):
        sampler, prefix, noise, embedding = fixture(base=TinyBF16Base(), cpm=TinyFlatCPM())
        actual = sampler.sample_cached(prefix, embedding, noise, record_trace=True)
        baseline = TouchGuideSampler(sampler.base, sampler.cpm, GuidanceConfig(scale=0), precision="float32")
        expected = baseline.sample_cached(prefix, None, noise, record_trace=True)
        self.assertEqual(torch.count_nonzero(actual.guidance_updates), 0)
        self.assertTrue(torch.equal(actual.trajectory, expected.trajectory))

    def test_late_steps_frozen_parameters_and_actual_local_probe(self):
        sampler, prefix, noise, embedding = fixture(local_score_probe=True)
        guided = sampler.sample_cached(prefix, embedding, noise, record_trace=True)
        baseline = TouchGuideSampler(sampler.base, sampler.cpm, GuidanceConfig(scale=0), precision="float32")
        stock = baseline.sample_cached(prefix, None, noise, record_trace=True)
        self.assertTrue(torch.equal(guided.trajectory[:8], stock.trajectory[:8]))
        active = [row for row in guided.diagnostics if row["guided"]]
        self.assertEqual([row["step"] for row in active], [7, 8, 9])
        for row in active:
            self.assertEqual(row["cpm_score_time"], 0)
            self.assertGreater(row["jacobian_contribution_rms"], 0)
            self.assertTrue(
                all(a > b for a, b in zip(row["score_after_guidance_only"], row["score_before"], strict=True))
            )
        self.assertFalse(guided.actions.requires_grad)
        self.assertTrue(
            all(not p.requires_grad and p.grad is None for m in (sampler.base, sampler.cpm) for p in m.parameters())
        )

    def test_no_probe_in_production_diagnostics(self):
        sampler, prefix, noise, embedding = fixture()
        result = sampler.sample_cached(prefix, embedding, noise)
        self.assertTrue(all("score_after_guidance_only" not in row for row in result.diagnostics))

    def test_batch_independence_and_outer_no_grad(self):
        sampler, prefix, noise, embedding = fixture(size=1)
        with torch.no_grad():
            one = sampler.forward_gradient(prefix, embedding, noise, 0.3)
            many_prefix = replace(prefix, state=prefix.state.repeat(4, 1), pad_masks=prefix.pad_masks.repeat(4, 1))
            many = sampler.forward_gradient(many_prefix, embedding.repeat(4, 1), noise.repeat(4, 1, 1), 0.3)
        torch.testing.assert_close(many.gradient, one.gradient.repeat(4, 1, 1), rtol=0, atol=0)

    def test_update_cap_is_per_sample(self):
        sampler, _, _, _ = fixture(max_update_rms=0.01)
        shift = torch.zeros(2, 30, 32)
        shift[0, :, :27] = 1
        shift[1, :, :27] = 0.001
        capped, factors = sampler.cap_update(shift)
        torch.testing.assert_close(factors, torch.tensor([0.01, 1.0]))
        torch.testing.assert_close(capped[..., :27].square().mean(dim=(1, 2)).sqrt(), torch.tensor([0.01, 0.001]))

    def test_cap_is_applied_to_recorded_sampling_updates(self):
        sampler, prefix, noise, embedding = fixture(scale=100000, max_update_rms=0.001)
        guided = sampler.sample_cached(prefix, embedding, noise, record_trace=True)
        self.assertTrue(all(row["capped_samples"] == 2 for row in guided.diagnostics[7:]))
        self.assertLessEqual(float(guided.guidance_updates[..., :27].square().mean(dim=(2, 3)).sqrt().max()), 0.001001)

    def test_inference_mode_is_rejected(self):
        sampler, prefix, noise, embedding = fixture()
        with torch.inference_mode(), self.assertRaisesRegex(RuntimeError, "input gradients"):
            sampler.sample_cached(prefix, embedding, noise)

    def test_invalid_times_and_configuration_are_rejected(self):
        sampler, prefix, noise, embedding = fixture()
        for t in (0.0, 1.0, float("nan")):
            with self.assertRaises(ValueError):
                sampler.forward_gradient(prefix, embedding, noise, t)
        for config in (
            {"gradient_mode": "wrong"},
            {"max_update_rms": 0},
            {"max_update_rms": -1},
            {"max_update_rms": float("nan")},
            {"scale": -1},
            {"guidance_steps": 10},
        ):
            with self.subTest(config=config), self.assertRaises(ValueError):
                ForwardGuidanceConfig(**config)

    def test_nonfinite_and_wrong_shape_inputs_are_rejected(self):
        sampler, prefix, noise, embedding = fixture()
        for bad in (noise.double(), noise[..., :27], noise * float("nan")):
            with self.assertRaises(ValueError):
                sampler.sample_cached(prefix, embedding, bad)

    def test_protocol_adapter_preserves_public_identity_and_request(self):
        policy = object.__new__(PokerForwardGuidancePolicy)
        policy.guided = SimpleNamespace(config=ForwardGuidanceConfig())
        request = {"variant": FORWARD_VARIANT}
        with patch.object(PokerTouchGuidePolicy, "infer", return_value={"diagnostics": {}}) as shared:
            result = policy.infer(request)
        self.assertEqual(request["variant"], FORWARD_VARIANT)
        self.assertEqual(shared.call_args.args[0]["variant"], "pi05+touchguidance")
        self.assertEqual(result["diagnostics"]["variant"], FORWARD_VARIANT)
        self.assertEqual(result["diagnostics"]["gradient_mode"], "full")


if __name__ == "__main__":
    unittest.main()
