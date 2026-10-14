"""Reuse trusted model loading and the existing poker observation/action contract."""

from dataclasses import asdict
from dataclasses import replace

from openpi.touchguide.inference import load_touchguide
from openpi.touchguide.policy import PokerTouchGuidePolicy
from openpi.touchguide.sampler import GuidanceConfig
from tac_forward_guidance.config import ForwardGuidanceConfig
from tac_forward_guidance.sampler import ForwardGuidanceSampler

FORWARD_VARIANT = "pi05+tac_forward_guidance"
VARIANTS = ("pi05", FORWARD_VARIANT)
POLICY_SCHEMA = "openpi_poker_tac_forward_guidance_policy_v1"


def load_forward_guidance(*, guidance=None, **kwargs):
    config = ForwardGuidanceConfig() if guidance is None else guidance
    loaded = load_touchguide(guidance=GuidanceConfig(config.num_steps, config.guidance_steps, 0.0), **kwargs)
    # This v1 is specifically for the user's original visual/tactile CPM. A
    # state-conditioned checkpoint needs a separately verified state adapter.
    if getattr(loaded.sampler.cpm, "uses_state", False):
        raise ValueError("Forward v1 uses the original visual/tactile CPM, not a state-conditioned checkpoint")
    sampler = ForwardGuidanceSampler(
        loaded.sampler.base, loaded.sampler.cpm, config, precision=loaded.sampler.precision
    )
    metadata = {
        **loaded.metadata,
        "method": "tac_forward_guidance",
        "cpm_score_time": 0.0,
        "gradient_mode": config.gradient_mode,
        "time_weight": "t/(1-t)",
        "clean_estimator": "x_t - t*v_theta(x_t,t,o)",
        "guidance": asdict(config),
        "training_performed": False,
    }
    return replace(loaded, sampler=sampler, metadata=metadata)


class PokerForwardGuidancePolicy(PokerTouchGuidePolicy):
    """Thin protocol adapter; preprocessing, seed rules and output coordinates stay shared.

    The parent uses its internal guided label solely to select loaded.sampler.
    Translate that label at the boundary, never in saved public diagnostics.
    """

    def __init__(self, loaded, *, tokenizer_model):
        super().__init__(loaded, tokenizer_model=tokenizer_model)
        self.metadata.update(schema=POLICY_SCHEMA, variants=list(VARIANTS), method="tac_forward_guidance")

    def infer(self, request):
        variant = request.get("variant")
        if variant not in VARIANTS:
            raise ValueError(f"Select one of {VARIANTS}")
        translated = dict(request)
        translated["variant"] = "pi05+touchguidance" if variant == FORWARD_VARIANT else "pi05"
        result = super().infer(translated)
        result["diagnostics"].update(
            variant=variant,
            method="tac_forward_guidance" if variant == FORWARD_VARIANT else "pi05",
            cpm_score_time=0.0 if variant == FORWARD_VARIANT else None,
            gradient_mode=self.guided.config.gradient_mode if variant == FORWARD_VARIANT else None,
        )
        return result
