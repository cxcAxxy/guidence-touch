"""Small CPU checks for CPM matching losses; no model or dataset is loaded."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from openpi.touchguide import losses as loss_module
from openpi.touchguide.losses import cpm_contrastive_loss
from openpi.touchguide.losses import symmetric_infonce


def _reject(function, *args) -> None:
    try:
        function(*args)
    except ValueError:
        return
    raise AssertionError("Invalid loss input was accepted")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    logits = torch.tensor([[1.7, -0.4, 0.9], [0.2, 2.6, -1.1], [-0.7, 1.3, 0.8]], dtype=torch.float64)
    logits.requires_grad_()
    targets = torch.arange(3)
    metrics = symmetric_infonce(logits, targets)
    # Independent log-normalizer formula, evaluated in both matching directions.
    diagonal = logits.diagonal()
    row_formula = (torch.logsumexp(logits, dim=1) - diagonal).mean()
    column_formula = (torch.logsumexp(logits, dim=0) - diagonal).mean()
    torch.testing.assert_close(metrics["loss_obs_to_action"], row_formula)
    torch.testing.assert_close(metrics["loss_action_to_obs"], column_formula)
    torch.testing.assert_close(metrics["loss"], (row_formula + column_formula) / 2)
    assert not torch.isclose(row_formula, column_formula)
    torch.testing.assert_close(symmetric_infonce(logits)["loss"], metrics["loss"])
    metrics["loss"].backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert torch.all(logits.grad.diagonal() < 0)
    assert torch.all(logits.grad[~torch.eye(3, dtype=torch.bool)] > 0)

    uniform = symmetric_infonce(torch.zeros(64, 64, dtype=torch.float64))
    torch.testing.assert_close(uniform["loss"], torch.tensor(math.log(64), dtype=torch.float64))
    correct = torch.eye(4, dtype=torch.float64) * 5
    matched = symmetric_infonce(correct)
    mismatched = symmetric_infonce(correct.roll(shifts=1, dims=1))
    assert matched["loss"] < mismatched["loss"]
    assert matched["accuracy_obs_to_action"] == matched["accuracy_action_to_obs"] == 1
    assert mismatched["accuracy_obs_to_action"] == mismatched["accuracy_action_to_obs"] == 0

    scores = torch.eye(4, dtype=torch.float64) - 0.2
    output = {
        "logits": scores / 0.2,
        "cosine_scores": scores,
        "observation_embedding": torch.eye(4),
        "action_embedding": torch.eye(4),
        "temperature": torch.tensor(0.2),
    }
    batch = {
        "targets": torch.arange(4),
        "positive_mask": torch.eye(4, dtype=torch.bool),
        "negative_mask": ~torch.eye(4, dtype=torch.bool),
    }
    summary = cpm_contrastive_loss(output, batch)
    torch.testing.assert_close(summary["positive_score"], torch.tensor(0.8, dtype=torch.float64))
    torch.testing.assert_close(summary["negative_score"], torch.tensor(-0.2, dtype=torch.float64))
    torch.testing.assert_close(summary["score_gap"], torch.tensor(1.0, dtype=torch.float64))
    assert all(isinstance(value, torch.Tensor) and value.ndim == 0 for value in summary.values())

    _reject(symmetric_infonce, torch.zeros(1, 1))
    _reject(symmetric_infonce, torch.zeros(2, 3))
    _reject(symmetric_infonce, torch.zeros(2, 2, dtype=torch.long))
    _reject(symmetric_infonce, torch.tensor([[0.0, math.nan], [0.0, 1.0]]))
    _reject(symmetric_infonce, torch.tensor([[0.0, math.inf], [0.0, 1.0]]))
    _reject(symmetric_infonce, torch.zeros(2, 2), torch.tensor([1, 0]))
    _reject(symmetric_infonce, torch.zeros(2, 2), torch.tensor([0.0, 1.0]))
    _reject(symmetric_infonce, torch.zeros(2, 2), torch.tensor([0]))
    _reject(cpm_contrastive_loss, output, {**batch, "positive_mask": torch.ones(4, 4, dtype=torch.bool)})
    _reject(cpm_contrastive_loss, output, {**batch, "negative_mask": batch["positive_mask"]})
    _reject(cpm_contrastive_loss, output, {**batch, "positive_mask": torch.eye(4)})
    _reject(cpm_contrastive_loss, output, {**batch, "targets": torch.arange(4).roll(1)})
    _reject(cpm_contrastive_loss, output, {**batch, "targets": None})
    _reject(cpm_contrastive_loss, {**output, "cosine_scores": torch.zeros(3, 3)}, batch)
    _reject(cpm_contrastive_loss, {**output, "temperature": torch.tensor(0.0)}, batch)
    _reject(cpm_contrastive_loss, {**output, "temperature": torch.tensor(math.nan)}, batch)
    _reject(cpm_contrastive_loss, {**output, "action_embedding": torch.zeros(4, 3)}, batch)
    result = {
        "status": "passed",
        "module_path": str(Path(loss_module.__file__).resolve()),
        "two_direction_formula_checked": True,
        "asymmetric_logits_checked": True,
        "uniform_random_baseline_log_b": float(uniform["loss"]),
        "matched_loss": float(matched["loss"]),
        "mismatched_loss": float(mismatched["loss"]),
        "finite_effective_logit_gradients": True,
        "identity_targets_and_masks_checked": True,
        "invalid_inputs_rejected": True,
        "label_semantics": "identity_matching_not_physical_success",
        "model_loaded": False,
        "training_started": False,
    }
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
