"""CPU synthetic checkpoint regression, without CPM data or actual CPM training."""

from __future__ import annotations

from collections.abc import Callable
import copy
import hashlib
import json
from pathlib import Path
import random
import tempfile
from typing import Any
from unittest import mock
import zipfile

import numpy as np
import torch

from openpi.touchguide.checkpoint import CHECKPOINT_FORMAT
from openpi.touchguide.checkpoint import capture_rng_state
from openpi.touchguide.checkpoint import load_checkpoint
from openpi.touchguide.checkpoint import restore_rng_state
from openpi.touchguide.checkpoint import save_checkpoint


def _equal(left: Any, right: Any) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            _equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right)
        assert len(left) == len(right)
        for left_value, right_value in zip(left, right, strict=True):
            _equal(left_value, right_value)
    else:
        assert left == right


def _reject(action: Callable[[], Any], exception: type[Exception] = ValueError) -> None:
    try:
        action()
    except exception:
        return
    raise AssertionError(f"Expected {exception.__name__}")


def _fixture() -> tuple[torch.nn.Module, torch.optim.Optimizer, Any]:
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Tanh(), torch.nn.Linear(8, 3))
    model[0].requires_grad_(requires_grad=False)
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=0.002)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
    return model, optimizer, scheduler


def _draw() -> dict[str, Any]:
    return {"python": random.random(), "numpy": np.random.rand(5), "torch": torch.rand(5)}


def main() -> None:
    torch.set_num_threads(1)
    torch.manual_seed(17)
    np.random.seed(17)
    random.seed(17)
    model, optimizer, scheduler = _fixture()
    original_frozen = copy.deepcopy(model[0].state_dict())
    # One optimizer step on tiny synthetic tensors, solely to populate Adam state.
    loss = model(torch.randn(3, 4)).square().mean()
    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    _equal(original_frozen, model[0].state_dict())
    progress = {"epoch": 2, "batch_in_epoch": 7, "global_step": 19, "best_val": 0.125}
    contract = {"config_digest": "synthetic-test-digest", "total_epochs": 3, "batch_size": 64, "noise": [0, 0.4]}
    run = {"id": "synthetic-id", "name": "checkpoint-regression", "entity": "offline", "project": "unit-check"}
    source_model = copy.deepcopy(model.state_dict())
    source_optimizer = copy.deepcopy(optimizer.state_dict())
    source_scheduler = copy.deepcopy(scheduler.state_dict())
    with tempfile.TemporaryDirectory(prefix="cpm-checkpoint-regression-") as directory:
        path = Path(directory) / "checkpoint.pt"
        save_checkpoint(
            path, model=model, optimizer=optimizer, scheduler=scheduler, progress=progress, contract=contract, run=run
        )
        expected_rng = capture_rng_state()
        expected_draw = _draw()
        target_model, target_optimizer, target_scheduler = _fixture()
        before_load_rng = capture_rng_state()
        payload = load_checkpoint(
            path, model=target_model, optimizer=target_optimizer, scheduler=target_scheduler, expected_contract=contract
        )
        _equal(before_load_rng, capture_rng_state())
        _equal(payload["progress"], progress)
        _equal(payload["contract"], contract)
        _equal(payload["run"], run)
        _equal(payload["rng_state"], expected_rng)
        _equal(target_model.state_dict(), source_model)
        _equal(target_optimizer.state_dict(), source_optimizer)
        _equal(target_scheduler.state_dict(), source_scheduler)
        assert all(not parameter.requires_grad for parameter in target_model[0].parameters())
        assert target_optimizer.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]
        for state in target_optimizer.state.values():
            assert state["step"].device.type == "cpu"
            assert state["exp_avg"].device.type == "cpu"
            assert state["exp_avg_sq"].device.type == "cpu"
        restore_rng_state(payload["rng_state"])
        _equal(_draw(), expected_draw)
        _draw()
        restore_rng_state(payload["rng_state"])
        _equal(_draw(), expected_draw)
        # Mismatches are rejected before touching model or optimizer state.
        before_model = copy.deepcopy(target_model.state_dict())
        before_optimizer = copy.deepcopy(target_optimizer.state_dict())
        _reject(
            lambda: load_checkpoint(
                path,
                model=target_model,
                optimizer=target_optimizer,
                scheduler=target_scheduler,
                expected_contract={**contract, "total_epochs": 4},
            )
        )
        _equal(target_model.state_dict(), before_model)
        _equal(target_optimizer.state_dict(), before_optimizer)
        _reject(
            lambda: load_checkpoint(
                path,
                model=target_model,
                optimizer=target_optimizer,
                scheduler=target_scheduler,
                expected_contract={**contract, "batch_size": 64.0},
            )
        )
        wrong_format = Path(directory) / "wrong-format.pt"
        bad_payload = {**payload, "format": "some_other_training_format"}
        torch.save(bad_payload, wrong_format)
        with zipfile.ZipFile(wrong_format, "a") as archive:
            root = next(name for name in archive.namelist() if name.endswith("/data.pkl")).rsplit("/", 1)[0]
            archive.writestr(f"{root}/touchguide_checkpoint_format", CHECKPOINT_FORMAT)
        _reject(
            lambda: load_checkpoint(
                wrong_format,
                model=target_model,
                optimizer=target_optimizer,
                scheduler=target_scheduler,
                expected_contract=contract,
            )
        )
        unrelated = Path(directory) / "existing-artifact.txt"
        unrelated.write_text("keep this artifact\n")
        _reject(
            lambda: save_checkpoint(
                unrelated, model=model, optimizer=optimizer, scheduler=scheduler, progress=progress, contract=contract
            )
        )
        assert unrelated.read_text() == "keep this artifact\n"
        link = Path(directory) / "checkpoint-link.pt"
        link.symlink_to(path)
        _reject(
            lambda: save_checkpoint(
                link, model=model, optimizer=optimizer, scheduler=scheduler, progress=progress, contract=contract
            )
        )
        # Atomic failure leaves the existing checkpoint and no temporary artifacts.
        original_digest = hashlib.sha256(path.read_bytes()).hexdigest()
        with mock.patch("openpi.touchguide.checkpoint.torch.save", side_effect=RuntimeError("simulated write failure")):
            _reject(
                lambda: save_checkpoint(
                    path, model=model, optimizer=optimizer, scheduler=scheduler, progress=progress, contract=contract
                ),
                RuntimeError,
            )
        assert hashlib.sha256(path.read_bytes()).hexdigest() == original_digest
        assert not list(Path(directory).glob(".*.tmp"))
        save_checkpoint(
            path, model=model, optimizer=optimizer, scheduler=scheduler, progress=progress, contract=contract, run=run
        )
        _reject(
            lambda: save_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                progress={**progress, "batch_in_epoch": -1},
                contract=contract,
            )
        )
        print(
            json.dumps(
                {
                    "status": "passed",
                    "scope": "synthetic_cpu_checkpoint_roundtrip_no_actual_cpm_training",
                    "format": CHECKPOINT_FORMAT,
                    "full_model_including_frozen_weights": True,
                    "optimizer_scheduler_progress_run_roundtrip": True,
                    "global_rng_roundtrip": True,
                    "rng_restore_is_explicit": True,
                    "strict_contract_and_format_rejection": True,
                    "atomic_failure_and_artifact_protection": True,
                    "cuda_state_tested": False,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
