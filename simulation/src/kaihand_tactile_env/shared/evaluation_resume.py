"""Checks for trial rows that a resumed batch may safely skip."""

from __future__ import annotations


def require_valid_resume_trials(rows: list[dict]) -> None:
  for index, row in enumerate(rows):
    status = row.get("status") if isinstance(row, dict) else None
    expected_success = status == "success"
    if (
      status not in {"success", "task_not_completed"}
      or row.get("valid_trial") is not True
      or row.get("returncode") != 0
      or row.get("success") is not expected_success
    ):
      raise RuntimeError(
        f"resume trial row {index} is invalid or incomplete; "
        "start a new output directory after resolving the trial error"
      )
