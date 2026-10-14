"""Independent staged_v1 artifacts; never replace legacy task success fields."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


class StagedStateLog:
  def __init__(self, output: Path, monitor: Any, *, identity: dict[str, Any] | None = None):
    self.output = Path(output)
    self.monitor = monitor
    identity = identity or {}
    self.identity = {key: identity[key] for key in (
      "seed", "prediction_horizon", "execute_steps", "checkpoint_path", "checkpoint_sha256", "deployment_id",
    ) if key in identity}
    self.stream = (self.output / "staged_states.jsonl").open("x", encoding="utf-8")
    self.last_time = None
    self.samples = 0

  def record(self, control_tick: int) -> None:
    payload = self.monitor.snapshot()
    timestamp = payload["measurements"].get("simulation_time_s", payload["measurements"].get("time_s"))
    if timestamp is None:
      raise ValueError("staged observer must provide a finite simulation timestamp")
    if self.last_time is not None and timestamp <= self.last_time:
      return
    self.stream.write(json.dumps({**self.identity, "control_tick": control_tick,
                                 "time_s": timestamp, "simulation_time_s": timestamp, **payload},
                                 ensure_ascii=False, allow_nan=False) + "\n")
    self.stream.flush()
    self.last_time = timestamp
    self.samples += 1

  def finish(self, report: dict[str, Any]) -> dict[str, Any]:
    payload = self.monitor.report()
    payload.update(
      diagnostic_only=bool(report.get("diagnostic_only", True)),
      formal_metrics_valid=bool(report.get("formal_metrics_valid", False)),
      physics_valid=report.get("physics_valid"),
      state_log="staged_states.jsonl",
      state_log_samples=self.samples,
      state_log_sampling="every completed 30 Hz control endpoint; observer updates at physics rate",
      legacy_task_success=report.get("task_success"),
      full_duration_evaluation=True,
      **self.identity,
    )
    report["staged_evaluation"] = payload
    with (self.output / "staged_evaluation.json").open("x", encoding="utf-8") as stream:
      json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
      stream.write("\n")
    self.stream.close()
    return payload


class StagedMetricView:
  """Expose stages to video without changing the original outcome monitor."""
  def __init__(self, legacy: Any, monitor: Any):
    self.legacy, self.monitor = legacy, monitor

  def metadata(self) -> dict[str, Any]:
    return {**self.legacy.metadata(), "staged_evaluation_version": "staged_v1"}

  def snapshot(self, simulation: Any = None) -> dict[str, Any]:
    staged = self.monitor.snapshot()
    return {**staged["stages"], "staged_highest_stage": staged["highest_stage"],
            **self.legacy.snapshot(simulation), "staged_evaluation": staged}


def add_staged_batch_results(aggregate: dict[str, Any], output: Path, rows: list[dict[str, Any]]) -> None:
  """Append separate stage rates from completed valid trials, never old flags."""
  reports = []
  for row in rows:
    if row.get("valid_trial") is not True:
      continue
    path = Path(output) / f"seed_{int(row['seed']):03d}" / "staged_evaluation.json"
    if not path.is_file():
      continue
    report = json.loads(path.read_text())
    if report.get("version") != "staged_v1" or report.get("task") != aggregate["task"]:
      raise ValueError("staged batch report task/version mismatch")
    reports.append(report)
  if not reports:
    return
  definitions = reports[0]["stage_definitions"]
  if any(report["stage_definitions"] != definitions or report["thresholds"] != reports[0]["thresholds"]
         for report in reports):
    raise ValueError("staged definitions or thresholds changed between trials")
  count = len(reports)
  stages = {key: {"definition": definition, "count": sum(report["stages"][key] for report in reports),
                 "rate": sum(report["stages"][key] for report in reports) / count}
            for key, definition in definitions.items()}
  is_board = aggregate["task"].startswith("whiteboard-")
  complete = None if is_board else sum(report["complete_success"] for report in reports)
  keys = set().union(*(report.get("continuous_metrics", {}) for report in reports))
  continuous = {}
  for key in sorted(keys):
    values = [report["continuous_metrics"].get(key) for report in reports]
    values = [value for value in values if type(value) in (int, float) and math.isfinite(value)]
    if values:
      continuous[key] = {"mean": sum(values) / len(values), "count": len(values)}
  aggregate["staged_evaluation"] = {
    "version": "staged_v1", "task": aggregate["task"], "valid_staged_trials": count,
    "planned": aggregate["planned"], "stages": stages, "complete_success_count": complete,
    "complete_success_rate": None if complete is None else complete / count,
    "continuous_metrics": continuous, "thresholds": reports[0]["thresholds"],
    "diagnostic_only": True, "formal_metrics_valid": False,
    "metric_scope": "staged diagnostic measurements; legacy success fields preserved",
  }
