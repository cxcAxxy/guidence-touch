"""Separate observable task violations from unrecoverable evaluation faults.

Only model-evaluation runners opt in.  Unconfigured simulations keep their
strict behaviour, including expert collection and replay tools.
"""

from __future__ import annotations

import math
from typing import Any

import mujoco
import numpy as np

GUARD_MODES = ("observe", "strict")
DEFAULT_MAX_SIM_SECONDS = 90.0


class TaskGuardViolation(RuntimeError):
  """A task limit that terminates strict evaluation, but not observation."""

  def __init__(self, kind: str, message: str) -> None:
    self.kind = kind
    super().__init__(message)


class FatalSimulationError(RuntimeError):
  """Physics can no longer be advanced or rendered as a valid trajectory."""


def _json_safe(value: Any) -> Any:
  if isinstance(value, dict):
    return {str(key): _json_safe(item) for key, item in value.items()}
  if isinstance(value, np.ndarray) and value.ndim == 0:
    return _json_safe(value.item())
  if isinstance(value, (list, tuple, np.ndarray)):
    return [_json_safe(item) for item in value]
  if isinstance(value, (float, np.floating)):
    return float(value) if math.isfinite(float(value)) else str(float(value))
  if isinstance(value, np.integer):
    return int(value)
  if isinstance(value, np.bool_):
    return bool(value)
  return value


class GuardEventRecorder:
  """Bounded, in-memory aggregation; repeated violations do not flood logs."""

  def __init__(self, mode: str = "observe") -> None:
    if mode not in GUARD_MODES:
      raise ValueError(f"guard mode must be one of {GUARD_MODES}, got {mode!r}")
    self.mode = mode
    self._events: dict[tuple[str, str, str], dict[str, Any]] = {}

  def record(
    self,
    kind: str,
    message: str,
    sim_time: float,
    *,
    measured: float | None = None,
    threshold: float | None = None,
    recovery: str | None = None,
    details: dict[str, Any] | None = None,
  ) -> None:
    # Callers use stable messages.  Times, measurements and contact identities
    # belong in the other fields, not in the aggregation key.
    key = (kind, message, str(threshold))
    timestamp = _json_safe(float(sim_time))
    event = self._events.get(key)
    if event is None:
      event = {
        "kind": kind,
        "message": message,
        "first_sim_time_s": timestamp,
        "last_sim_time_s": timestamp,
        "count": 0,
        "threshold": _json_safe(threshold),
        "measured_min": None,
        "measured_max": None,
        "recovery": recovery,
      }
      self._events[key] = event
    event["count"] += 1
    event["last_sim_time_s"] = timestamp
    if measured is not None:
      event["last_measured"] = _json_safe(float(measured))
      if math.isfinite(float(measured)):
        value = float(measured)
        previous_min = event["measured_min"]
        previous_max = event["measured_max"]
        event["measured_min"] = (
          value if previous_min is None else min(previous_min, value)
        )
        event["measured_max"] = (
          value if previous_max is None else max(previous_max, value)
        )
    if recovery is not None:
      event["recovery"] = recovery
    if details is not None:
      event["last_details"] = _json_safe(details)

  def summary(self) -> list[dict[str, Any]]:
    # Return an independent JSON-safe snapshot for trial/video metadata.
    return _json_safe(list(self._events.values()))


def configure_evaluation_guards(
  simulation: Any, mode: str = "observe"
) -> GuardEventRecorder:
  recorder = GuardEventRecorder(mode)
  simulation._evaluation_guards = recorder
  simulation._evaluation_action_rejected = False
  simulation._evaluation_physics_invalid = False
  return recorder


def is_observe_mode(simulation: Any) -> bool:
  recorder = getattr(simulation, "_evaluation_guards", None)
  return recorder is not None and recorder.mode == "observe"


def record_guard_event(
  simulation: Any,
  kind: str,
  message: str,
  *,
  measured: float | None = None,
  threshold: float | None = None,
  recovery: str | None = None,
  details: dict[str, Any] | None = None,
) -> None:
  recorder = getattr(simulation, "_evaluation_guards", None)
  if recorder is None:
    # Never enable observe mode implicitly for collection/expert simulations.
    recorder = GuardEventRecorder("strict")
    simulation._evaluation_guards = recorder
  recorder.record(
    kind,
    message,
    float(simulation.data.time),
    measured=measured,
    threshold=threshold,
    recovery=recovery,
    details=details,
  )


def task_guard(
  simulation: Any,
  condition: bool,
  kind: str,
  message: str,
  *,
  measured: float | None = None,
  threshold: float | None = None,
  recovery: str | None = None,
  details: dict[str, Any] | None = None,
) -> bool:
  if not condition:
    return False
  record_guard_event(
    simulation,
    kind,
    message,
    measured=measured,
    threshold=threshold,
    recovery=recovery or ("continue" if is_observe_mode(simulation) else "stop"),
    details=details,
  )
  if not is_observe_mode(simulation):
    raise TaskGuardViolation(kind, message)
  return True


def mark_physics_invalid(simulation: Any) -> None:
  simulation._evaluation_physics_invalid = True


def ensure_evaluation_physics(
  simulation: Any, *, before_time: float | None = None
) -> None:
  """Fail closed even when MuJoCo has auto-reset nonfinite arrays to finite."""
  data = simulation.data
  failure: str | None = None
  if not math.isfinite(float(data.time)):
    failure = "nonfinite simulation time"
  for name in ("qpos", "qvel", "qacc", "ctrl"):
    values = getattr(data, name, None)
    if values is not None and (
      not np.isfinite(values).all() or np.any(np.abs(values) > mujoco.mjMAXVAL)
    ):
      failure = f"nonfinite simulation state or out-of-range {name}"
      break
  if failure is None and any(
    warning.number for warning in getattr(data, "warning", ())
  ):
    failure = "simulation solver warning"
  if failure is None and before_time is not None:
    timestep = float(simulation.timestep)
    if not math.isfinite(before_time) or not math.isclose(
      float(data.time) - before_time, timestep, rel_tol=1e-7, abs_tol=1e-12
    ):
      failure = "simulation clock did not advance by one timestep"
  if failure is not None:
    mark_physics_invalid(simulation)
    record_guard_event(simulation, "physics_invalid", failure, recovery="stop")
    raise FatalSimulationError(failure)


def recover_nonfinite_action(simulation: Any, error: Exception) -> bool:
  """Recognize only rejected model outputs, not unrelated runtime failures."""
  message = str(error).casefold()
  known = (
    "nonfinite" in message and ("action" in message or "prediction" in message)
  ) or "tactile policy produced invalid physical actions" in message
  if not known or not is_observe_mode(simulation):
    return False
  record_guard_event(
    simulation,
    "nonfinite_action",
    "policy actions contain nonfinite values",
    recovery="hold_last_valid_command_and_replan",
    details={"error": str(error)},
  )
  simulation._evaluation_action_rejected = True
  return True


def evaluation_guard_report(
  simulation: Any, *, status: str | None = None, error: str | None = None
) -> dict[str, Any]:
  recorder = getattr(simulation, "_evaluation_guards", None)
  mode = recorder.mode if recorder is not None else "strict"
  physics_valid = not getattr(simulation, "_evaluation_physics_invalid", False)
  data = getattr(simulation, "data", None)
  if data is not None:
    physics_valid &= math.isfinite(float(data.time))
    for name in ("qpos", "qvel", "qacc", "ctrl"):
      values = getattr(data, name, None)
      if values is not None:
        physics_valid &= bool(
          np.isfinite(values).all() and np.all(np.abs(values) <= mujoco.mjMAXVAL)
        )
    physics_valid &= not any(w.number for w in getattr(data, "warning", ()))
  if status == "success":
    stop_reason = "success"
  elif status == "policy_action_rejected":
    stop_reason = "guard"
  elif status == "task_not_completed":
    stop_reason = "time_limit"
  elif status in {"error", "viewer_closed"}:
    stop_reason = (
      "guard"
      if physics_valid
      and error
      and ("TaskGuardViolation" in error or "PolicyActionRejected" in error)
      else "fatal_error"
    )
  else:
    stop_reason = None
  return {
    "guard_mode": mode,
    "guard_events": recorder.summary() if recorder is not None else [],
    "stop_reason": stop_reason,
    "physics_valid": bool(physics_valid),
    "formal_metrics_valid": (
      mode == "strict" and bool(physics_valid) and stop_reason != "fatal_error"
    ),
    "diagnostic_only": mode == "observe",
  }
