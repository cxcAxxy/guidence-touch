"""Fail closed on numerical instability; never repair or replay invalid physics."""

from __future__ import annotations

import math

import mujoco
import numpy as np

PHYSICS_INTEGRITY_VERSION = "poker-implicit-failfast-v1"
_NUMERICAL_WARNINGS = (
  mujoco.mjtWarning.mjWARN_BADQPOS,
  mujoco.mjtWarning.mjWARN_BADQVEL,
  mujoco.mjtWarning.mjWARN_BADQACC,
  mujoco.mjtWarning.mjWARN_BADCTRL,
)


class PhysicsIntegrityError(RuntimeError):
  """An invalid trial, with JSON-safe diagnostics for its report."""

  def __init__(self, reason: str, **details: object) -> None:
    self.details = {"reason": reason, **details}
    super().__init__(f"Invalid physics: {reason} ({details})")


def check_physics_state(data: mujoco.MjData, *, stage: str) -> None:
  """Catch warnings even when MuJoCo's auto-reset has restored finite arrays."""
  if not math.isfinite(float(data.time)):
    raise PhysicsIntegrityError("nonfinite simulation time", stage=stage)
  for warning in _NUMERICAL_WARNINGS:
    stat = data.warning[int(warning)]
    if stat.number:
      raise PhysicsIntegrityError(
        "MuJoCo numerical warning",
        stage=stage,
        time_s=float(data.time),
        warning=warning.name,
        count=int(stat.number),
        lastinfo=int(stat.lastinfo),
      )
  for name in ("qpos", "qvel", "qacc", "ctrl"):
    values = getattr(data, name)
    bad = np.flatnonzero(~np.isfinite(values) | (np.abs(values) > mujoco.mjMAXVAL))
    if bad.size:
      raise PhysicsIntegrityError(
        "nonfinite or out-of-range state",
        stage=stage,
        time_s=float(data.time),
        field=name,
        index=int(bad[0]),
      )


def check_physics_clock(
  before: float, after: float, *, timestep: float | None = None
) -> None:
  """Observe every physical step, not just the lower-rate recording clock."""
  if not math.isfinite(before) or not math.isfinite(after):
    raise PhysicsIntegrityError("nonfinite simulation time")
  if after < before - 1.0e-12:
    raise PhysicsIntegrityError(
      "simulation time moved backwards", before_s=before, after_s=after
    )
  if timestep is not None and not math.isclose(
    after - before, timestep, rel_tol=1.0e-7, abs_tol=1.0e-12
  ):
    raise PhysicsIntegrityError(
      "physics step did not advance by one timestep",
      before_s=before,
      after_s=after,
      expected_timestep_s=timestep,
    )
