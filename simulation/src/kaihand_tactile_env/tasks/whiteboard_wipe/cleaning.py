"""Accumulate ink removal from loaded sliding and dissipated friction work.

These thresholds are initial simulation task parameters, not calibrated material
properties. Normal force is the total pad/board load; tangent load, speed and
friction power must be measured locally for each ink patch. Power is dissipated
contact power in watts, not a Coulomb friction bound or commanded motion.

The 8 N value is an evaluation safety/out-of-distribution guard, not an erasure
efficacy ceiling. Real loaded sliding does not stop removing ink merely because
normal load crosses that threshold, so this accumulator deliberately has no
upper normal-force gate. Formal evaluation may still terminate at 8 N, while an
explicit diagnostic evaluation can continue and retain modelled cleaning
progress. This simple model does not represent pad or board damage caused by
excessive pressure.
"""

import operator

import numpy as np

MIN_NORMAL_FORCE_N = 1.5
BOARD_NORMAL_FORCE_GUARD_N = 8.0
MIN_TANGENT_LOAD_N = 0.02
MIN_SLIDING_SPEED_M_S = 0.005
MAX_SLIDING_SPEED_M_S = 0.15
REQUIRED_WORK_J = 0.001
REQUIRED_STROKE_M = 0.03
REQUIRED_LOADED_TIME_S = 0.25


class CleaningProgress:
  """Track gradual erasure; work, sliding distance and loaded time are required.

  ``remaining`` is an in-place updated opacity fraction in [0, 1]. Each patch
  retains its cumulative valid work, stroke and time across separate passes.
  Static pressing, unloaded motion and short impacts cannot erase a patch.
  """

  def __init__(self, patch_count: int):
    try:
      count = operator.index(patch_count)
    except TypeError as exc:
      raise ValueError("patch_count must be a positive integer") from exc
    if isinstance(patch_count, (bool, np.bool_)) or count <= 0:
      raise ValueError("patch_count must be a positive integer")
    self.patch_count = count
    self.remaining = np.ones(count, dtype=np.float64)
    self.work_j = np.zeros(count, dtype=np.float64)
    self.stroke_m = np.zeros(count, dtype=np.float64)
    self.loaded_time_s = np.zeros(count, dtype=np.float64)

  @property
  def progress(self) -> np.ndarray:
    """Return the completed fraction for every patch."""
    return 1.0 - self.remaining

  def reset(self) -> None:
    """Restore all ink while preserving array references held by the scene."""
    self.remaining.fill(1.0)
    self.work_j.fill(0.0)
    self.stroke_m.fill(0.0)
    self.loaded_time_s.fill(0.0)

  @staticmethod
  def _scalar(value: float, name: str) -> float:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != () or not np.isfinite(array):
      raise ValueError(f"{name} must be a finite scalar")
    return float(array)

  def _patch_array(self, value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (self.patch_count,) or not np.all(np.isfinite(array)):
      raise ValueError(f"{name} must be a finite array of shape ({self.patch_count},)")
    return array

  def update(
    self,
    dt: float,
    normal_force_n: float,
    per_patch_tangent_load_n: np.ndarray,
    per_patch_speed_m_s: np.ndarray,
    per_patch_power_w: np.ndarray,
  ) -> None:
    """Integrate one physics step using actual contact loads and slip velocity.

    Finite inputs with the wrong sign or outside the contact gates simply do
    not contribute. Invalid shapes, non-finite values and nonpositive time
    steps raise ``ValueError`` before any accumulated state changes.
    """
    dt = self._scalar(dt, "dt")
    if dt <= 0:
      raise ValueError("dt must be positive")
    normal_force_n = self._scalar(normal_force_n, "normal_force_n")
    tangent = self._patch_array(per_patch_tangent_load_n, "per_patch_tangent_load_n")
    speed = self._patch_array(per_patch_speed_m_s, "per_patch_speed_m_s")
    power = self._patch_array(per_patch_power_w, "per_patch_power_w")

    if normal_force_n < MIN_NORMAL_FORCE_N:
      return
    active = (
      (self.remaining > 0)
      & (tangent >= MIN_TANGENT_LOAD_N)
      & (speed >= MIN_SLIDING_SPEED_M_S)
      & (speed <= MAX_SLIDING_SPEED_M_S)
      & (power > 0)
    )
    self.work_j[active] += power[active] * dt
    self.stroke_m[active] += speed[active] * dt
    self.loaded_time_s[active] += dt
    progress = np.minimum.reduce(
      (
        self.work_j / REQUIRED_WORK_J,
        self.stroke_m / REQUIRED_STROKE_M,
        self.loaded_time_s / REQUIRED_LOADED_TIME_S,
      )
    )
    self.remaining[:] = 1.0 - np.clip(progress, 0.0, 1.0)
