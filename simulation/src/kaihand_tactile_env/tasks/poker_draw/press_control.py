"""Small force-control primitives for the four-finger card press.

The task owns the MuJoCo contact measurement and hand kinematics.  This
module deliberately contains only the filtered integral controller and the
per-finger quality monitor so they can be tested without loading a model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PressForceSummary:
  """Per-finger force and continuity metrics over one monitored interval."""

  sample_count: int
  force_means_n: tuple[float, ...]
  force_peaks_n: tuple[float, ...]
  contact_fractions: tuple[float, ...]
  target_band_fractions: tuple[float, ...]
  maximum_contact_gaps_s: tuple[float, ...]
  four_finger_contact_fraction: float
  four_finger_target_band_fraction: float


class PressForceMonitor:
  """Accumulate force tracking and uninterrupted-contact measurements."""

  def __init__(
    self,
    finger_count: int,
    *,
    target_force_n: float,
    timestep: float,
    contact_force_ratio: float,
    target_tolerance_n: float,
  ) -> None:
    if finger_count <= 0:
      raise ValueError("finger_count must be positive")
    if not np.isfinite(target_force_n) or target_force_n <= 0.0:
      raise ValueError("target_force_n must be finite and positive")
    if not np.isfinite(timestep) or timestep <= 0.0:
      raise ValueError("timestep must be finite and positive")
    if not np.isfinite(contact_force_ratio) or not 0.0 < contact_force_ratio <= 1.0:
      raise ValueError("contact_force_ratio must be in (0, 1]")
    if not np.isfinite(target_tolerance_n) or target_tolerance_n <= 0.0:
      raise ValueError("target_tolerance_n must be finite and positive")
    self.finger_count = int(finger_count)
    self.target_force_n = float(target_force_n)
    self.timestep = float(timestep)
    self.contact_force_n = self.target_force_n * float(contact_force_ratio)
    self.target_tolerance_n = float(target_tolerance_n)
    self.reset()

  def reset(self) -> None:
    self._sample_count = 0
    self._force_sum = np.zeros(self.finger_count, dtype=float)
    self._force_peak = np.zeros(self.finger_count, dtype=float)
    self._contact_count = np.zeros(self.finger_count, dtype=int)
    self._target_count = np.zeros(self.finger_count, dtype=int)
    self._current_gap = np.zeros(self.finger_count, dtype=int)
    self._maximum_gap = np.zeros(self.finger_count, dtype=int)
    self._all_contact_count = 0
    self._all_target_count = 0

  def observe(self, forces_n: np.ndarray | tuple[float, ...]) -> None:
    forces = np.asarray(forces_n, dtype=float)
    if forces.shape != (self.finger_count,):
      raise ValueError(f"expected {self.finger_count} force values")
    if not np.all(np.isfinite(forces)) or np.any(forces < 0.0):
      raise ValueError("forces must be finite and non-negative")

    contact = forces >= self.contact_force_n
    in_target_band = np.abs(forces - self.target_force_n) <= self.target_tolerance_n
    self._sample_count += 1
    self._force_sum += forces
    self._force_peak = np.maximum(self._force_peak, forces)
    self._contact_count += contact
    self._target_count += in_target_band
    self._all_contact_count += int(np.all(contact))
    self._all_target_count += int(np.all(in_target_band))
    self._current_gap = np.where(contact, 0, self._current_gap + 1)
    self._maximum_gap = np.maximum(self._maximum_gap, self._current_gap)

  def summary(self) -> PressForceSummary:
    count = self._sample_count
    if count:
      force_means = self._force_sum / count
      contact_fractions = self._contact_count / count
      target_fractions = self._target_count / count
      all_contact_fraction = self._all_contact_count / count
      all_target_fraction = self._all_target_count / count
    else:
      force_means = np.zeros(self.finger_count, dtype=float)
      contact_fractions = np.zeros(self.finger_count, dtype=float)
      target_fractions = np.zeros(self.finger_count, dtype=float)
      all_contact_fraction = 0.0
      all_target_fraction = 0.0
    return PressForceSummary(
      sample_count=count,
      force_means_n=tuple(float(value) for value in force_means),
      force_peaks_n=tuple(float(value) for value in self._force_peak),
      contact_fractions=tuple(float(value) for value in contact_fractions),
      target_band_fractions=tuple(float(value) for value in target_fractions),
      maximum_contact_gaps_s=tuple(
        float(steps * self.timestep) for steps in self._maximum_gap
      ),
      four_finger_contact_fraction=float(all_contact_fraction),
      four_finger_target_band_fraction=float(all_target_fraction),
    )


class FourFingerForceController:
  """Filtered integral force loop producing flat-finger linkage offsets.

  Positive offsets ask the task kinematics to move a pad farther into the
  card.  The task applies each offset as equal-and-opposite changes to joints
  2 and 3, leaving the distal link orientation and joint 4 unchanged.
  """

  def __init__(
    self,
    finger_count: int,
    *,
    target_force_n: float,
    timestep: float,
    update_period_s: float,
    filter_time_constant_s: float,
    integral_gain_rad_per_n_s: float,
    maximum_offset_rad: float,
    maximum_offset_rate_rad_s: float,
    contact_force_n: float,
    contact_recovery_rate_rad_s: float,
    force_deadband_n: float,
  ) -> None:
    positive_values = {
      "target_force_n": target_force_n,
      "timestep": timestep,
      "update_period_s": update_period_s,
      "filter_time_constant_s": filter_time_constant_s,
      "integral_gain_rad_per_n_s": integral_gain_rad_per_n_s,
      "maximum_offset_rad": maximum_offset_rad,
      "maximum_offset_rate_rad_s": maximum_offset_rate_rad_s,
      "contact_force_n": contact_force_n,
      "contact_recovery_rate_rad_s": contact_recovery_rate_rad_s,
    }
    if finger_count <= 0:
      raise ValueError("finger_count must be positive")
    for name, value in positive_values.items():
      if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    if not np.isfinite(force_deadband_n) or force_deadband_n < 0.0:
      raise ValueError("force_deadband_n must be finite and non-negative")
    self.finger_count = int(finger_count)
    self.target_force_n = float(target_force_n)
    self.timestep = float(timestep)
    self.update_steps = max(1, int(round(update_period_s / timestep)))
    self.filter_alpha = float(
      1.0 - np.exp(-self.timestep / float(filter_time_constant_s))
    )
    self.integral_gain = float(integral_gain_rad_per_n_s)
    self.maximum_offset = float(maximum_offset_rad)
    self.maximum_offset_rate = float(maximum_offset_rate_rad_s)
    self.contact_force_n = float(contact_force_n)
    self.contact_recovery_rate = float(contact_recovery_rate_rad_s)
    self.force_deadband_n = float(force_deadband_n)
    self.reset()

  def reset(
    self, initial_forces_n: np.ndarray | tuple[float, ...] | None = None
  ) -> None:
    forces = (
      np.zeros(self.finger_count, dtype=float)
      if initial_forces_n is None
      else self._validated_forces(initial_forces_n)
    )
    self.filtered_forces_n = forces.copy()
    self.offsets_rad = np.zeros(self.finger_count, dtype=float)
    self._steps_since_update = 0

  def observe(
    self,
    forces_n: np.ndarray | tuple[float, ...],
  ) -> tuple[np.ndarray, bool]:
    forces = self._validated_forces(forces_n)
    self.filtered_forces_n += self.filter_alpha * (forces - self.filtered_forces_n)
    self._steps_since_update += 1
    if self._steps_since_update < self.update_steps:
      return self.offsets_rad.copy(), False

    elapsed = self._steps_since_update * self.timestep
    self._steps_since_update = 0
    error = self.target_force_n - self.filtered_forces_n
    effective_error = np.sign(error) * np.maximum(
      np.abs(error) - self.force_deadband_n,
      0.0,
    )
    requested_delta = self.integral_gain * effective_error * elapsed
    maximum_delta = self.maximum_offset_rate * elapsed
    delta = np.clip(requested_delta, -maximum_delta, maximum_delta)
    # A fully unloaded pad needs a bounded approach rather than an integral
    # response based on a stale filtered force.  Once it bears load again the
    # much slower filtered loop takes over, avoiding four stiff servos chasing
    # each other's 2 ms contact impulses.
    delta = np.where(
      forces < self.contact_force_n,
      self.contact_recovery_rate * elapsed,
      delta,
    )
    self.offsets_rad = np.clip(
      self.offsets_rad + delta,
      -self.maximum_offset,
      self.maximum_offset,
    )
    return self.offsets_rad.copy(), True

  def _validated_forces(
    self,
    forces_n: np.ndarray | tuple[float, ...],
  ) -> np.ndarray:
    forces = np.asarray(forces_n, dtype=float)
    if forces.shape != (self.finger_count,):
      raise ValueError(f"expected {self.finger_count} force values")
    if not np.all(np.isfinite(forces)) or np.any(forces < 0.0):
      raise ValueError("forces must be finite and non-negative")
    return forces


__all__ = [
  "FourFingerForceController",
  "PressForceMonitor",
  "PressForceSummary",
]
