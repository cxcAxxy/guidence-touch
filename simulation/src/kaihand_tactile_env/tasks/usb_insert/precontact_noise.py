"""Bounded, smooth Gaussian wrist offsets with a one-way tactile cutoff."""

from __future__ import annotations

import secrets

import numpy as np

KNOT_INTERVAL_S = 0.30
RECOVERY_DURATION_S = 0.12
CONTACT_THRESHOLD_N = 1e-6
NEAR_GRASP_SCALE = 0.2


def _smoothstep(value: float) -> float:
  value = float(np.clip(value, 0.0, 1.0))
  return value**3 * (10.0 - 15.0 * value + 6.0 * value**2)


class PrecontactMotionNoise:
  """Sample only before first tactile force; then deterministically blend out.

  The Gaussian standard deviation describes independent XY knot samples.
  Clipping and quintic interpolation make the applied process bounded and
  temporally correlated. No random draws occur after ``latch_contact``.
  """

  def __init__(self, std_m: float = 0.0, seed: int | None = None):
    self.std_m = float(std_m)
    if not np.isfinite(self.std_m) or self.std_m < 0.0:
      raise ValueError(
        "precontact noise standard deviation must be finite and nonnegative"
      )
    if seed is not None and (
      not isinstance(seed, (int, np.integer)) or isinstance(seed, bool) or seed < 0
    ):
      raise ValueError("noise seed must be a nonnegative integer or None")
    self.enabled = self.std_m > 0.0
    self.seed = (
      int(seed) if seed is not None else secrets.randbits(63) if self.enabled else None
    )
    self.bound_m = min(3.0 * self.std_m, 0.002)
    self._rng = np.random.default_rng(self.seed) if self.enabled else None
    self._start_time = None
    self._knot_index = 0
    self._left = np.zeros(3)
    self._right = np.zeros(3)
    self._last_time = -np.inf
    self.random_offset_m = np.zeros(3)
    self.recovery_offset_m = np.zeros(3)
    self._recovery_start = np.zeros(3)
    self.first_contact = None
    self.draw_count = 0
    self.maximum_offset_m = 0.0
    self.commands: list[dict] = []

  def _draw(self) -> np.ndarray:
    self.draw_count += 1
    return np.r_[
      np.clip(self._rng.normal(0, self.std_m, 2), -self.bound_m, self.bound_m), 0.0
    ]

  def sample(self, time_s: float, scale: float = 1.0) -> np.ndarray:
    """Return a target offset; after contact it contains only deterministic recovery."""
    if (
      not np.isfinite([time_s, scale]).all()
      or time_s < self._last_time
      or not 0 <= scale <= 1
    ):
      raise ValueError(
        "noise sampling requires nondecreasing finite time and scale in [0, 1]"
      )
    self._last_time = time_s
    if not self.enabled:
      return np.zeros(3)
    if self.first_contact is not None:
      self.random_offset_m[:] = 0.0
      fraction = (time_s - self.first_contact["time_s"]) / RECOVERY_DURATION_S
      self.recovery_offset_m = (1.0 - _smoothstep(fraction)) * self._recovery_start
      return self.recovery_offset_m.copy()
    if self._start_time is None:
      self._start_time = time_s
      self._right = self._draw()
    elapsed = time_s - self._start_time
    while elapsed >= (self._knot_index + 1) * KNOT_INTERVAL_S:
      self._left = self._right
      self._right = self._draw()
      self._knot_index += 1
    alpha = _smoothstep(
      (elapsed - self._knot_index * KNOT_INTERVAL_S) / KNOT_INTERVAL_S
    )
    self.random_offset_m = scale * ((1.0 - alpha) * self._left + alpha * self._right)
    self.maximum_offset_m = max(
      self.maximum_offset_m, float(np.linalg.norm(self.random_offset_m))
    )
    return self.random_offset_m.copy()

  def latch_contact(
    self, time_s: float, phase: str, pad_name: str, force_n: float
  ) -> bool:
    """A single loaded fingertip permanently stops the episode's random source."""
    if (
      not np.isfinite([time_s, force_n]).all()
      or time_s < self._last_time
      or force_n < 0
    ):
      raise ValueError(
        "tactile cutoff requires finite nonnegative force and nondecreasing time"
      )
    if (
      not self.enabled
      or self.first_contact is not None
      or force_n <= CONTACT_THRESHOLD_N
    ):
      return False
    self.first_contact = {
      "time_s": float(time_s),
      "phase": phase,
      "pad_name": pad_name,
      "normal_force_n": float(force_n),
    }
    self._recovery_start = self.random_offset_m.copy()
    self.recovery_offset_m = self._recovery_start.copy()
    self.random_offset_m[:] = 0.0
    return True

  def report(self) -> dict:
    return {
      "enabled": self.enabled,
      "seed": self.seed,
      "gaussian_knot_std_m": self.std_m,
      "per_axis_bound_m": self.bound_m,
      "axes": "world XY wrist translation; Z, wrist rotation and finger joint targets unchanged",
      "knot_interval_s": KNOT_INTERVAL_S,
      "interpolation": "quintic between clipped independent Gaussian knots",
      "near_grasp_scale": NEAR_GRASP_SCALE,
      "contact_threshold_n": CONTACT_THRESHOLD_N,
      "first_contact": self.first_contact,
      "recovery_duration_s": RECOVERY_DURATION_S,
      "recovery_semantics": "no new random samples after contact; blend the last target offset to zero",
      "gaussian_knot_draws": self.draw_count,
      "maximum_random_offset_norm_m": self.maximum_offset_m,
      "commands": self.commands,
    }
