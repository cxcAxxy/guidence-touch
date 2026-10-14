"""Opt-in right-arm control variation until the first right fingertip signal.

Noise is added only to position-servo controls before a physics step. Measured
state, task goals, fingers, contact parameters and the accepted force controller
are untouched. A contact signal permanently ends random input for the episode.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np

from kaihand_tactile_env.shared.tactile import (
  RIGHT_FINGERTIP_LINK_NAMES,
  SolverContactTactileProvider,
)

from .mid_full import MidForcePokerSimulation, middle_force_simulation

PRECONTACT_PRESET = "middle-force-precontact-v1"
DEFAULT_XY_JITTER_M = 0.004
DEFAULT_YAW_JITTER_RAD = float(np.deg2rad(0.5))
DEFAULT_PRECONTACT_STD_RAD = float(np.deg2rad(0.03))
MAX_PRECONTACT_STD_RAD = float(np.deg2rad(0.05))
PRECONTACT_STREAM_TAG = 0x5052434E
PRECONTACT_SCHEMA = "poker-precontact-arm-noise-v1"


def precontact_noise_settings(
  std_rad: float = DEFAULT_PRECONTACT_STD_RAD,
) -> dict[str, object]:
  """Pure settings validation; the bounds are not a task-success guarantee."""
  if (
    isinstance(std_rad, (bool, np.bool_))
    or not isinstance(std_rad, (int, float, np.integer, np.floating))
    or not np.isscalar(std_rad)
    or not np.isfinite(std_rad)
    or not 0 <= std_rad <= MAX_PRECONTACT_STD_RAD
  ):
    raise ValueError("precontact std must be finite and between 0 and 0.05 degrees")
  return {
    "schema_version": "poker-precontact-arm-noise-settings-v1",
    "std_rad": float(std_rad),
    "correlation_time_s": 0.15,
    "maximum_offset_rad": 3.0 * float(std_rad),
    "maximum_offset_rate_rad_s": float(np.deg2rad(0.5)),
    "distribution": "bounded_ornstein_uhlenbeck_gaussian",
    "tactile_source": SolverContactTactileProvider.source,
    "contact_link_names": list(RIGHT_FINGERTIP_LINK_NAMES),
    "contact_stop_rule": "first_any_right_fingertip_contact",
    "actuator_scope": "right_arm_seven_position_servo_controls",
    "permanent_until_reset": True,
  }


@contextmanager
def precontact_force_simulation(
  model_path: str | Path,
) -> Iterator[tuple[PrecontactPokerSimulation, dict]]:
  """Use exactly the accepted middle-force model and contact settings."""
  with middle_force_simulation(
    model_path, simulation_type=PrecontactPokerSimulation
  ) as configured:
    yield configured


_TRACE_SHAPES = {
  "time_s": (),
  "tactile_time_s": (),
  "contact": (5,),
  "normal_force_n": (5,),
  "latched_before_step": (),
  "noise_offset_rad": (7,),
  "nominal_ctrl_rad": (7,),
  "actual_ctrl_rad": (7,),
  "control_mode": (),
}


@dataclass
class _NoiseState:
  settings: dict[str, object]
  seed: int
  rng: np.random.Generator
  provider: SolverContactTactileProvider
  lower: np.ndarray
  upper: np.ndarray
  initial_command: np.ndarray
  previous_actual: np.ndarray
  initial_contact: np.ndarray
  initial_force: np.ndarray
  ou: np.ndarray = field(default_factory=lambda: np.zeros(7))
  offset: np.ndarray = field(default_factory=lambda: np.zeros(7))
  latched: bool = False
  detected_time: float | None = None
  tactile_time: float | None = None
  stop_reason: str | None = None
  samples: int = 0
  steps: int = 0
  handoffs: int = 0
  failed: bool = False
  step_nominal: np.ndarray | None = None
  step_offset: np.ndarray = field(default_factory=lambda: np.zeros(7))
  trace: dict[str, list] = field(
    default_factory=lambda: {name: [] for name in _TRACE_SHAPES}
  )


class PrecontactPokerSimulation(MidForcePokerSimulation):
  """Keep random input local to the original right-arm position servo."""

  def reset(self, *args: object, **kwargs: object) -> None:
    # The base constructor calls reset before this subclass has any state.
    self.__dict__.pop("_precontact", None)
    super().reset(*args, **kwargs)

  def configure_precontact_noise(
    self, seed: int, std_rad: float = DEFAULT_PRECONTACT_STD_RAD
  ) -> None:
    settings = precontact_noise_settings(std_rad)
    if (
      isinstance(seed, (bool, np.bool_))
      or not isinstance(seed, (int, np.integer))
      or seed < 0
    ):
      raise ValueError("precontact seed must be a nonnegative integer")
    if float(self.data.time) != 0.0 or self.drive_limit_n is not None:
      raise RuntimeError("configure precontact noise only at reset time before motion")
    if self.scene != "poker-draw":
      raise ValueError("precontact noise requires the isolated poker-draw scene")
    ids = self._arm_actuators["right"]
    joint_limits = self.model.jnt_range[self._arm_joint_ids["right"]]
    control_limits = self.model.actuator_ctrlrange[ids]
    lower = np.maximum(joint_limits[:, 0], control_limits[:, 0])
    upper = np.minimum(joint_limits[:, 1], control_limits[:, 1])
    command = self._arm_command["right"].copy()
    if (
      command.shape != (7,)
      or not np.all(np.isfinite(command))
      or np.any(command < lower)
      or np.any(command > upper)
      or not np.isfinite(self.arm_speed_limit)
      or self.arm_speed_limit <= 0
      or not np.isfinite(self.timestep)
      or self.timestep <= 0
    ):
      raise ValueError("precontact control requires a valid bounded initial command")
    provider = SolverContactTactileProvider(
      self.model, link_names=RIGHT_FINGERTIP_LINK_NAMES
    )
    contact, force = self._read_precontact_sample(provider)
    self._precontact = _NoiseState(
      settings=settings,
      seed=int(seed),
      rng=np.random.default_rng(
        np.random.SeedSequence([int(seed), PRECONTACT_STREAM_TAG])
      ),
      provider=provider,
      lower=lower.copy(),
      upper=upper.copy(),
      initial_command=command.copy(),
      # mj_resetData leaves ctrl zero; that is not an executed arm command.
      previous_actual=command.copy(),
      initial_contact=contact.copy(),
      initial_force=force.copy(),
    )
    self._observe_precontact(contact)

  def _read_precontact_sample(
    self, provider: SolverContactTactileProvider
  ) -> tuple[np.ndarray, np.ndarray]:
    sample = provider.read(self.data)
    contact = np.asarray(sample.contact, dtype=bool).copy()
    force = np.asarray(sample.normal_force, dtype=float).copy()
    if (
      contact.shape != (5,)
      or force.shape != (5,)
      or not np.all(np.isfinite(force))
      or np.any(force < 0)
    ):
      raise RuntimeError("invalid right fingertip tactile sample")
    return contact, force

  def _observe_precontact(self, contact: np.ndarray) -> None:
    state = self._precontact
    if state.latched or not np.any(contact):
      return
    state.latched = True
    state.detected_time = float(self.data.time)
    state.tactile_time = float(self.observation_time)
    state.stop_reason = "first_right_fingertip_contact"
    state.offset.fill(0.0)
    # Preserve the last executed target once, then let the original slew
    # converge toward the unmodified task goal. Later task hold commands win.
    if state.steps and not np.array_equal(
      self._arm_command["right"], state.previous_actual
    ):
      self._arm_command["right"] = state.previous_actual.copy()
      state.handoffs += 1

  def _before_physics_step(self) -> None:
    super()._before_physics_step()
    state = getattr(self, "_precontact", None)
    if state is None:
      return
    ids = self._arm_actuators["right"]
    nominal = self.data.ctrl[ids].copy()
    state.step_nominal = nominal
    state.step_offset = np.zeros(7)
    if state.latched or float(state.settings["std_rad"]) == 0.0:
      return
    dt = self.timestep
    tau = float(state.settings["correlation_time_s"])
    std = float(state.settings["std_rad"])
    state.ou = np.exp(-dt / tau) * state.ou + (
      std * np.sqrt(-np.expm1(-2.0 * dt / tau)) * state.rng.normal(size=7)
    )
    state.samples += 1
    maximum = float(state.settings["maximum_offset_rad"])
    offset_delta = float(state.settings["maximum_offset_rate_rad_s"]) * dt
    command_delta = self.arm_speed_limit * dt
    # Intersect all bounds rather than relaxing one after another. If an
    # externally changed nominal command makes them incompatible, stop safely.
    lower = np.maximum.reduce(
      (
        np.full(7, -maximum),
        state.offset - offset_delta,
        state.lower - nominal,
        state.previous_actual - command_delta - nominal,
      )
    )
    upper = np.minimum.reduce(
      (
        np.full(7, maximum),
        state.offset + offset_delta,
        state.upper - nominal,
        state.previous_actual + command_delta - nominal,
      )
    )
    if np.any(lower > upper + 1e-12):
      raise RuntimeError("precontact offset and actuator slew bounds are incompatible")
    # Tolerate sub-ulp intersections caused by saturated nominal servo steps.
    lower = np.minimum(lower, upper)
    offset = np.clip(np.clip(state.ou, -maximum, maximum), lower, upper)
    self.data.ctrl[ids] = nominal + offset
    state.offset = self.data.ctrl[ids] - nominal
    state.step_offset = state.offset.copy()

  def step(self, steps: int = 1) -> None:
    state = getattr(self, "_precontact", None)
    if state is None:
      return super().step(steps)
    if state.failed:
      raise RuntimeError("reset is required after a failed precontact physics step")
    for _ in range(max(1, int(steps))):
      try:
        contact, _ = self._read_precontact_sample(state.provider)
        self._observe_precontact(contact)
        latched_before = state.latched
        mode = int(self.drive_limit_n is not None)
        if mode and not latched_before:
          raise RuntimeError("force budget cannot start before first fingertip contact")
        time = float(self.data.time)
        state.step_nominal = None
        state.step_offset = np.zeros(7)
        super().step(1)
        actual = self.data.ctrl[self._arm_actuators["right"]].copy()
        state.previous_actual = actual.copy()
        state.steps += 1
        contact, force = self._read_precontact_sample(state.provider)
        self._observe_precontact(contact)
      except Exception:
        state.failed = True
        state.stop_reason = "precontact_step_exception"
        raise
      # The Cartesian budget has its own step implementation and no shared
      # hook. Its unmodified executed control is both nominal and actual here.
      nominal = actual if state.step_nominal is None else state.step_nominal
      values = {
        "time_s": time,
        "tactile_time_s": float(self.observation_time),
        "contact": contact,
        "normal_force_n": force,
        "latched_before_step": latched_before,
        "noise_offset_rad": state.step_offset.copy(),
        "nominal_ctrl_rad": nominal.copy(),
        "actual_ctrl_rad": actual,
        "control_mode": mode,
      }
      for name, value in values.items():
        state.trace[name].append(value)

  def begin_cartesian_drive(self, limit_n: float) -> np.ndarray:
    state = getattr(self, "_precontact", None)
    if state is None or state.failed or not state.latched:
      raise RuntimeError("force budget requires the first fingertip contact latch")
    return super().begin_cartesian_drive(limit_n)

  def precontact_noise_metadata(self) -> dict[str, object]:
    state = getattr(self, "_precontact", None)
    if state is None:
      return {"schema_version": PRECONTACT_SCHEMA, "configured": False}
    ids = self._arm_actuators["right"]
    return {
      "schema_version": PRECONTACT_SCHEMA,
      "configured": True,
      "settings": deepcopy(state.settings),
      "seed": state.seed,
      "stream_tag": PRECONTACT_STREAM_TAG,
      "timestep_s": float(self.timestep),
      "right_arm_actuator_names": [self.model.actuator(int(i)).name for i in ids],
      "right_arm_ctrlrange_rad": self.model.actuator_ctrlrange[ids].tolist(),
      "right_arm_effective_control_bounds_rad": np.column_stack(
        (state.lower, state.upper)
      ).tolist(),
      "right_arm_max_velocity_rad_s": [float(self.arm_speed_limit)] * 7,
      "initial_arm_command_rad": state.initial_command.tolist(),
      "initial_contact": state.initial_contact.tolist(),
      "initial_normal_force_n": state.initial_force.tolist(),
      "contact_latched": state.latched,
      "contact_detected_time_s": state.detected_time,
      "contact_tactile_time_s": state.tactile_time,
      "stop_reason": state.stop_reason,
      "random_sample_count": state.samples,
      "physics_step_count": state.steps,
      "postcontact_random_sample_count": 0,
      "postcontact_noise_disabled": state.latched,
      "command_handoff_count": state.handoffs,
      "failed": state.failed,
      "trace_time_semantics": "time_s is control start; tactile is completed-step solver cache",
      "control_mode_names": {"0": "joint_position_servo", "1": "bounded_cartesian"},
      "noise_offset_semantics": "actual additive random input; zero after latch; nominal may retain servo recovery",
    }

  def precontact_noise_trace(self) -> dict[str, np.ndarray]:
    state = getattr(self, "_precontact", None)
    values = {name: [] for name in _TRACE_SHAPES} if state is None else state.trace
    result = {}
    for name, shape in _TRACE_SHAPES.items():
      dtype = bool if name in {"contact", "latched_before_step"} else float
      if name == "control_mode":
        dtype = np.int8
      result[name] = np.asarray(values[name], dtype=dtype).reshape((-1, *shape)).copy()
    return result
