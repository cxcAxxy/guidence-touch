"""Five-finger expert for physical reverse-thread removal and tabletop return.

The executor changes only robot actuator goals. It never writes bulb/freejoint
state, passive thread joints, equality activation, or applied body forces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from kaihand_tactile_env.shared.simulation import _rotation_vector_world
from kaihand_tactile_env.tasks.bulb_screw.execution import (
  BulbScrewExecutor,
  _TaskFailure,
)
from kaihand_tactile_env.tasks.bulb_screw.grasp import calibrated_grasp

from . import config
from .task import BulbUnscrewMonitor, BulbUnscrewSimulation, BulbUnscrewState


@dataclass(frozen=True)
class BulbUnscrewResult:
  success: bool
  reason: str
  elapsed_s: float
  strokes: int
  maximum_lift_m: float
  peak_fingertip_load_n: tuple[float, ...]
  final_fingertip_load_n: tuple[float, ...]
  phases: tuple[str, ...]
  phase_durations_s: dict[str, float]
  state: BulbUnscrewState
  fingertip_names: tuple[str, ...]
  turn_contact_fraction: tuple[float, ...]
  simultaneous_turn_contact_fraction: float
  rotation_driver: str
  maximum_wrist_rotation_deg: float
  maximum_wrist_displacement_m: float
  initially_installed_verified: bool
  grasp_verified: bool
  thread_disengaged: bool
  fully_extracted: bool
  placed_on_table: bool
  released: bool
  hand_retreated: bool
  placement_peak_support_load_n: float
  placement_peak_right_hand_nonpad_contacts: int
  placement_release_drop_height_m: float


class BulbUnscrewExecutor(BulbScrewExecutor):
  """Bounded counterclockwise gait, lift, return, release, and home policy."""

  def __init__(
    self,
    simulation: BulbUnscrewSimulation,
    observer: Callable[[BulbUnscrewSimulation, str], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    *,
    max_strokes: int = 8,
    speed: str = "fast",
  ) -> None:
    if not isinstance(simulation, BulbUnscrewSimulation):
      raise ValueError("BulbUnscrewExecutor requires BulbUnscrewSimulation")
    super().__init__(
      simulation,
      observer=observer,
      should_stop=should_stop,
      max_strokes=max_strokes,
      speed=speed,
      grasp_mode="five-finger",
    )
    self.monitor = BulbUnscrewMonitor(simulation)
    self._state = self.monitor.measure()
    self._placement_peak_support_load_n = 0.0
    self._placement_peak_nonpad_contacts = 0
    self._placement_release_drop_height_m = 0.0

  def _step(
    self,
    phase,
    *,
    require_grip=False,
    require_thread=False,
    allow_disengage=False,
  ):
    if self.should_stop is not None and self.should_stop():
      raise _TaskFailure("cancelled")
    if self.sim.data.time < self._state.timestamp - 1e-12:
      raise _TaskFailure("simulation reset during task")
    if self.sim.data.time - self._start_time > 360:
      raise _TaskFailure("task exceeded 360 simulation seconds")
    if not self._phases or self._phases[-1] != phase:
      self._phases.append(phase)
    # Recorder callbacks run inside ``sim.step`` after each physics integration,
    # so publish the phase before entering that per-step callback boundary.
    self.sim.task_phase = phase
    self.sim.step()
    if not (
      np.isfinite(self.sim.data.qpos).all() and np.isfinite(self.sim.data.qvel).all()
    ):
      raise _TaskFailure("non-finite simulation state")
    self._state = self.monitor.update()
    self._grip = self._read_grip()
    if self._state.failure_reason is not None:
      raise _TaskFailure(self._state.failure_reason)
    if self._finger_drive_active:
      if not np.array_equal(self.sim.arm_goal["right"], self._fixed_arm_goal):
        raise _TaskFailure("arm goal changed during finger-driven unscrewing")
      wrist, rotation = self.sim.current_pose_matrix("right")
      self._maximum_wrist_rotation = max(
        self._maximum_wrist_rotation,
        float(
          np.linalg.norm(_rotation_vector_world(rotation, self._fixed_wrist_rotation))
        ),
      )
      self._maximum_wrist_displacement = max(
        self._maximum_wrist_displacement,
        float(np.linalg.norm(wrist - self._fixed_wrist_position)),
      )
    self._peak_grip = np.maximum(self._peak_grip, self._grip)
    self._phase_durations[phase] = (
      self._phase_durations.get(phase, 0.0) + self.sim.timestep
    )
    if phase in {"breakaway", "unscrew_thread"}:
      contact = self._grip >= config.GRASP_MIN_FORCE_N
      self._turn_steps += 1
      self._turn_contact_steps += contact
      self._all_contact_steps += int(np.all(contact))
    if phase in {
      "lower_to_table",
      "settle_on_table",
      "release_on_table",
    }:
      self._placement_peak_support_load_n = max(
        self._placement_peak_support_load_n,
        self._state.table_contact_load_n,
      )
      self._placement_peak_nonpad_contacts = max(
        self._placement_peak_nonpad_contacts,
        self._state.right_hand_nonpad_contact_count,
      )
      if self._state.table_contact_load_n > config.MAX_PLACEMENT_SUPPORT_LOAD_N:
        raise _TaskFailure("tabletop placement support load exceeded safety limit")
    self._maximum_height = max(self._maximum_height, self.sim.object_pose("bulb")[2])
    if self.observer is not None:
      self.observer(self.sim, phase)
    self._gap_s = (
      self._gap_s + self.sim.timestep
      if require_grip and np.min(self._grip) < config.GRASP_MIN_FORCE_N
      else 0.0
    )
    if self._gap_s > config.GRIP_GAP_TIMEOUT_S:
      raise _TaskFailure(f"required fingertip contact lost during {phase}")
    if require_thread and not self._state.engaged:
      raise _TaskFailure(f"thread disengaged during {phase}")
    if (
      not allow_disengage
      and not self._state.engaged
      and not self._state.thread_disengaged
    ):
      raise _TaskFailure(f"unexpected thread disengagement during {phase}")
    if np.max(self._grip) > config.MAX_FINGERTIP_LOAD_N:
      raise _TaskFailure(
        f"fingertip load exceeded {config.MAX_FINGERTIP_LOAD_N:g} N during {phase}"
      )

  def _unscrew_finger_step(self, angle_step: float, phase: str) -> None:
    target = self._finger_targets(angle_step)
    self._ramp_hand(
      target,
      0.02,
      phase,
      require_grip=True,
      allow_disengage=True,
    )

  def _finger_regrasp(self) -> None:
    """Reverse-gait reset while the complementary pads keep real support.

    The socket load decreases while backing out, so the screw-in policy's
    1.5 N-per-pad regrasp threshold becomes unnecessarily brittle. Require
    actual contact here, then restore the normal-force set point with all five
    pads before permitting the next turning stroke.
    """
    groups = ((1, 2, 3), (0, 4))
    if self._finger_reset_open is None:
      self._finger_reset_open = np.array(
        [self.sim._hand_targets["right"][name] for name in self._names]
      )
    for group in groups:
      for index in range(40):
        self._finger_step(
          opening=True,
          phase="open_for_regrasp",
          active_fingers=group,
        )
        if index >= 9 and np.max(self._grip[list(group)]) < 0.01:
          break
      if np.max(self._grip[list(group)]) > 0.1:
        raise _TaskFailure("finger group did not unload before reverse reset")
      target = np.array([self.sim._hand_targets["right"][name] for name in self._names])
      for finger in group:
        selection = slice(4 * finger, 4 * finger + 4)
        if self._finger_reset_ready:
          target[selection] = self._finger_reset_open[selection]
        else:
          self._finger_reset_open[selection] = target[selection]
      if self._finger_reset_ready:
        self._hand_motion(
          target,
          0.16,
          "reset_fingers",
          reset_group=group,
          require_thread=True,
        )
      stable = 0
      for _ in range(70):
        self._finger_step(phase="regrasp", require_grip=False, active_fingers=group)
        forces = self._grip[list(group)]
        stable = stable + 1 if min(forces) >= 0.20 and max(forces) < 8.0 else 0
        if stable >= 3:
          break
      else:
        raise _TaskFailure("finger group did not restore physical reverse contact")
    self._finger_reset_ready = True
    self._preload(0.6, "regrasp", require_thread=True)
    if np.min(self._grip) < config.GRASP_MIN_FORCE_N:
      raise _TaskFailure("five-finger contact was not restored after reverse regrasp")

  def _unscrew_with_fingers(self) -> None:
    wrist, rotation = self.sim.current_pose_matrix("right")
    self._command_ee(wrist, rotation)
    self._fixed_arm_goal = self.sim.arm_goal["right"].copy()
    self._fixed_wrist_position = wrist.copy()
    self._fixed_wrist_rotation = rotation.copy()
    self._finger_drive_active = True
    self._finger_regrasp()

    while self._state.engaged:
      if self._strokes >= self.max_strokes:
        raise _TaskFailure("maximum number of counterclockwise strokes reached")
      if self._strokes:
        self._finger_regrasp()
      start_removed = self._state.turns_removed
      target_removed = min(
        start_removed + config.FINGER_STROKE_TURNS,
        config.EXPECTED_RELEASE_TURNS + config.RELEASE_TURN_TOLERANCE,
      )
      phase = "breakaway" if not self._state.breakaway_verified else "unscrew_thread"
      for _ in range(110 if self.speed == "fast" else 150):
        error = target_removed - self._state.turns_removed
        angle = -float(
          np.clip(
            0.12 * 2 * np.pi * error, 0.0, 0.008 if self.speed == "fast" else 0.006
          )
        )
        self._unscrew_finger_step(angle, phase)
        if (
          not self._state.engaged
          or self._state.turns_removed >= target_removed - 0.4 / 360
        ):
          break
      self._strokes += 1
      progress = self._state.turns_removed - start_removed
      if self._state.engaged and progress < min(
        0.025, (target_removed - start_removed) / 2
      ):
        raise _TaskFailure("counterclockwise finger stroke did not rotate the bulb")
    self._finger_drive_active = False
    if not self._state.thread_disengaged or not self._state.disengagement_valid:
      raise _TaskFailure(
        "bulb did not physically disengage at the reverse-thread limit"
      )

  def _motion(self) -> None:
    if not self.sim.thread_engaged or not self.sim.bulb_lit:
      raise _TaskFailure("task must start from the installed, illuminated reset")
    self.grasp = calibrated_grasp(self.sim, five_finger=True)
    pickup = self.timing.pickup_scale
    retreat = self.timing.retreat_scale
    self.sim.set_hand_joint_targets(self._names, self.grasp.open_hand)
    self._advance(
      config.INITIAL_STABLE_DWELL_S + 0.04, "validate_installed", require_thread=True
    )
    if not self._state.initially_installed_verified:
      raise _TaskFailure("installed initial state did not remain physically stable")

    self._move_ee(
      self.grasp.wrist_position + [0, 0, 0.10],
      self.grasp.wrist_rotation,
      2.5 * pickup,
      "hover_socket",
      require_thread=True,
    )
    self._move_ee(
      self.grasp.wrist_position,
      self.grasp.wrist_rotation,
      2.5 * pickup,
      "approach_installed",
      require_thread=True,
    )
    self._hand_motion(
      self.grasp.closed_hand,
      2.0 * pickup,
      "grasp_installed",
      require_thread=True,
    )
    self._preload(
      1.0 * pickup,
      "preload_installed",
      require_thread=True,
    )
    self._advance(
      config.GRASP_STABLE_DWELL_S + 0.08,
      "verify_grasp",
      require_grip=True,
      require_thread=True,
    )
    if not self._state.grasp_verified:
      raise _TaskFailure("five-finger installed grasp did not verify")

    self._unscrew_with_fingers()
    _, carry_rotation = self._object_pose()
    released_position = self.sim.object_pose("bulb")[:3].copy()
    lifted = released_position.copy()
    lifted[2] = config.TRANSFER_HEIGHT_M
    self._move_object(
      lifted,
      carry_rotation,
      3.0 * pickup,
      "extract_clear",
      require_grip=True,
      allow_disengage=True,
    )
    for _ in range(round(0.25 / self.sim.timestep)):
      if self._state.fully_extracted:
        break
      self._servo_object(
        lifted,
        carry_rotation,
        "verify_extracted",
        require_grip=True,
        allow_disengage=True,
      )
    if not self._state.fully_extracted:
      raise _TaskFailure("bulb did not clear the socket and funnel")

    table_target = self.sim.table_target_position
    across = table_target.copy()
    across[2] = config.TRANSFER_HEIGHT_M
    self._move_object(
      across,
      carry_rotation,
      3.0 * pickup,
      "transfer_to_table",
      require_grip=True,
      allow_disengage=True,
    )
    place = table_target.copy()
    place[2] = config.PLACEMENT_COMMAND_Z_M
    self._move_object(
      place,
      carry_rotation,
      3.0 * pickup,
      "lower_to_table",
      require_grip=True,
      allow_disengage=True,
    )
    # Timed Cartesian motion can end while the physical arm is still following
    # its final target. Keep the closed five-pad grasp centered and descend at
    # the same bounded actuator rate until the bulb is genuinely supported and
    # stable on the bare tabletop; only then may any finger unload.
    for _ in range(round(5.0 / 0.02)):
      if self._state.placed_on_table:
        break
      self._servo_object(
        place,
        carry_rotation,
        "settle_on_table",
        require_grip=True,
        allow_disengage=True,
      )
    if not self._state.placed_on_table:
      raise _TaskFailure("bulb did not settle on the bare tabletop target")
    self._placement_release_drop_height_m = max(
      0.0,
      float(self.sim.object_pose("bulb")[2] - config.TABLETOP_HEIGHT_M),
    )
    self._hand_motion(
      self.grasp.open_hand,
      1.0 * retreat,
      "release_on_table",
      allow_disengage=True,
    )

    wrist, rotation = self.sim.current_pose_matrix("right")
    self._move_ee(
      wrist + [0, 0, 0.12],
      rotation,
      1.5 * retreat,
      "retreat_from_table",
      allow_disengage=True,
    )
    self._advance(
      config.RELEASE_DWELL_S + 0.06,
      "verify_release",
      allow_disengage=True,
    )
    if not self._state.released:
      raise _TaskFailure("robot did not release the placed bulb")
    self.sim.set_hand_home("right")
    self._move_ee(
      self._home_wrist_position + [0, 0, 0.18],
      self._home_wrist_rotation,
      2.0 * retreat,
      "return_home",
      allow_disengage=True,
    )
    self._move_ee(
      self._home_wrist_position,
      self._home_wrist_rotation,
      2.0 * retreat,
      "return_home",
      allow_disengage=True,
    )
    start = self.sim.data.qpos[self.sim._arm_qpos["right"]].copy()
    for index in range(1, 151):
      u = index / 150
      alpha = 10 * u**3 - 15 * u**4 + 6 * u**5
      self.sim.set_arm_joint_goal(
        "right", start + alpha * (config.ARM_HOME["right"] - start)
      )
      self._advance(0.02, "return_home", allow_disengage=True)
    self._advance(0.8, "verify_complete", allow_disengage=True)
    if not self._state.success:
      raise _TaskFailure(
        "ordered unscrew, extraction, tabletop placement, release, and home did not complete"
      )

  def execute(self) -> BulbUnscrewResult:
    if self._executed:
      raise RuntimeError("create a fresh executor for each episode")
    self._executed = True
    self._start_time = float(self.sim.data.time)
    self._initial_position = self.sim.object_pose("bulb")[:3].copy()
    self._maximum_height = float(self._initial_position[2])
    reason = "completed"
    success = False
    try:
      self._motion()
      success = bool(self._state.success)
    except _TaskFailure as error:
      reason = str(error)
      self.sim.hold_current_arm_position("right")
    state = self._state
    return BulbUnscrewResult(
      success=success,
      reason=reason,
      elapsed_s=float(self.sim.data.time) - self._start_time,
      strokes=self._strokes,
      maximum_lift_m=float(self._maximum_height - self._initial_position[2]),
      peak_fingertip_load_n=tuple(float(value) for value in self._peak_grip),
      final_fingertip_load_n=tuple(float(value) for value in self._grip),
      phases=tuple(self._phases),
      phase_durations_s=dict(self._phase_durations),
      state=state,
      fingertip_names=("thumb", "index", "middle", "ring", "pinky"),
      turn_contact_fraction=tuple(
        float(value) / max(1, self._turn_steps) for value in self._turn_contact_steps
      ),
      simultaneous_turn_contact_fraction=self._all_contact_steps
      / max(1, self._turn_steps),
      rotation_driver=config.ROTATION_DRIVER,
      maximum_wrist_rotation_deg=float(np.rad2deg(self._maximum_wrist_rotation)),
      maximum_wrist_displacement_m=float(self._maximum_wrist_displacement),
      initially_installed_verified=state.initially_installed_verified,
      grasp_verified=state.grasp_verified,
      thread_disengaged=state.thread_disengaged,
      fully_extracted=state.fully_extracted,
      placed_on_table=state.placed_on_table,
      released=state.released,
      hand_retreated=state.hand_retreated,
      placement_peak_support_load_n=self._placement_peak_support_load_n,
      placement_peak_right_hand_nonpad_contacts=self._placement_peak_nonpad_contacts,
      placement_release_drop_height_m=self._placement_release_drop_height_m,
    )

  def run(self) -> BulbUnscrewResult:
    """Run one episode through the unified expert interface."""
    return self.execute()


Executor = BulbUnscrewExecutor

__all__ = ["BulbUnscrewExecutor", "BulbUnscrewResult", "Executor"]
