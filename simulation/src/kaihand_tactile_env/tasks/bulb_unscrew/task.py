"""Installed reset and ordered physical state monitor for bulb unscrewing."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from kaihand_tactile_env.shared.pickup_randomization import PickupPositionRandomization
from kaihand_tactile_env.shared.simulation import ArmHandSimulation
from kaihand_tactile_env.tasks.bulb_screw.task import (
  BulbScrewMonitor,
  BulbScrewSimulation,
)

from . import config


@dataclass(frozen=True)
class BulbUnscrewState:
  """Instantaneous measurements plus monotonic, ordered task milestones."""

  timestamp: float
  engaged: bool
  seated: bool
  bulb_lit: bool
  clockwise_turns: float
  turns_removed: float
  insertion_depth_m: float
  axial_withdrawal_m: float
  thread_error_m: float
  lateral_error_m: float
  tilt_rad: float
  linear_speed_m_s: float
  angular_speed_rad_s: float
  fingertip_normal_force_n: tuple[float, ...]
  hand_clockwise_torque_nm: float
  hand_counterclockwise_torque_nm: float
  robot_contact_load_n: float
  robot_contact_count: int
  right_hand_contact_load_n: float
  right_hand_contact_count: int
  right_hand_nonpad_contact_count: int
  backstop_load_n: float
  shoulder_contact_load_n: float
  cushion_contact_load_n: float
  socket_contact_load_n: float
  socket_contact_count: int
  table_contact_load_n: float
  table_contact_count: int
  table_xy_error_m: float
  table_z_error_m: float
  wrist_object_distance_m: float
  initially_installed_now: bool
  five_finger_grasp_now: bool
  fully_clear_now: bool
  at_table_target_now: bool
  released_now: bool
  hand_home_now: bool
  arm_home_error_rad: float
  hand_home_error_rad: float
  home_joint_speed_rad_s: float
  initial_stable_duration_s: float = 0.0
  grasp_stable_duration_s: float = 0.0
  clear_duration_s: float = 0.0
  placement_stable_duration_s: float = 0.0
  release_stable_duration_s: float = 0.0
  retreat_stable_duration_s: float = 0.0
  maximum_counterclockwise_torque_nm: float = 0.0
  maximum_turns_removed: float = 0.0
  maximum_axial_withdrawal_m: float = 0.0
  initially_installed_verified: bool = False
  grasp_verified: bool = False
  breakaway_verified: bool = False
  thread_disengaged: bool = False
  disengagement_valid: bool = False
  fully_extracted: bool = False
  placed_on_table: bool = False
  released: bool = False
  hand_retreated: bool = False
  failure_reason: str | None = None
  success: bool = False


class BulbUnscrewSimulation(BulbScrewSimulation):
  """v15 bulb mechanics initialized in a settled, fully installed state.

  Object pose, passive carriage coordinates, and equality activation are written
  only inside :meth:`reset`. Execution can remove the bulb only through the
  normal robot actuators and the inherited automatic release transition.
  """

  def __init__(
    self,
    *,
    position_seed: int | None = None,
    physics_observer: Callable[["BulbUnscrewSimulation"], None] | None = None,
    **kwargs,
  ) -> None:
    if "scene" in kwargs:
      raise TypeError("BulbUnscrewSimulation owns scene='bulb-unscrew'")
    self.position_seed = position_seed
    self._pickup_randomization = None
    self.physics_observer = physics_observer
    self.initialization: dict[str, Any] | None = None
    self.destination_randomization: dict[str, Any] | None = None
    self.destination_pose_wxyz: np.ndarray | None = None
    self._right_hand_home_qpos: np.ndarray | None = None
    model_path = kwargs.pop("model_path", Path(__file__).with_name("scene.xml"))
    ArmHandSimulation.__init__(
      self,
      model_path=model_path,
      scene=config.SCENE_NAME,
      **kwargs,
    )

  def reset(self, *, position_seed: int | None = None, **kwargs) -> None:
    if position_seed is not None:
      self.position_seed = int(position_seed)
    # Use the original pickup-position randomization, without a support body.
    # Construct it before the inherited reset so its cradle-dependent fallback
    # is never entered. The ordinary tabletop target is saved before moving the
    # bulb to the socket as the episode's reset-only installed initial condition.
    if self._pickup_randomization is None:
      self._pickup_randomization = PickupPositionRandomization(self, "bulb")
    BulbScrewSimulation.reset(self, **kwargs)
    self.destination_randomization = dict(self.initial_position_randomization)
    self.destination_pose_wxyz = self.object_pose("bulb").copy()

    self.data.qpos[self._slide] = config.THREAD_TRAVEL_M
    self.data.qpos[self._hinge] = 2 * np.pi * config.TARGET_TURNS
    self.data.qvel[[self._slide_dof, self._hinge_dof]] = 0.0
    mujoco.mj_forward(self.model, self.data)
    carriage = self.model.body("bulb_thread_carriage").id
    self.set_object_pose(
      "bulb",
      self.data.xpos[carriage].copy(),
      self.data.xquat[carriage].copy(),
    )
    self.model.eq_data[self._weld, 3:10] = (0, 0, 0, 1, 0, 0, 0)
    self.data.eq_active[self._weld] = True
    self._capture_armed = False
    self._capture_alignment_started_s = None
    self._capture_alignment_initial_relpose = None
    self._capture_alignment_target_relpose = None
    self._set_bulb_lit(False)
    mujoco.mj_forward(self.model, self.data)

    # Let backstop/shoulder contacts find their loaded equilibrium using the
    # unchanged v15 solver. Bypass only this class's recorder callback while
    # retaining the inherited release/light mechanics before every mj_step.
    ArmHandSimulation.step(
      self, max(1, round(config.INSTALLED_SETTLE_S / self.timestep))
    )
    # Validate the *actual* end of the settle window before zeroing the episode
    # clock/state. This prevents a moving or diverging preset being disguised as
    # a stationary reset by the bookkeeping below.
    settled_state = BulbScrewMonitor(self, auto_confirm_tightening=False).measure()
    settled_valid = bool(
      settled_state.engaged
      and settled_state.seated
      and settled_state.backstop_load_n >= config.INSTALLED_MIN_CONTACT_LOAD_N
      and settled_state.shoulder_contact_load_n >= config.INSTALLED_MIN_CONTACT_LOAD_N
      and abs(settled_state.thread_error_m) <= config.SEATED_DEPTH_TOLERANCE_M
      and settled_state.robot_contact_count == 0
      and settled_state.linear_speed_m_s <= config.SETTLED_LINEAR_SPEED_M_S
      and settled_state.angular_speed_rad_s <= config.SETTLED_ANGULAR_SPEED_RAD_S
    )
    if not settled_valid:
      raise RuntimeError(
        "installed bulb failed to settle under physics: "
        f"engaged={settled_state.engaged}, seated={settled_state.seated}, "
        f"linear={settled_state.linear_speed_m_s:.4g} m/s, "
        f"angular={settled_state.angular_speed_rad_s:.4g} rad/s, "
        f"thread_error={settled_state.thread_error_m:.4g} m"
      )
    self.data.time = 0.0
    self.data.qvel.fill(0.0)
    self.data.qacc.fill(0.0)
    self.data.qacc_warmstart.fill(0.0)
    self.data.qfrc_applied.fill(0.0)
    self.data.xfrc_applied.fill(0.0)
    mujoco.mj_forward(self.model, self.data)

    state = BulbScrewMonitor(self, auto_confirm_tightening=False).measure()
    valid = bool(
      state.engaged
      and state.seated
      and state.backstop_load_n >= config.INSTALLED_MIN_CONTACT_LOAD_N
      and state.shoulder_contact_load_n >= config.INSTALLED_MIN_CONTACT_LOAD_N
      and abs(state.thread_error_m) <= config.SEATED_DEPTH_TOLERANCE_M
      and state.robot_contact_count == 0
      and state.linear_speed_m_s <= config.INSTALLED_LINEAR_SPEED_M_S
      and state.angular_speed_rad_s <= config.INSTALLED_ANGULAR_SPEED_RAD_S
    )
    if not valid:
      raise RuntimeError(
        "installed bulb reset failed physical validation: "
        f"engaged={state.engaged}, seated={state.seated}, "
        f"backstop={state.backstop_load_n:.4g} N, "
        f"shoulder={state.shoulder_contact_load_n:.4g} N, "
        f"thread_error={state.thread_error_m:.4g} m, "
        f"robot_contacts={state.robot_contact_count}"
      )
    self._set_bulb_lit(True)
    mujoco.mj_forward(self.model, self.data)
    self._right_hand_home_qpos = np.array(
      [
        self.data.qpos[self._qpos_address[name]]
        for name in self._hand_joint_names["right"]
      ]
    )
    self.initialization = {
      "schema": "bulb_unscrew_installed_reset_v1",
      "time_s": 0.0,
      "mechanics_version": config.MECHANICS_VERSION,
      "thread_capture_active": True,
      "thread_depth_qpos_m": float(self.data.qpos[self._slide]),
      "thread_angle_qpos_rad": float(self.data.qpos[self._hinge]),
      "bulb_pose_wxyz": self.object_pose("bulb").tolist(),
      "destination_pose_wxyz": self.destination_pose_wxyz.tolist(),
      "settled_under_physics_s": config.INSTALLED_SETTLE_S,
      "prezero_linear_speed_m_s": settled_state.linear_speed_m_s,
      "prezero_angular_speed_rad_s": settled_state.angular_speed_rad_s,
      "prezero_physically_settled": True,
      "velocity_zeroed": True,
      "physically_validated": True,
      "model_parameters_changed": False,
    }

  def initialize_installed(self, *, position_seed: int | None = None) -> None:
    """Reset to the only supported episode start state."""
    self.reset(position_seed=position_seed)

  def initialize_threaded(self) -> None:
    raise RuntimeError(
      "bulb-unscrew always starts fully installed; call initialize_installed()"
    )

  @property
  def table_target_position(self) -> np.ndarray:
    if self.destination_pose_wxyz is None:
      raise RuntimeError("simulation has not completed its installed reset")
    return self.destination_pose_wxyz[:3].copy()

  def _support_contact_metrics(self) -> tuple[float, int, float, int]:
    """Return actual bulb/socket and bulb/table contact metrics."""
    socket_bodies = {
      self.model.body("bulb_socket").id,
      self.model.body("bulb_fixture").id,
    }
    socket_geoms = {
      geom
      for geom in range(self.model.ngeom)
      if int(self.model.geom_bodyid[geom]) in socket_bodies
    }
    table = self.model.geom("tabletop").id
    loads = [0.0, 0.0]
    counts = [0, 0]
    wrench = np.empty(6)
    for index in range(self.data.ncon):
      contact = self.data.contact[index]
      first, second = int(contact.geom1), int(contact.geom2)
      if first in self._bulb_geoms:
        other = second
      elif second in self._bulb_geoms:
        other = first
      else:
        continue
      category = 0 if other in socket_geoms else 1 if other == table else None
      if category is None:
        continue
      mujoco.mj_contactForce(self.model, self.data, index, wrench)
      loads[category] += abs(float(wrench[0]))
      counts[category] += 1
    return loads[0], counts[0], loads[1], counts[1]

  def step(self, steps: int = 1) -> None:
    """Publish every post-physics state, including batched substeps."""
    for _ in range(max(1, int(steps))):
      super().step(1)
      mujoco.mj_forward(self.model, self.data)
      if self.physics_observer is not None:
        self.physics_observer(self)


class BulbUnscrewMonitor:
  """Require installed→grasp→disengage→extract→place→release→home."""

  def __init__(self, simulation: BulbUnscrewSimulation) -> None:
    if not isinstance(simulation, BulbUnscrewSimulation):
      raise ValueError("BulbUnscrewMonitor requires BulbUnscrewSimulation")
    self.simulation = simulation
    self._instant = BulbScrewMonitor(simulation, auto_confirm_tightening=False)
    self.reset()

  def reset(self) -> None:
    self._last_time: float | None = None
    self._initial_s = self._grasp_s = self._clear_s = 0.0
    self._placement_s = self._release_s = self._retreat_s = 0.0
    self._initial_verified = self._grasp_verified = False
    self._breakaway_verified = self._thread_disengaged = False
    self._disengagement_valid = False
    self._fully_extracted = self._placed = self._released = self._retreated = False
    self._failure_reason: str | None = None
    self._was_engaged: bool | None = None
    self._last_engaged_turns = config.TARGET_TURNS
    self._maximum_turns_removed = 0.0
    self._maximum_withdrawal = 0.0
    self._maximum_ccw_torque = 0.0

  def measure(self) -> BulbUnscrewState:
    sim = self.simulation
    screw = self._instant.measure()
    turns = screw.clockwise_turns if screw.engaged else self._last_engaged_turns
    turns_removed = max(0.0, config.TARGET_TURNS - turns)
    withdrawal = max(0.0, config.SEATED_DEPTH_M - screw.insertion_depth_m)
    socket_load, socket_count, table_load, table_count = sim._support_contact_metrics()
    target = sim.table_target_position
    bulb_position = np.asarray(screw.bulb_pose_wxyz[:3])
    table_xy_error = float(np.linalg.norm(bulb_position[:2] - target[:2]))
    table_z_error = float(abs(bulb_position[2] - target[2]))
    five_finger = bool(
      len(screw.fingertip_normal_force_n) == 5
      and min(screw.fingertip_normal_force_n) >= config.GRASP_MIN_FORCE_N
      and screw.right_hand_nonpad_contact_count == 0
    )
    installed_now = bool(
      sim.initialization is not None
      and sim.initialization.get("physically_validated") is True
      and screw.engaged
      and screw.seated
      and screw.bulb_lit
      and screw.robot_contact_count == 0
      and screw.linear_speed_m_s <= 0.001
      and screw.angular_speed_rad_s <= 0.05
    )
    fully_clear_now = bool(
      not screw.engaged
      and screw.insertion_depth_m <= config.FULL_CLEAR_DEPTH_M
      and socket_load <= config.FULL_CLEAR_SOCKET_LOAD_N
      and socket_count == 0
    )
    at_table_target_now = bool(
      not screw.engaged
      and table_xy_error <= config.PLACEMENT_XY_TOLERANCE_M
      and table_z_error <= config.PLACEMENT_Z_TOLERANCE_M
      and screw.tilt_rad <= config.PLACEMENT_TILT_RAD
      and screw.linear_speed_m_s <= config.PLACEMENT_LINEAR_SPEED_M_S
      and screw.angular_speed_rad_s <= config.PLACEMENT_ANGULAR_SPEED_RAD_S
      and table_load >= config.PLACEMENT_MIN_SUPPORT_LOAD_N
      and table_load <= config.MAX_PLACEMENT_SUPPORT_LOAD_N
      and table_count > 0
      and socket_count == 0
    )
    released_now = bool(
      max(screw.fingertip_normal_force_n, default=0.0)
      <= config.RELEASE_FINGERTIP_LOAD_N
      and screw.robot_contact_load_n <= config.RELEASE_HAND_LOAD_N
      and screw.robot_contact_count == 0
    )
    wrist = sim.current_pose_matrix("right")[0]
    wrist_distance = float(np.linalg.norm(wrist - bulb_position))
    arm_position = sim.data.qpos[sim._arm_qpos["right"]]
    arm_velocity = sim.data.qvel[sim._arm_dofs["right"]]
    hand_position = np.array(
      [
        sim.data.qpos[sim._qpos_address[name]]
        for name in sim._hand_joint_names["right"]
      ]
    )
    hand_velocity = np.array(
      [
        sim.data.qvel[sim._qvel_address[name]]
        for name in sim._hand_joint_names["right"]
      ]
    )
    arm_home_error = float(np.max(np.abs(arm_position - config.ARM_HOME["right"])))
    hand_home_error = float(
      np.inf
      if sim._right_hand_home_qpos is None
      else np.max(np.abs(hand_position - sim._right_hand_home_qpos))
    )
    home_speed = float(
      max(
        np.max(np.abs(arm_velocity), initial=0.0),
        np.max(np.abs(hand_velocity), initial=0.0),
      )
    )
    hand_home_now = bool(
      arm_home_error <= config.ARM_HOME_TOLERANCE_RAD
      and sim._right_hand_home_qpos is not None
      and hand_home_error <= config.HAND_HOME_TOLERANCE_RAD
      and home_speed <= config.HOME_SPEED_RAD_S
      and wrist_distance >= config.RETREAT_CLEARANCE_M
    )
    ccw_torque = -screw.hand_clockwise_torque_nm
    return BulbUnscrewState(
      timestamp=screw.timestamp,
      engaged=screw.engaged,
      seated=screw.seated,
      bulb_lit=screw.bulb_lit,
      clockwise_turns=turns,
      turns_removed=turns_removed,
      insertion_depth_m=screw.insertion_depth_m,
      axial_withdrawal_m=withdrawal,
      thread_error_m=screw.thread_error_m,
      lateral_error_m=screw.lateral_error_m,
      tilt_rad=screw.tilt_rad,
      linear_speed_m_s=screw.linear_speed_m_s,
      angular_speed_rad_s=screw.angular_speed_rad_s,
      fingertip_normal_force_n=screw.fingertip_normal_force_n,
      hand_clockwise_torque_nm=screw.hand_clockwise_torque_nm,
      hand_counterclockwise_torque_nm=ccw_torque,
      robot_contact_load_n=screw.robot_contact_load_n,
      robot_contact_count=screw.robot_contact_count,
      right_hand_contact_load_n=screw.right_hand_contact_load_n,
      right_hand_contact_count=screw.right_hand_contact_count,
      right_hand_nonpad_contact_count=screw.right_hand_nonpad_contact_count,
      backstop_load_n=screw.backstop_load_n,
      shoulder_contact_load_n=screw.shoulder_contact_load_n,
      cushion_contact_load_n=screw.cushion_contact_load_n,
      socket_contact_load_n=socket_load,
      socket_contact_count=socket_count,
      table_contact_load_n=table_load,
      table_contact_count=table_count,
      table_xy_error_m=table_xy_error,
      table_z_error_m=table_z_error,
      wrist_object_distance_m=wrist_distance,
      initially_installed_now=installed_now,
      five_finger_grasp_now=five_finger,
      fully_clear_now=fully_clear_now,
      at_table_target_now=at_table_target_now,
      released_now=released_now,
      hand_home_now=hand_home_now,
      arm_home_error_rad=arm_home_error,
      hand_home_error_rad=hand_home_error,
      home_joint_speed_rad_s=home_speed,
      initial_stable_duration_s=self._initial_s,
      grasp_stable_duration_s=self._grasp_s,
      clear_duration_s=self._clear_s,
      placement_stable_duration_s=self._placement_s,
      release_stable_duration_s=self._release_s,
      retreat_stable_duration_s=self._retreat_s,
      maximum_counterclockwise_torque_nm=self._maximum_ccw_torque,
      maximum_turns_removed=self._maximum_turns_removed,
      maximum_axial_withdrawal_m=self._maximum_withdrawal,
      initially_installed_verified=self._initial_verified,
      grasp_verified=self._grasp_verified,
      breakaway_verified=self._breakaway_verified,
      thread_disengaged=self._thread_disengaged,
      disengagement_valid=self._disengagement_valid,
      fully_extracted=self._fully_extracted,
      placed_on_table=self._placed,
      released=self._released,
      hand_retreated=self._retreated,
      failure_reason=self._failure_reason,
      success=bool(
        self._retreated
        and self._failure_reason is None
        and not screw.engaged
        and not screw.bulb_lit
        and at_table_target_now
        and released_now
        and hand_home_now
      ),
    )

  def update(self) -> BulbUnscrewState:
    state = self.measure()
    elapsed = 0.0 if self._last_time is None else state.timestamp - self._last_time
    discontinuity = bool(
      self._last_time is not None
      and (elapsed < -1e-12 or elapsed > 1.5 * self.simulation.timestep)
    )
    if discontinuity:
      self.reset()
      state = self.measure()
      elapsed = 0.0
    self._last_time = state.timestamp
    elapsed = elapsed if elapsed > 1e-12 else 0.0

    if state.engaged:
      self._last_engaged_turns = state.clockwise_turns
    removed = max(0.0, config.TARGET_TURNS - self._last_engaged_turns)
    self._maximum_turns_removed = max(self._maximum_turns_removed, removed)
    self._maximum_withdrawal = max(self._maximum_withdrawal, state.axial_withdrawal_m)
    self._maximum_ccw_torque = max(
      self._maximum_ccw_torque, state.hand_counterclockwise_torque_nm
    )

    if not self._initial_verified:
      self._initial_s = (
        self._initial_s + elapsed if state.initially_installed_now else 0.0
      )
      self._initial_verified = self._initial_s >= config.INITIAL_STABLE_DWELL_S - 1e-12
    if self._initial_verified and not self._grasp_verified:
      self._grasp_s = self._grasp_s + elapsed if state.five_finger_grasp_now else 0.0
      self._grasp_verified = self._grasp_s >= config.GRASP_STABLE_DWELL_S - 1e-12
    if (
      self._grasp_verified
      and not self._breakaway_verified
      and not state.seated
      and not state.bulb_lit
      and removed >= config.BREAKAWAY_MIN_TURNS
      and self._maximum_ccw_torque >= config.BREAKAWAY_MIN_COUNTERCLOCKWISE_TORQUE_NM
    ):
      self._breakaway_verified = True

    previous_engaged = self._was_engaged
    self._was_engaged = state.engaged
    if previous_engaged is True and not state.engaged:
      valid = bool(
        self._breakaway_verified
        and removed >= config.EXPECTED_RELEASE_TURNS - config.RELEASE_TURN_TOLERANCE
      )
      self._thread_disengaged = valid
      self._disengagement_valid = valid
      if not valid and self._failure_reason is None:
        self._failure_reason = (
          "thread disengaged before verified counterclockwise removal"
        )

    if self._thread_disengaged and not self._fully_extracted:
      self._clear_s = self._clear_s + elapsed if state.fully_clear_now else 0.0
      self._fully_extracted = self._clear_s >= config.FULL_CLEAR_DWELL_S - 1e-12
    if self._fully_extracted and not self._placed:
      self._placement_s = (
        self._placement_s + elapsed if state.at_table_target_now else 0.0
      )
      self._placed = self._placement_s >= config.PLACEMENT_DWELL_S - 1e-12
    if self._placed and not self._released:
      self._release_s = (
        self._release_s + elapsed
        if state.at_table_target_now and state.released_now
        else 0.0
      )
      self._released = self._release_s >= config.RELEASE_DWELL_S - 1e-12
    if self._released and not self._retreated:
      self._retreat_s = (
        self._retreat_s + elapsed
        if state.at_table_target_now and state.released_now and state.hand_home_now
        else 0.0
      )
      self._retreated = self._retreat_s >= config.RETREAT_DWELL_S - 1e-12
    current = self.measure()
    return replace(
      current,
      initial_stable_duration_s=self._initial_s,
      grasp_stable_duration_s=self._grasp_s,
      clear_duration_s=self._clear_s,
      placement_stable_duration_s=self._placement_s,
      release_stable_duration_s=self._release_s,
      retreat_stable_duration_s=self._retreat_s,
      maximum_counterclockwise_torque_nm=self._maximum_ccw_torque,
      maximum_turns_removed=self._maximum_turns_removed,
      maximum_axial_withdrawal_m=self._maximum_withdrawal,
      initially_installed_verified=self._initial_verified,
      grasp_verified=self._grasp_verified,
      breakaway_verified=self._breakaway_verified,
      thread_disengaged=self._thread_disengaged,
      disengagement_valid=self._disengagement_valid,
      fully_extracted=self._fully_extracted,
      placed_on_table=self._placed,
      released=self._released,
      hand_retreated=self._retreated,
      failure_reason=self._failure_reason,
      success=bool(
        self._retreated
        and self._failure_reason is None
        and not current.engaged
        and not current.bulb_lit
        and current.at_table_target_now
        and current.released_now
        and current.hand_home_now
      ),
    )


Simulation = BulbUnscrewSimulation

__all__ = [
  "BulbUnscrewMonitor",
  "BulbUnscrewSimulation",
  "BulbUnscrewState",
  "Simulation",
]
