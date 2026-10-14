"""Free eraser, shared robot/taxels, and spatial contact-driven ink removal."""

from __future__ import annotations

import mujoco
import numpy as np

from ...shared.contact_tactile import SolverDistributedTactileProvider
from ...shared.simulation import ArmHandSimulation
from . import config as C
from . import grasp
from .cleaning import CleaningProgress


class WhiteboardWipeSimulation(ArmHandSimulation):
  def __init__(self, *, ink_seed=None, task_config=C, **kwargs):
    self.task_config = task_config
    self.ink_seed = ink_seed
    kwargs.setdefault("ik_damping", 0.006)
    super().__init__(scene=self.task_config.SCENE_NAME, **kwargs)
    self.model.opt.timestep = self.timestep = 0.001
    # Harder friction constraints reduce soft-contact creep without changing mu.
    self.model.opt.impratio = 10.0
    # Keep the convex contact manifold stable at micrometre-scale separations.
    # Zero eraser margins in scene.xml enable native multi-contact generation.
    self.model.opt.ccd_tolerance = 1e-10
    self.model.opt.ccd_iterations = 100
    right_actuators = list(self._hand_actuators["right"].values())
    self.model.actuator_gainprm[right_actuators, 0] = (
      self.task_config.HAND_VELOCITY_GAIN
    )
    self.model.actuator_biasprm[
      right_actuators, 2
    ] = -self.task_config.HAND_VELOCITY_GAIN
    self.forces = SolverDistributedTactileProvider(
      self.model, target_geom_names=("eraser_handle", "eraser_pad")
    )
    self.refresh_observation()

  def reset(self, **kwargs):
    self.ink_seed = kwargs.pop("ink_seed", self.ink_seed)
    super().reset(**kwargs)
    self.ink_ids = np.array(
      [self.model.geom(f"ink_{i}").id for i in range(self.task_config.INK_COUNT)]
    )
    if not hasattr(self, "_initial_ink_positions"):
      self._initial_ink_positions = self.model.geom_pos[self.ink_ids].copy()
    offset = (
      np.zeros(2)
      if self.ink_seed is None
      else np.random.default_rng(self.ink_seed).uniform(
        -self.task_config.INK_CENTER_RANGE_M, self.task_config.INK_CENTER_RANGE_M
      )
    )
    positions = self._initial_ink_positions.copy()
    positions[:, :2] += offset
    self.set_ink_layout(positions, seed=self.ink_seed)
    self.cleaning = CleaningProgress(self.task_config.INK_COUNT)
    self.remaining = self.cleaning.remaining
    self.sliding_distance = self.cleaning.stroke_m
    self.phase = "observe"
    self.physics_observer = None
    self.evaluation_contact_observer = None
    self.board_tangent_force = 0.0
    self.peak_board_tangent_force = 0.0
    self.direct_hand_board_force = 0.0
    self.peak_direct_hand_board_force = 0.0
    self._hand_geom_ids = {
      i
      for i in range(self.model.ngeom)
      if self.model.body(int(self.model.geom_bodyid[i])).name.startswith("hand_r_")
    }
    self.patch_tangent_load = np.zeros(self.task_config.INK_COUNT)
    self.patch_speed = np.zeros(self.task_config.INK_COUNT)
    self.patch_power = np.zeros(self.task_config.INK_COUNT)
    self.model.geom_rgba[self.ink_ids, 3] = 1
    self.board_force = self.table_force = self.peak_board_force = 0.0
    self.board_contact_center = np.zeros(3)
    self.board_contact_torque = np.zeros(3)
    self.board_contact_count = 0
    self.mean_board_force = 0.0
    self.eraser_body = self.model.body("eraser").id
    self.pad_id = self.model.geom("eraser_pad").id
    self.board_id = self.model.geom("board_surface").id
    self.table_id = self.model.geom("tabletop").id
    self.handle_id = self.model.geom("eraser_handle").id
    self.pickup_center = self.object_pose("eraser")[:3].copy()
    self.grip = grasp.CLOSED_HAND.copy()
    self.spread_grip = grasp.SPREAD_HAND.copy()
    self.open_grip = grasp.OPEN_HAND.copy()
    self.pickup_rotation = (
      self.task_config.INITIAL_ERASER_ROTATION @ grasp.WRIST_ROTATION
    )
    self.wrist_offset = self.task_config.INITIAL_ERASER_ROTATION @ (
      grasp.WRIST_POSITION - np.array([0.43, -0.40, 0.6955])
    )
    result = self.solve_ik(
      "right",
      self.pickup_center
      + [0, 0, self.task_config.PICKUP_CLEARANCE_M]
      + self.wrist_offset,
      self.pickup_rotation,
      # This is an IK branch seed, not the reset posture.
      seed=grasp.APPROACH_ARM_SEED,
      max_iterations=500,
      position_tolerance=0.0003,
      orientation_tolerance=0.004,
      posture_weight=0,
    )
    if not result.success:
      raise RuntimeError(f"Initial open-hand approach is unreachable: {result}")
    # Reset stays at shared home with both hands open. Execute this approach
    # through the actuators after recording has begun.
    self.approach_arm = result.joint_positions.copy()
    self._arm_segments = {}
    self._interpolated_arm_command = {
      side: command.copy() for side, command in self._arm_command.items()
    }
    self._hand_segments = {}
    self._hand_command_names = {
      side: tuple(targets) for side, targets in self._hand_targets.items()
    }
    self._interpolated_hand_command = {
      side: np.array([self._hand_targets[side][name] for name in names])
      for side, names in self._hand_command_names.items()
    }
    if hasattr(self, "forces"):
      self.refresh_observation()

  def set_ink_layout(self, positions, *, seed=None):
    """Set board-local ink geometry before execution, also used for replay."""
    positions = np.asarray(positions, dtype=float)
    if (
      positions.shape != (self.task_config.INK_COUNT, 3)
      or not np.isfinite(positions).all()
    ):
      raise ValueError(
        f"ink positions must be a finite [{self.task_config.INK_COUNT}, 3] array"
      )
    if not np.allclose(positions[:, 2], self._initial_ink_positions[:, 2]):
      raise ValueError("ink must remain on the board surface")
    offset = (positions - self._initial_ink_positions).mean(axis=0)
    if np.any(np.abs(offset[:2]) > self.task_config.INK_CENTER_RANGE_M + 1e-10):
      raise ValueError("ink center is outside the supported randomization region")
    if not np.allclose(positions - self._initial_ink_positions, offset):
      raise ValueError("only translation of the original ink patch is supported")
    self.model.geom_pos[self.ink_ids] = positions
    self.ink_surface_center = (
      self.task_config.BOARD_SURFACE + self.task_config.BOARD_ROTATION @ offset
    )
    self.ink_randomization = {
      "pattern": self.task_config.INK_PATTERN,
      "enabled": seed is not None or bool(np.any(offset)),
      "seed": seed,
      "center_offset_board_m": offset[:2].tolist(),
      "center_range_board_m": self.task_config.INK_CENTER_RANGE_M.tolist(),
      "geom_positions_board_m": positions.tolist(),
      "planning_source": "known simulation ink center; no image recognition",
    }
    mujoco.mj_forward(self.model, self.data)

  def set_arm_joint_goal(self, side, joint_positions):
    super().set_arm_joint_goal(side, joint_positions)
    if hasattr(self, "_arm_segments"):
      self._arm_segments[side] = (
        self.data.time,
        self._interpolated_arm_command[side].copy(),
        self._arm_goal[side].copy(),
      )

  def _before_physics_step(self):
    # Spread each 100 Hz target across the physical steps instead of reaching
    # a small position jump in 1 ms and holding it for the other 9 ms.
    for side, (start_time, start, target) in self._arm_segments.items():
      u = np.clip(
        (self.data.time + self.timestep - start_time)
        / self.task_config.CONTROL_PERIOD_S,
        0,
        1,
      )
      desired = start + u * (target - start)
      command = self._interpolated_arm_command[side]
      limit = self.arm_speed_limit * self.timestep
      command += np.clip(desired - command, -limit, limit)
      self.data.ctrl[self._arm_actuators[side]] = command
      self._arm_command[side][:] = command
    for side, (start_time, start, target) in self._hand_segments.items():
      u = np.clip(
        (self.data.time + self.timestep - start_time)
        / self.task_config.CONTROL_PERIOD_S,
        0,
        1,
      )
      command = start + u * (target - start)
      self._interpolated_hand_command[side][:] = command
      for name, position in zip(self._hand_command_names[side], command, strict=True):
        actuator = self._hand_actuators[side][name]
        velocity = self.hand_position_gain * (
          position - self.data.qpos[self._qpos_address[name]]
        )
        self.data.ctrl[actuator] = np.clip(
          velocity, *self.model.actuator_ctrlrange[actuator]
        )
    mujoco.mj_forward(self.model, self.data)
    self._update_cleaning(integrate=False)
    if self.physics_observer is not None:
      self.physics_observer(self)

  def set_hand_joint_targets(self, names, positions):
    names = tuple(names)
    accepted = super().set_hand_joint_targets(names, positions)
    if hasattr(self, "_hand_segments"):
      for side, ordered in self._hand_command_names.items():
        if any(name in self._hand_targets[side] for name in names):
          self._hand_segments[side] = (
            self.data.time,
            self._interpolated_hand_command[side].copy(),
            np.array([self._hand_targets[side][name] for name in ordered]),
          )
    return accepted

  def refresh_observation(self):
    """Evaluate a synchronized initial/terminal sample without advancing time."""
    for side, dofs in self._arm_dofs.items():
      self.data.qfrc_applied[dofs] = self.data.qfrc_bias[dofs]
      self.data.ctrl[self._arm_actuators[side]] = self._arm_command[side]
      for name, target in self._hand_targets[side].items():
        actuator = self._hand_actuators[side][name]
        velocity = self.hand_position_gain * (
          target - self.data.qpos[self._qpos_address[name]]
        )
        self.data.ctrl[actuator] = np.clip(
          velocity, *self.model.actuator_ctrlrange[actuator]
        )
    mujoco.mj_forward(self.model, self.data)
    self._update_cleaning(integrate=False)

  def step(self, steps=1):
    force_sum = 0.0
    for _ in range(steps):
      before_time = float(self.data.time)
      super().step()
      # mj_step leaves the contact solution used by this integration step.
      # Its position/velocity caches are from the beginning of that step.
      self._update_cleaning()
      if self.evaluation_contact_observer is not None:
        # Cache the integrated solver forces before mj_forward replaces them.
        # The policy adapter commits this sample only after physics validation.
        self.evaluation_contact_observer(self, before_time)
      force_sum += self.board_force
      # Refresh geometry for the next Cartesian control update, only after
      # consuming the actual integrated contact wrench for cleaning.
      mujoco.mj_forward(self.model, self.data)
    self.mean_board_force = force_sum / max(steps, 1)

  def _update_cleaning(self, *, integrate=True):
    self.board_force = self.board_tangent_force = self.table_force = 0.0
    self.board_contact_center.fill(0)
    self.board_contact_torque.fill(0)
    self.board_contact_count = 0
    self.direct_hand_board_force = 0.0
    self.patch_tangent_load.fill(0)
    self.patch_speed.fill(0)
    self.patch_power.fill(0)
    wrench = np.zeros(6)
    rotation = self.data.xmat[self.eraser_body].reshape(3, 3)
    points = self.data.geom_xpos[self.ink_ids]
    local = (points - self.data.xpos[self.eraser_body]) @ rotation
    # A long pen segment may fade only while its whole footprint lies under
    # the felt, rather than when just its center passes under the eraser.
    axes = self.data.geom_xmat[self.ink_ids].reshape(-1, 3, 3)[:, :, 2] @ rotation
    extent = (
      abs(axes) * self.model.geom_size[self.ink_ids, 1, None]
      + self.model.geom_size[self.ink_ids, 0, None]
    )
    covered = (
      (abs(local[:, 0]) + extent[:, 0] < 0.053)
      & (abs(local[:, 1]) + extent[:, 1] < 0.024)
      & (abs(local[:, 2] - self.task_config.PAD_BOTTOM[2]) < 0.002)
    )
    velocity = np.zeros(6)
    mujoco.mj_objectVelocity(
      self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.eraser_body, velocity, 0
    )
    for i, contact in enumerate(self.data.contact):
      pair = {int(contact.geom1), int(contact.geom2)}
      if self.table_id in pair and (self.pad_id in pair or self.handle_id in pair):
        mujoco.mj_contactForce(self.model, self.data, i, wrench)
        self.table_force += max(0.0, float(wrench[0]))
      if self.board_id in pair and pair & self._hand_geom_ids:
        mujoco.mj_contactForce(self.model, self.data, i, wrench)
        self.direct_hand_board_force += max(0.0, float(wrench[0]))
      if self.pad_id not in pair or self.board_id not in pair:
        continue
      mujoco.mj_contactForce(self.model, self.data, i, wrench)
      normal_load = max(0.0, float(wrench[0]))
      self.board_force += normal_load
      self.board_contact_center += normal_load * contact.pos
      self.board_contact_count += int(normal_load > 1e-8)
      tangent_load = float(np.linalg.norm(wrench[1:3]))
      self.board_tangent_force += tangent_load
      # Solver wrench acts on geom2. Preserve the tangential force sign on pad.
      sign = 1.0 if int(contact.geom2) == self.pad_id else -1.0
      frame = contact.frame.reshape(3, 3)
      contact_force = sign * frame.T @ wrench[:3]
      self.board_contact_torque += (
        np.cross(contact.pos - self.data.xipos[self.eraser_body], contact_force)
        + sign * frame.T @ wrench[3:]
      )
      tangential_force = sign * contact.frame.reshape(3, 3)[1:].T @ wrench[1:3]
      point_velocity = velocity[3:] + np.cross(
        velocity[:3], contact.pos - self.data.xipos[self.eraser_body]
      )
      tangential_velocity = point_velocity - self.task_config.BOARD_NORMAL * (
        point_velocity @ self.task_config.BOARD_NORMAL
      )
      speed = float(np.linalg.norm(tangential_velocity))
      power = max(0.0, -float(tangential_force @ tangential_velocity))
      # Approximate the finite felt patch around each solver contact. Every
      # contact's weights sum to at most one, avoiding duplicated friction work.
      distance = np.linalg.norm(points - contact.pos, axis=1)
      weights = np.exp(-0.5 * (distance / 0.028) ** 2) * covered
      total = float(weights.sum())
      if total <= 1e-12:
        continue
      weights /= max(total, 1.0)
      self.patch_tangent_load += weights * tangent_load
      self.patch_power += weights * power
      self.patch_speed += weights * tangent_load * speed
    if self.board_force > 1e-12:
      self.board_contact_center /= self.board_force
    valid = self.patch_tangent_load > 1e-12
    self.patch_speed[valid] /= self.patch_tangent_load[valid]
    if integrate:
      self.peak_board_force = max(self.peak_board_force, self.board_force)
      self.peak_board_tangent_force = max(
        self.peak_board_tangent_force, self.board_tangent_force
      )
      self.peak_direct_hand_board_force = max(
        self.peak_direct_hand_board_force, self.direct_hand_board_force
      )
      self.cleaning.update(
        self.timestep,
        self.board_force,
        self.patch_tangent_load,
        self.patch_speed,
        self.patch_power,
      )
    self.model.geom_rgba[self.ink_ids, 3] = self.remaining**0.5
