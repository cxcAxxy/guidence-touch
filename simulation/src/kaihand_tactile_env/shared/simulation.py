"""Thread-agnostic MuJoCo core for the dual-arm KaiHand workcell.

The arm/hand controller and IK were imported from ``kaihand_teleop_sim`` and
kept ROS-free.  Workcell object state, reproducible reset and replay hooks are
owned by this repository.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import mujoco
import numpy as np

from .config import (
  ARM_HOME,
  OBJECT_NAMES,
  SCENE_NAMES,
  SCENE_OBJECTS,
  SIDES,
  default_model_path,
  legacy_model_path,
  task_config,
)
from .tactile import compile_model_with_genesis_probes

ARM_JOINT_NAMES = {
  side: tuple(f"{side}_arm_joint{i}" for i in range(1, 8)) for side in SIDES
}
HAND_FINGERS = ("thumb", "index", "middle", "ring", "pinky")
HAND_JOINT_NAMES = {
  side: tuple(
    f"hand_{side[0]}_{finger}_joint{joint}"
    for finger in HAND_FINGERS
    for joint in ((1, 2, 3, 5) if finger == "thumb" else (1, 2, 3, 4))
  )
  for side in SIDES
}
FINGERTIP_SITE_NAMES = {
  side: tuple(
    f"hand_{side[0]}_{finger}_{'link6' if finger == 'thumb' else 'link4'}_site"
    for finger in HAND_FINGERS
  )
  for side in SIDES
}
# Historical imports are aliases of the same shared reset posture.
POKER_ARM_HOME = ARM_HOME


@dataclass(frozen=True)
class IkResult:
  success: bool
  joint_positions: np.ndarray
  position_error: float
  orientation_error: float
  iterations: int


@dataclass(frozen=True)
class HandIkResult:
  """Best hand posture found for five world-frame fingertip targets.

  ``joint_names`` and ``joint_positions`` contain the 20 actuated joints in
  canonical thumb/index/middle/ring/pinky order.  The passive
  ``thumb_joint6`` is omitted because the MJCF constrains it one-to-one to
  ``thumb_joint5``.
  """

  success: bool
  joint_names: tuple[str, ...]
  joint_positions: np.ndarray
  fingertip_errors: np.ndarray
  position_error: float
  iterations: int


def _matrix_from_wxyz(quaternion: Iterable[float]) -> np.ndarray:
  quat = np.asarray(tuple(quaternion), dtype=float)
  if quat.shape != (4,) or not np.all(np.isfinite(quat)):
    raise ValueError("quaternion must contain four finite values in wxyz order")
  norm = float(np.linalg.norm(quat))
  if norm < 1.0e-9:
    raise ValueError("quaternion norm is zero")
  quat /= norm
  matrix = np.empty(9, dtype=float)
  mujoco.mju_quat2Mat(matrix, quat)
  return matrix.reshape(3, 3)


def _wxyz_from_matrix(matrix: np.ndarray) -> np.ndarray:
  quat = np.empty(4, dtype=float)
  mujoco.mju_mat2Quat(quat, np.asarray(matrix, dtype=float).reshape(9))
  if quat[0] < 0.0:
    quat *= -1.0
  return quat


def _rotation_vector_world(target: np.ndarray, current: np.ndarray) -> np.ndarray:
  """Return the world-frame axis-angle error taking current to target."""
  error_rotation = target @ current.T
  cosine = float(np.clip((np.trace(error_rotation) - 1.0) * 0.5, -1.0, 1.0))
  angle = float(np.arccos(cosine))
  if angle < 1.0e-8:
    return np.zeros(3)
  if np.pi - angle < 1.0e-5:
    # The skew formula is ill-conditioned at pi.  Recover a stable axis
    # from the diagonal and use the off-diagonal signs for continuity.
    diagonal = np.maximum((np.diag(error_rotation) + 1.0) * 0.5, 0.0)
    axis = np.sqrt(diagonal)
    axis[0] = np.copysign(axis[0], error_rotation[2, 1] - error_rotation[1, 2])
    axis[1] = np.copysign(axis[1], error_rotation[0, 2] - error_rotation[2, 0])
    axis[2] = np.copysign(axis[2], error_rotation[1, 0] - error_rotation[0, 1])
    norm = float(np.linalg.norm(axis))
    return angle * (axis / norm if norm > 1.0e-9 else np.array([1.0, 0.0, 0.0]))
  skew = np.array(
    [
      error_rotation[2, 1] - error_rotation[1, 2],
      error_rotation[0, 2] - error_rotation[2, 0],
      error_rotation[1, 0] - error_rotation[0, 1],
    ]
  )
  return angle * skew / (2.0 * np.sin(angle))


class ArmHandSimulation:
  """
  MuJoCo model, Cartesian IK and low-level actuator target handling.

  The class deliberately has no ROS or viewer dependency.  Callers must
  serialize access if physics stepping and command callbacks use different
  threads.
  """

  def __init__(
    self,
    model_path: str | Path | None = None,
    *,
    arm_speed_limit: float = 3.1416,
    hand_position_gain: float = 24.0,
    ik_damping: float = 0.04,
    add_genesis_probes: bool = True,
    probe_layout_path: str | Path | None = None,
    scene: str = "pick-place",
  ) -> None:
    if scene not in SCENE_NAMES:
      raise ValueError(f"scene must be one of {SCENE_NAMES}, got {scene!r}")
    self.scene = scene
    self._compiled_scene = scene
    self.model_path = (
      Path(model_path or default_model_path(scene)).expanduser().resolve()
    )
    self.genesis_probe_layout = None
    if add_genesis_probes:
      self.model, self.genesis_probe_layout = compile_model_with_genesis_probes(
        self.model_path, probe_layout_path
      )
    else:
      self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
    self.data = mujoco.MjData(self.model)
    self.ik_data = mujoco.MjData(self.model)
    self.timestep = float(self.model.opt.timestep)
    self.arm_speed_limit = float(arm_speed_limit)
    self.hand_position_gain = float(hand_position_gain)
    self.ik_damping = float(ik_damping)

    self._joint_id: dict[str, int] = {}
    self._qpos_address: dict[str, int] = {}
    self._qvel_address: dict[str, int] = {}
    for joint_id in range(self.model.njnt):
      name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
      if name:
        self._joint_id[name] = joint_id
        self._qpos_address[name] = int(self.model.jnt_qposadr[joint_id])
        self._qvel_address[name] = int(self.model.jnt_dofadr[joint_id])

    self._actuator_id: dict[str, int] = {}
    for actuator_id in range(self.model.nu):
      name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
      if name:
        self._actuator_id[name] = actuator_id

    self._arm_joint_ids = {
      side: np.array([self._require_joint(name) for name in ARM_JOINT_NAMES[side]])
      for side in SIDES
    }
    self._arm_qpos = {
      side: np.array([self._qpos_address[name] for name in ARM_JOINT_NAMES[side]])
      for side in SIDES
    }
    self._arm_dofs = {
      side: np.array([self._qvel_address[name] for name in ARM_JOINT_NAMES[side]])
      for side in SIDES
    }
    self._arm_actuators = {
      side: np.array([self._require_actuator(name) for name in ARM_JOINT_NAMES[side]])
      for side in SIDES
    }
    self._site_id = {
      side: self._require_object(mujoco.mjtObj.mjOBJ_SITE, f"{side}_ee_site")
      for side in SIDES
    }
    self._target_mocap = {}
    for side in SIDES:
      body_id = self._require_object(mujoco.mjtObj.mjOBJ_BODY, f"{side}_target")
      mocap_id = int(self.model.body_mocapid[body_id])
      if mocap_id < 0:
        raise RuntimeError(f"{side}_target is not a mocap body")
      self._target_mocap[side] = mocap_id

    self._hand_actuators = {
      side: {
        name: actuator_id
        for name, actuator_id in self._actuator_id.items()
        if name.startswith(f"hand_{side[0]}_")
      }
      for side in SIDES
    }
    self._hand_joint_names = {side: HAND_JOINT_NAMES[side] for side in SIDES}
    self._hand_joint_ids = {
      side: np.array(
        [self._require_joint(name) for name in self._hand_joint_names[side]]
      )
      for side in SIDES
    }
    self._hand_qpos = {
      side: np.array(
        [self._qpos_address[name] for name in self._hand_joint_names[side]]
      )
      for side in SIDES
    }
    self._hand_dofs = {
      side: np.array(
        [self._qvel_address[name] for name in self._hand_joint_names[side]]
      )
      for side in SIDES
    }
    self._fingertip_site_ids = {
      side: np.array(
        [
          self._require_object(mujoco.mjtObj.mjOBJ_SITE, name)
          for name in FINGERTIP_SITE_NAMES[side]
        ]
      )
      for side in SIDES
    }
    self._thumb_joint5_index = {
      side: self._hand_joint_names[side].index(f"hand_{side[0]}_thumb_joint5")
      for side in SIDES
    }
    self._thumb_joint6_qpos = {
      side: self._qpos_address[f"hand_{side[0]}_thumb_joint6"] for side in SIDES
    }
    self._thumb_joint6_dof = {
      side: self._qvel_address[f"hand_{side[0]}_thumb_joint6"] for side in SIDES
    }
    self._hand_targets = {
      side: {name: 0.0 for name in self._hand_actuators[side]} for side in SIDES
    }
    initial_arm_home = ARM_HOME
    self._arm_goal = {side: initial_arm_home[side].copy() for side in SIDES}
    self._arm_command = {side: initial_arm_home[side].copy() for side in SIDES}
    self._pose_target_position = {side: np.zeros(3) for side in SIDES}
    self._pose_target_rotation = {side: np.eye(3) for side in SIDES}

    self._published_joint_names = tuple(
      name
      for name in self._joint_id
      if name.startswith("left_arm_joint")
      or name.startswith("right_arm_joint")
      or name.startswith("hand_l_")
      or name.startswith("hand_r_")
    )
    self.object_names = SCENE_OBJECTS[scene]
    self.model_object_names = tuple(
      name for name in OBJECT_NAMES if f"{name}_freejoint" in self._joint_id
    )
    missing_objects = set(self.object_names) - set(self.model_object_names)
    if missing_objects:
      raise ValueError(
        f"model {self.model_path.name!r} cannot run {scene!r}; "
        f"missing task objects: {sorted(missing_objects)}"
      )
    self._object_body_ids = {
      name: self._require_object(mujoco.mjtObj.mjOBJ_BODY, name)
      for name in self.model_object_names
    }
    self._object_joint_ids = {
      name: self._require_joint(f"{name}_freejoint") for name in self.model_object_names
    }
    self._object_qpos = {
      name: int(self.model.jnt_qposadr[joint_id])
      for name, joint_id in self._object_joint_ids.items()
    }
    self._object_dofs = {
      name: int(self.model.jnt_dofadr[joint_id])
      for name, joint_id in self._object_joint_ids.items()
    }
    self._initial_object_pose = {
      name: self.model.qpos0[address : address + 7].copy()
      for name, address in self._object_qpos.items()
    }
    self._scene_geom_names = {
      scene_name: task_config(scene_name).SCENE_GEOM_NAMES for scene_name in SCENE_NAMES
    }
    # Independent scenes contain only their own geometry.  Filtering is only
    # needed to retain the explicit legacy combined-model replay path.
    for name in self._scene_geom_names[scene]:
      self._require_object(mujoco.mjtObj.mjOBJ_GEOM, name)
    self._scene_geom_names = {
      name: tuple(
        geom
        for geom in geoms
        if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom) >= 0
      )
      for name, geoms in self._scene_geom_names.items()
    }
    self._scene_geom_defaults = {
      self.model.geom(name).id: (
        int(self.model.geom_contype[self.model.geom(name).id]),
        int(self.model.geom_conaffinity[self.model.geom(name).id]),
        self.model.geom_rgba[self.model.geom(name).id].copy(),
      )
      for names in self._scene_geom_names.values()
      for name in names
    }
    self.reset()

  def _require_joint(self, name: str) -> int:
    if name not in self._joint_id:
      raise RuntimeError(f"MuJoCo model is missing joint {name!r}")
    return self._joint_id[name]

  def _require_actuator(self, name: str) -> int:
    if name not in self._actuator_id:
      raise RuntimeError(f"MuJoCo model is missing actuator {name!r}")
    return self._actuator_id[name]

  def _require_object(self, object_type: mujoco.mjtObj, name: str) -> int:
    object_id = mujoco.mj_name2id(self.model, object_type, name)
    if object_id < 0:
      raise RuntimeError(f"MuJoCo model is missing object {name!r}")
    return object_id

  @property
  def joint_names(self) -> tuple[str, ...]:
    return self._published_joint_names

  @property
  def observation_time(self) -> float:
    """Clock of the current forward-evaluated state used by touch and RGB."""
    return float(self.data.time)

  def reset(
    self,
    *,
    seed: int | None = None,
    object_xy_jitter: float = 0.0,
    object_yaw_jitter: float = 0.0,
    randomized_objects: Iterable[str] | None = None,
  ) -> None:
    """Reset the selected scene and optionally randomize its free object."""
    mujoco.mj_resetData(self.model, self.data)
    if object_xy_jitter < 0.0 or object_yaw_jitter < 0.0:
      raise ValueError("object pose jitters cannot be negative")
    randomized = set(
      self.object_names if randomized_objects is None else randomized_objects
    )
    unknown = randomized - set(self.object_names)
    if unknown:
      raise ValueError(f"unknown randomized objects: {sorted(unknown)}")
    rng = np.random.default_rng(seed)
    active_objects = set(self.object_names)
    arm_home = ARM_HOME
    self._configure_scene_geometry()
    for name, address in self._object_qpos.items():
      pose = self._initial_object_pose[name].copy()
      if name not in active_objects:
        # Only the explicit legacy combined model has inactive task objects.
        # Preserve their historical state for old recorded trajectories.
        pose[:3] = (0.0, 0.0, -10.0)
      if name in active_objects and name in randomized and object_xy_jitter:
        pose[:2] += rng.uniform(-object_xy_jitter, object_xy_jitter, size=2)
      if name in active_objects and name in randomized and object_yaw_jitter:
        yaw = float(rng.uniform(-object_yaw_jitter, object_yaw_jitter))
        yaw_quaternion = np.array([np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)])
        rotated = np.empty(4)
        mujoco.mju_mulQuat(rotated, yaw_quaternion, pose[3:7])
        pose[3:7] = rotated
      self.data.qpos[address : address + 7] = pose
      dof = self._object_dofs[name]
      self.data.qvel[dof : dof + 6] = 0.0
    for side in SIDES:
      self.data.qpos[self._arm_qpos[side]] = arm_home[side]
      self.data.qvel[self._arm_dofs[side]] = 0.0
      self._arm_goal[side] = arm_home[side].copy()
      self._arm_command[side] = arm_home[side].copy()
      self.set_hand_closure(side, 0.0)
      for name, target in self._hand_targets[side].items():
        self.data.qpos[self._qpos_address[name]] = target
        self.data.qvel[self._qvel_address[name]] = 0.0
    mujoco.mj_forward(self.model, self.data)

  def _configure_scene_geometry(self) -> None:
    """Show and enable only the collision geometry for ``self.scene``."""
    # Variants share geometry names. An inactive variant must not disable the
    # active scene's board/plug/bulb when walking the registry.
    active = set(self._scene_geom_names[self.scene])
    for name in {name for names in self._scene_geom_names.values() for name in names}:
      geom_id = self.model.geom(name).id
      visible = name in active
      contype, conaffinity, rgba = self._scene_geom_defaults[geom_id]
      self.model.geom_contype[geom_id] = contype if visible else 0
      self.model.geom_conaffinity[geom_id] = conaffinity if visible else 0
      self.model.geom_rgba[geom_id] = rgba if visible else (0.0, 0.0, 0.0, 0.0)

  def set_scene(self, scene: str) -> None:
    """Retain legacy switching; independent scenes require a new instance."""
    if scene not in SCENE_NAMES:
      raise ValueError(f"scene must be one of {SCENE_NAMES}, got {scene!r}")
    legacy_switch = (
      self.model_path == legacy_model_path().resolve()
      and scene in {"pick-place", "poker-draw"}
    )
    if (
      scene != self._compiled_scene and not legacy_switch
      or not set(SCENE_OBJECTS[scene]).issubset(self.model_object_names)
    ):
      raise ValueError(
        "independent task models cannot switch scene in place; "
        f"create ArmHandSimulation(scene={scene!r}) instead"
      )
    self.scene = scene
    self.object_names = SCENE_OBJECTS[scene]
    for side in SIDES:
      position, rotation = self.current_pose_matrix(side)
      self._pose_target_position[side] = position.copy()
      self._pose_target_rotation[side] = rotation.copy()
      self._set_target_mocap(side, position, rotation)
      self.data.ctrl[self._arm_actuators[side]] = self._arm_command[side]
      for name in self._hand_targets[side]:
        self.data.ctrl[self._hand_actuators[side][name]] = 0.0
    mujoco.mj_forward(self.model, self.data)

  def set_arm_joint_goal(self, side: str, joint_positions: Iterable[float]) -> None:
    """Set a clipped seven-joint position goal for one arm."""
    if side not in SIDES:
      raise ValueError(f"side must be one of {SIDES}, got {side!r}")
    values = np.asarray(tuple(joint_positions), dtype=float)
    if values.shape != (7,) or not np.all(np.isfinite(values)):
      raise ValueError("joint_positions must contain seven finite values")
    joint_ids = self._arm_joint_ids[side]
    self._arm_goal[side] = np.clip(
      values,
      self.model.jnt_range[joint_ids, 0],
      self.model.jnt_range[joint_ids, 1],
    )

  def set_hand_closure(self, side: str, closure: float) -> None:
    """Interpolate all 20 actuated hand joints from open to power grasp."""
    if side not in SIDES:
      raise ValueError(f"side must be one of {SIDES}, got {side!r}")
    amount = float(np.clip(closure, 0.0, 1.0))
    prefix = f"hand_{side[0]}_"
    open_pose = {
      "thumb_joint1": 0.0,
      # This is the relaxed collision-free thumb pose.  The former zero target
      # was not physically reachable after closing and caused every home phase
      # to run to its timeout.
      "thumb_joint2": 0.25,
      "thumb_joint3": 0.05,
      "thumb_joint5": 0.0,
    }
    closed_pose = {
      "thumb_joint1": np.deg2rad(10.0),
      # Opposition grasp: abduct across the cylinder while keeping the distal
      # thumb on the side wall, opposite the four fingertips.
      "thumb_joint2": np.deg2rad(35.0),
      "thumb_joint3": np.deg2rad(10.0),
      "thumb_joint5": np.deg2rad(30.0),
    }
    for finger in ("index", "middle", "ring", "pinky"):
      open_pose.update(
        {
          f"{finger}_joint1": 0.0,
          f"{finger}_joint2": 0.0,
          f"{finger}_joint3": 0.0,
          f"{finger}_joint4": 0.0,
        }
      )
      closed_pose.update(
        {
          f"{finger}_joint1": 0.0,
          f"{finger}_joint2": np.deg2rad(35.0),
          f"{finger}_joint3": np.deg2rad(45.0),
          f"{finger}_joint4": np.deg2rad(25.0),
        }
      )
    names = [prefix + suffix for suffix in open_pose]
    targets = [
      (1.0 - amount) * open_pose[suffix] + amount * closed_pose[suffix]
      for suffix in open_pose
    ]
    self.set_hand_joint_targets(names, targets)

  def set_hand_home(self, side: str) -> None:
    """Return every actuated hand joint to the calibrated open pose."""
    if side not in SIDES:
      raise ValueError(f"side must be one of {SIDES}, got {side!r}")
    self.set_hand_closure(side, 0.0)

  def arm_goal_error(self, side: str) -> float:
    """Maximum absolute position error of one arm's commanded joints."""
    return float(
      np.max(np.abs(self.data.qpos[self._arm_qpos[side]] - self._arm_goal[side]))
    )

  def arm_velocity(self, side: str) -> float:
    return float(np.max(np.abs(self.data.qvel[self._arm_dofs[side]])))

  def hold_current_arm_position(self, side: str) -> np.ndarray:
    """Stop a contact approach at the arm's measured joint position."""
    if side not in SIDES:
      raise ValueError(f"side must be one of {SIDES}, got {side!r}")
    joint_positions = self.data.qpos[self._arm_qpos[side]].copy()
    self._arm_goal[side] = joint_positions.copy()
    self._arm_command[side] = joint_positions.copy()
    self.data.ctrl[self._arm_actuators[side]] = joint_positions
    return joint_positions

  def hand_goal_error(self, side: str) -> float:
    return float(
      max(
        abs(self.data.qpos[self._qpos_address[name]] - target)
        for name, target in self._hand_targets[side].items()
      )
    )

  def hand_velocity(self, side: str) -> float:
    return float(
      max(
        abs(self.data.qvel[self._qvel_address[name]])
        for name in self._hand_targets[side]
      )
    )

  def set_object_pose(
    self,
    name: str,
    position: Iterable[float],
    quaternion_wxyz: Iterable[float] = (1.0, 0.0, 0.0, 0.0),
  ) -> None:
    """Teleport one free object; intended for reset and deterministic replay."""
    if name not in self._object_qpos:
      raise ValueError(
        f"unknown object {name!r}; expected one of {self.model_object_names}"
      )
    xyz = np.asarray(tuple(position), dtype=float)
    if xyz.shape != (3,) or not np.all(np.isfinite(xyz)):
      raise ValueError("position must contain three finite values")
    quat = np.asarray(tuple(quaternion_wxyz), dtype=float)
    if quat.shape != (4,) or not np.all(np.isfinite(quat)):
      raise ValueError("quaternion must contain four finite values")
    norm = float(np.linalg.norm(quat))
    if norm < 1.0e-9:
      raise ValueError("quaternion norm is zero")
    quat /= norm
    address = self._object_qpos[name]
    self.data.qpos[address : address + 3] = xyz
    self.data.qpos[address + 3 : address + 7] = quat
    dof = self._object_dofs[name]
    self.data.qvel[dof : dof + 6] = 0.0
    mujoco.mj_forward(self.model, self.data)

  def set_grasp_stabilizer(self, active: bool) -> None:
    """Toggle a soft carry constraint at the cylinder's current relative pose."""
    self.set_object_stabilizer("cylinder", active)

  def set_object_stabilizer(
    self, name: str, active: bool, *, side: str = "right"
  ) -> None:
    """Toggle a soft carry constraint after a verified physical grasp."""
    if name not in self._object_body_ids:
      raise ValueError(
        f"unknown object {name!r}; expected one of {self.model_object_names}"
      )
    if side not in SIDES:
      raise ValueError(f"side must be one of {SIDES}, got {side!r}")
    equality_name = task_config(self.scene).OBJECT_STABILIZERS.get((side, name))
    if equality_name is None:
      raise ValueError(f"no carry stabilizer is available for {side} {name}")
    equality_id = mujoco.mj_name2id(
      self.model, mujoco.mjtObj.mjOBJ_EQUALITY, equality_name
    )
    if equality_id < 0:
      return
    if active:
      hand = self.model.body(f"hand_{side[0]}_base_link").id
      object_id = self._object_body_ids[name]
      hand_rotation = self.data.xmat[hand].reshape(3, 3)
      relative_position = hand_rotation.T @ (
        self.data.xpos[object_id] - self.data.xpos[hand]
      )
      relative_quaternion = np.empty(4)
      inverse_hand_quaternion = self.data.xquat[hand].copy()
      inverse_hand_quaternion[1:] *= -1.0
      mujoco.mju_mulQuat(
        relative_quaternion, inverse_hand_quaternion, self.data.xquat[object_id]
      )
      self.model.eq_data[equality_id, :3] = 0.0
      self.model.eq_data[equality_id, 3:6] = relative_position
      self.model.eq_data[equality_id, 6:10] = relative_quaternion
    self.data.eq_active[equality_id] = active

  def object_pose(self, name: str) -> np.ndarray:
    body_id = self._object_body_ids[name]
    return np.concatenate((self.data.xpos[body_id], self.data.xquat[body_id])).copy()

  def object_twist(self, name: str) -> np.ndarray:
    twist = np.empty(6, dtype=float)
    mujoco.mj_objectVelocity(
      self.model,
      self.data,
      mujoco.mjtObj.mjOBJ_BODY,
      self._object_body_ids[name],
      twist,
      0,
    )
    # MuJoCo orders spatial velocity as angular then linear.
    return np.concatenate((twist[3:], twist[:3]))

  def full_state(self) -> tuple[np.ndarray, np.ndarray]:
    return self.data.qpos.copy(), self.data.qvel.copy()

  def restore_full_state(self, qpos: np.ndarray, qvel: np.ndarray) -> None:
    if qpos.shape != (self.model.nq,) or qvel.shape != (self.model.nv,):
      raise ValueError("qpos/qvel shape does not match the workcell model")
    np.copyto(self.data.qpos, qpos)
    np.copyto(self.data.qvel, qvel)
    mujoco.mj_forward(self.model, self.data)

  def command_state(self) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    arm = np.concatenate(tuple(self._arm_goal[side] for side in SIDES))
    names = tuple(name for side in SIDES for name in self._hand_targets[side])
    hand = np.array(
      [
        self._hand_targets[side][name]
        for side in SIDES
        for name in self._hand_targets[side]
      ],
      dtype=float,
    )
    return arm, hand, names

  @property
  def arm_goal(self) -> dict[str, np.ndarray]:
    return {side: value.copy() for side, value in self._arm_goal.items()}

  def _set_target_mocap(
    self, side: str, position: np.ndarray, rotation: np.ndarray
  ) -> None:
    mocap_id = self._target_mocap[side]
    self.data.mocap_pos[mocap_id] = position
    self.data.mocap_quat[mocap_id] = _wxyz_from_matrix(rotation)

  def current_pose_matrix(self, side: str) -> tuple[np.ndarray, np.ndarray]:
    site_id = self._site_id[side]
    return (
      self.data.site_xpos[site_id].copy(),
      self.data.site_xmat[site_id].reshape(3, 3).copy(),
    )

  def current_pose_wxyz(self, side: str) -> tuple[np.ndarray, np.ndarray]:
    position, rotation = self.current_pose_matrix(side)
    return position, _wxyz_from_matrix(rotation)

  def target_pose_wxyz(self, side: str) -> tuple[np.ndarray, np.ndarray]:
    return (
      self._pose_target_position[side].copy(),
      _wxyz_from_matrix(self._pose_target_rotation[side]),
    )

  def set_pose_target(
    self,
    side: str,
    position: Iterable[float],
    quaternion_wxyz: Iterable[float],
  ) -> IkResult:
    if side not in SIDES:
      raise ValueError(f"side must be one of {SIDES}, got {side!r}")
    target_position = np.asarray(tuple(position), dtype=float)
    if target_position.shape != (3,) or not np.all(np.isfinite(target_position)):
      raise ValueError("position must contain three finite values")
    target_rotation = _matrix_from_wxyz(quaternion_wxyz)

    result = self.solve_ik(
      side,
      target_position,
      target_rotation,
      seed=self._arm_goal[side],
    )
    if result.success:
      self._arm_goal[side] = result.joint_positions.copy()
      self._pose_target_position[side] = target_position.copy()
      self._pose_target_rotation[side] = target_rotation.copy()
      self._set_target_mocap(side, target_position, target_rotation)
    return result

  def nudge_pose_target(
    self,
    side: str,
    *,
    translation_world: Iterable[float] = (0.0, 0.0, 0.0),
    rotation_world: Iterable[float] = (0.0, 0.0, 0.0),
  ) -> IkResult:
    translation = np.asarray(tuple(translation_world), dtype=float)
    rotation_vector = np.asarray(tuple(rotation_world), dtype=float)
    angle = float(np.linalg.norm(rotation_vector))
    delta_rotation = np.eye(3)
    if angle > 1.0e-12:
      axis = rotation_vector / angle
      skew = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
      )
      delta_rotation = (
        np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)
      )
    target_position = self._pose_target_position[side] + translation
    target_rotation = delta_rotation @ self._pose_target_rotation[side]
    return self.set_pose_target(
      side, target_position, _wxyz_from_matrix(target_rotation)
    )

  def solve_ik(
    self,
    side: str,
    target_position: np.ndarray,
    target_rotation: np.ndarray,
    *,
    seed: np.ndarray | None = None,
    max_iterations: int = 80,
    position_tolerance: float = 1.0e-3,
    orientation_tolerance: float = 1.0e-2,
    posture_weight: float = 0.005,
  ) -> IkResult:
    np.copyto(self.ik_data.qpos, self.data.qpos)
    np.copyto(self.ik_data.qvel, self.data.qvel)
    qpos_addresses = self._arm_qpos[side]
    dof_addresses = self._arm_dofs[side]
    joint_ids = self._arm_joint_ids[side]
    q_reference = np.asarray(
      seed if seed is not None else self.data.qpos[qpos_addresses]
    ).copy()
    q = q_reference.copy()
    lower = self.model.jnt_range[joint_ids, 0]
    upper = self.model.jnt_range[joint_ids, 1]
    q = np.clip(q, lower, upper)

    jacobian_position = np.zeros((3, self.model.nv))
    jacobian_rotation = np.zeros((3, self.model.nv))
    site_id = self._site_id[side]
    position_error_norm = float("inf")
    orientation_error_norm = float("inf")

    for iteration in range(1, max_iterations + 1):
      self.ik_data.qpos[qpos_addresses] = q
      mujoco.mj_forward(self.model, self.ik_data)
      current_position = self.ik_data.site_xpos[site_id]
      current_rotation = self.ik_data.site_xmat[site_id].reshape(3, 3)
      position_error = target_position - current_position
      orientation_error = _rotation_vector_world(target_rotation, current_rotation)
      position_error_norm = float(np.linalg.norm(position_error))
      orientation_error_norm = float(np.linalg.norm(orientation_error))
      if (
        position_error_norm <= position_tolerance
        and orientation_error_norm <= orientation_tolerance
      ):
        return IkResult(
          True, q.copy(), position_error_norm, orientation_error_norm, iteration
        )

      jacobian_position.fill(0.0)
      jacobian_rotation.fill(0.0)
      mujoco.mj_jacSite(
        self.model,
        self.ik_data,
        jacobian_position,
        jacobian_rotation,
        site_id,
      )
      jacobian = np.vstack(
        (jacobian_position[:, dof_addresses], jacobian_rotation[:, dof_addresses])
      )
      error = np.concatenate((position_error, orientation_error))
      # Damped least squares plus a null-space posture objective.  The latter
      # resolves the redundant 7-DoF arm toward the supplied seed instead of
      # allowing arbitrary elbow flips and wrist winding.
      damping_matrix = (self.ik_damping**2) * np.eye(6)
      try:
        jacobian_pinv = jacobian.T @ np.linalg.inv(
          jacobian @ jacobian.T + damping_matrix
        )
      except np.linalg.LinAlgError:
        break
      nullspace = np.eye(len(q)) - jacobian_pinv @ jacobian
      delta_q = jacobian_pinv @ error + posture_weight * nullspace @ (q_reference - q)
      delta_norm = float(np.linalg.norm(delta_q))
      if delta_norm > 0.15:
        delta_q *= 0.15 / delta_norm
      q = np.clip(q + 0.7 * delta_q, lower, upper)

    return IkResult(
      False,
      q.copy(),
      position_error_norm,
      orientation_error_norm,
      max_iterations,
    )

  def fingertip_positions(self, side: str) -> np.ndarray:
    """Return the five fingertip sites in world coordinates.

    The returned ``(5, 3)`` array is ordered thumb, index, middle, ring,
    pinky, matching EgoSteer's hand state convention.
    """
    if side not in SIDES:
      raise ValueError(f"side must be one of {SIDES}, got {side!r}")
    return self.data.site_xpos[self._fingertip_site_ids[side]].copy()

  def solve_hand_ik(
    self,
    side: str,
    target_positions: Iterable[Iterable[float]],
    *,
    target_rotations: Iterable[Iterable[Iterable[float]]] | None = None,
    seed: Iterable[float] | None = None,
    arm_joint_positions: Iterable[float] | None = None,
    max_iterations: int = 100,
    position_tolerance: float = 1.5e-3,
    position_weights: float | Iterable[float] = 1.0,
    orientation_weights: float | Iterable[float] = 0.01,
    posture_weight: float = 0.01,
    damping: float = 0.003,
    step_limit: float = 0.2,
  ) -> HandIkResult:
    """Solve one hand's 20 actuators for world-frame fingertip targets.

    ``target_positions`` must be shaped ``(5, 3)`` and ordered thumb, index,
    middle, ring, pinky.  Optional ``target_rotations`` are world-from-site
    rotation matrices shaped ``(5, 3, 3)`` in the same order.  Rotations are
    a weighted soft objective: they select a distal-link posture without
    making success depend on satisfying a generally over-constrained 6-DoF
    pose for every fingertip.  ``orientation_weights`` scale radians into the
    position residual's metre-like units and may be a scalar or one value per
    finger.  ``position_weights`` likewise scale each finger's three position
    residuals, but all position targets must retain a positive weight.

    By default the kinematics use the arm's current Cartesian IK goal, which
    makes the usual call sequence
    ``set_pose_target(...); set_fingertip_targets(...)`` consistent even
    before the simulated arm has moved.  ``arm_joint_positions`` can override
    that seven-joint posture explicitly.

    The supplied ``seed`` (or the previous commanded hand posture) is also a
    null-space posture objective.  This prevents the redundant finger chains
    from changing branch between successive policy actions.  This method only
    uses ``ik_data`` and never changes the live simulation or command targets.
    """
    if side not in SIDES:
      raise ValueError(f"side must be one of {SIDES}, got {side!r}")
    try:
      targets = np.asarray(
        tuple(tuple(point) for point in target_positions), dtype=float
      )
    except (TypeError, ValueError) as error:
      raise ValueError(
        "target_positions must be a finite (5, 3) array in canonical finger order"
      ) from error
    if targets.shape != (5, 3) or not np.all(np.isfinite(targets)):
      raise ValueError(
        "target_positions must be a finite (5, 3) array in canonical finger order"
      )

    rotations: np.ndarray | None
    if target_rotations is None:
      rotations = None
    else:
      try:
        rotations = np.asarray(tuple(target_rotations), dtype=float)
      except (TypeError, ValueError) as error:
        raise ValueError(
          "target_rotations must be five finite 3x3 rotation matrices"
        ) from error
      if rotations.shape != (5, 3, 3) or not np.all(np.isfinite(rotations)):
        raise ValueError(
          "target_rotations must be five finite 3x3 rotation matrices"
        )
      identity = np.eye(3)
      gram = np.swapaxes(rotations, -1, -2) @ rotations
      determinants = np.linalg.det(rotations)
      if not np.allclose(gram, identity, atol=5.0e-4, rtol=0.0) or not np.allclose(
        determinants, 1.0, atol=5.0e-4, rtol=0.0
      ):
        raise ValueError(
          "target_rotations must be five finite 3x3 rotation matrices"
        )

    def finger_weights(
      values: float | Iterable[float], name: str, *, strictly_positive: bool
    ) -> np.ndarray:
      try:
        if np.isscalar(values):
          weights = np.full(5, float(values), dtype=float)
        else:
          weights = np.asarray(tuple(values), dtype=float)
      except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite scalar or five values") from error
      invalid = weights.shape != (5,) or not np.all(np.isfinite(weights))
      if strictly_positive:
        invalid = invalid or bool(np.any(weights <= 0.0))
      else:
        invalid = invalid or bool(np.any(weights < 0.0))
      if invalid:
        qualifier = "positive " if strictly_positive else "non-negative "
        raise ValueError(
          f"{name} must be a finite {qualifier}scalar or five values"
        )
      return weights

    position_scales = finger_weights(
      position_weights, "position_weights", strictly_positive=True
    )
    orientation_scales = finger_weights(
      orientation_weights, "orientation_weights", strictly_positive=False
    )
    use_orientation = rotations is not None and bool(np.any(orientation_scales > 0.0))
    if max_iterations <= 0:
      raise ValueError("max_iterations must be positive")
    if position_tolerance <= 0.0 or not np.isfinite(position_tolerance):
      raise ValueError("position_tolerance must be positive and finite")
    if posture_weight < 0.0 or not np.isfinite(posture_weight):
      raise ValueError("posture_weight must be non-negative and finite")
    if damping <= 0.0 or not np.isfinite(damping):
      raise ValueError("damping must be positive and finite")
    if step_limit <= 0.0 or not np.isfinite(step_limit):
      raise ValueError("step_limit must be positive and finite")

    joint_names = self._hand_joint_names[side]
    if seed is None:
      q_reference = np.array(
        [self._hand_targets[side][name] for name in joint_names], dtype=float
      )
    else:
      q_reference = np.asarray(tuple(seed), dtype=float)
      if q_reference.shape != (20,) or not np.all(np.isfinite(q_reference)):
        raise ValueError("seed must contain 20 finite actuated hand joint values")

    if arm_joint_positions is None:
      arm_q = self._arm_goal[side].copy()
    else:
      arm_q = np.asarray(tuple(arm_joint_positions), dtype=float)
      if arm_q.shape != (7,) or not np.all(np.isfinite(arm_q)):
        raise ValueError("arm_joint_positions must contain seven finite values")
      arm_q = np.clip(
        arm_q,
        self.model.jnt_range[self._arm_joint_ids[side], 0],
        self.model.jnt_range[self._arm_joint_ids[side], 1],
      )

    joint_ids = self._hand_joint_ids[side]
    qpos_addresses = self._hand_qpos[side]
    dof_addresses = self._hand_dofs[side]
    lower = self.model.jnt_range[joint_ids, 0]
    upper = self.model.jnt_range[joint_ids, 1]
    q_reference = np.clip(q_reference, lower, upper)
    q = q_reference.copy()

    # Establish the position-feasible branch first, then use the soft pose
    # rows to resolve its redundant distal posture.  Besides preserving the
    # hard position-success semantics, this avoids a poor orientation local
    # minimum when the supplied seed is far from a closing grasp.
    initialization_iterations = 0
    if use_orientation:
      position_initialization = self.solve_hand_ik(
        side,
        targets,
        seed=q_reference,
        arm_joint_positions=arm_q,
        max_iterations=max_iterations,
        position_tolerance=position_tolerance,
        position_weights=position_scales,
        posture_weight=posture_weight,
        damping=damping,
        step_limit=step_limit,
      )
      q = position_initialization.joint_positions.copy()
      initialization_iterations = position_initialization.iterations

    np.copyto(self.ik_data.qpos, self.data.qpos)
    np.copyto(self.ik_data.qvel, self.data.qvel)
    self.ik_data.qpos[self._arm_qpos[side]] = arm_q
    thumb_joint5_index = self._thumb_joint5_index[side]
    thumb_joint6_qpos = self._thumb_joint6_qpos[side]
    thumb_joint6_dof = self._thumb_joint6_dof[side]
    site_ids = self._fingertip_site_ids[side]

    def evaluate(
      candidate: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
      self.ik_data.qpos[qpos_addresses] = candidate
      # mj_forward does not project equality constraints onto qpos.  Enforce
      # the model's thumb_joint6 == thumb_joint5 relation explicitly so FK and
      # the solver agree with the physical trajectory after actuation.
      self.ik_data.qpos[thumb_joint6_qpos] = candidate[thumb_joint5_index]
      mujoco.mj_forward(self.model, self.ik_data)
      errors = targets - self.ik_data.site_xpos[site_ids]
      error_norms = np.linalg.norm(errors, axis=1)
      orientation_errors = np.zeros((5, 3), dtype=float)
      orientation_error_norms = np.zeros(5, dtype=float)
      if use_orientation:
        for finger_index, site_id in enumerate(site_ids):
          current_rotation = self.ik_data.site_xmat[site_id].reshape(3, 3)
          orientation_errors[finger_index] = _rotation_vector_world(
            rotations[finger_index], current_rotation
          )
        orientation_error_norms = np.linalg.norm(orientation_errors, axis=1)
      weighted_position_errors = position_scales[:, None] * errors
      task_cost = float(np.sum(weighted_position_errors**2))
      if use_orientation:
        weighted_orientation_errors = orientation_scales[:, None] * orientation_errors
        task_cost += float(np.sum(weighted_orientation_errors**2))
      return (
        errors,
        error_norms,
        orientation_errors,
        orientation_error_norms,
        task_cost,
      )

    (
      errors,
      error_norms,
      orientation_errors,
      orientation_error_norms,
      task_cost,
    ) = evaluate(q)

    def result_key(
      position_errors: np.ndarray, norms: np.ndarray, cost: float
    ) -> tuple[float, float, float]:
      maximum_error = float(np.max(norms))
      if use_orientation:
        # Position remains the hard acceptance condition.  Once it is met,
        # rank equivalent IK branches by the weighted pose objective.
        return (
          max(0.0, maximum_error - position_tolerance),
          cost,
          maximum_error,
        )
      return (maximum_error, float(np.sum(position_errors**2)), cost)

    best_q = q.copy()
    best_error_norms = error_norms.copy()
    best_key = result_key(errors, error_norms, task_cost)
    if float(np.max(error_norms)) <= position_tolerance and not use_orientation:
      return HandIkResult(
        True,
        joint_names,
        q.copy(),
        error_norms.copy(),
        float(np.max(error_norms)),
        0,
      )

    jacobian_position = np.zeros((3, self.model.nv))
    jacobian_rotation = np.zeros((3, self.model.nv))
    task_dimensions = 30 if use_orientation else 15
    jacobian = np.zeros((task_dimensions, 20))
    identity_task = np.eye(task_dimensions)
    identity_joint = np.eye(20)
    completed_iterations = initialization_iterations
    stalled_iterations = 0

    for iteration in range(1, max_iterations + 1):
      completed_iterations = initialization_iterations + iteration
      jacobian.fill(0.0)
      for finger_index, site_id in enumerate(site_ids):
        jacobian_position.fill(0.0)
        jacobian_rotation.fill(0.0)
        mujoco.mj_jacSite(
          self.model,
          self.ik_data,
          jacobian_position,
          jacobian_rotation if use_orientation else None,
          int(site_id),
        )
        row = slice(3 * finger_index, 3 * finger_index + 3)
        jacobian[row] = (
          position_scales[finger_index] * jacobian_position[:, dof_addresses]
        )
        if finger_index == 0:
          # joint5 rotates both its own link and, through the equality, joint6.
          # The reduced-coordinate derivative is therefore the sum of the two
          # unconstrained MuJoCo Jacobian columns.
          jacobian[row, thumb_joint5_index] += (
            position_scales[finger_index]
            * jacobian_position[:, thumb_joint6_dof]
          )
        if use_orientation:
          rotation_row = slice(15 + 3 * finger_index, 18 + 3 * finger_index)
          jacobian[rotation_row] = (
            orientation_scales[finger_index]
            * jacobian_rotation[:, dof_addresses]
          )
          if finger_index == 0:
            # The equality-reduced rotational derivative must include the
            # passive distal joint for exactly the same reason as position.
            jacobian[rotation_row, thumb_joint5_index] += (
              orientation_scales[finger_index]
              * jacobian_rotation[:, thumb_joint6_dof]
            )

      try:
        if use_orientation:
          # With pose rows there are more residuals than hand DoFs.  Solving
          # in joint space avoids inverting a rank-deficient 30x30 task-space
          # matrix and preserves the independent per-finger block structure.
          jacobian_pinv = np.linalg.solve(
            jacobian.T @ jacobian + (damping**2) * identity_joint,
            jacobian.T,
          )
        else:
          task_matrix = jacobian @ jacobian.T + (damping**2) * identity_task
          jacobian_pinv = np.linalg.solve(task_matrix, jacobian).T
      except np.linalg.LinAlgError:
        break
      if use_orientation:
        # A damped pseudo-inverse leaves a small artificial "null space" even
        # when the pose rows fully determine a finger.  Feeding the continuity
        # objective through that leakage biases the solution away from an
        # otherwise reachable pose.  Use an SVD projector for posture only;
        # the task update itself remains damped for robustness.
        nullspace = identity_joint - np.linalg.pinv(
          jacobian, rcond=1.0e-6
        ) @ jacobian
      else:
        nullspace = identity_joint - jacobian_pinv @ jacobian
      weighted_errors = position_scales[:, None] * errors
      if use_orientation:
        weighted_errors = np.concatenate(
          (
            weighted_errors.reshape(-1),
            (orientation_scales[:, None] * orientation_errors).reshape(-1),
          )
        )
      else:
        weighted_errors = weighted_errors.reshape(-1)
      delta_q = jacobian_pinv @ weighted_errors
      delta_q += posture_weight * nullspace @ (q_reference - q)
      delta_q = np.clip(delta_q, -step_limit, step_limit)

      current_cost = task_cost
      accepted = False
      # Backtracking keeps the DLS update stable around singular/limited
      # postures and gives a useful best-effort result for unreachable policy
      # predictions instead of allowing an oscillating final iterate.
      for scale in (1.0, 0.5, 0.25, 0.125, 0.0625):
        candidate = np.clip(q + scale * delta_q, lower, upper)
        (
          candidate_errors,
          candidate_norms,
          candidate_orientation_errors,
          candidate_orientation_norms,
          candidate_cost,
        ) = evaluate(candidate)
        if candidate_cost <= current_cost + 1.0e-14:
          q = candidate
          errors = candidate_errors
          error_norms = candidate_norms
          orientation_errors = candidate_orientation_errors
          orientation_error_norms = candidate_orientation_norms
          task_cost = candidate_cost
          accepted = True
          break
      if not accepted:
        # Restore FK for the best known posture before reporting it.
        evaluate(best_q)
        break

      improvement = current_cost - candidate_cost
      if improvement <= max(1.0e-14, 1.0e-8 * current_cost):
        stalled_iterations += 1
      else:
        stalled_iterations = 0

      key = result_key(errors, error_norms, task_cost)
      if key < best_key:
        best_key = key
        best_q = q.copy()
        best_error_norms = error_norms.copy()
      maximum_error = float(np.max(error_norms))
      if maximum_error <= position_tolerance and not use_orientation:
        return HandIkResult(
          True,
          joint_names,
          q.copy(),
          error_norms.copy(),
          maximum_error,
          completed_iterations,
        )
      if (
        maximum_error <= position_tolerance
        and use_orientation
        and float(
          np.max(orientation_scales * orientation_error_norms, initial=0.0)
        )
        <= 1.0e-7
      ):
        return HandIkResult(
          True,
          joint_names,
          q.copy(),
          error_norms.copy(),
          maximum_error,
          completed_iterations,
        )
      if stalled_iterations >= 8:
        break

    # Thumb FK has two strongly curved flexion joints because joint5 also
    # drives passive joint6.  A long target jump can put the previous posture
    # on the wrong side of that chain's local minimum.  Retry only the thumb
    # from two deterministic, anatomically useful postures; the other fingers
    # retain their already optimized values and temporal continuity.
    if best_error_norms[0] > position_tolerance:
      thumb_indices = np.arange(4)
      thumb_lower = lower[thumb_indices]
      thumb_upper = upper[thumb_indices]
      thumb_midpoint = 0.5 * (thumb_lower + thumb_upper)
      thumb_power_grasp = np.clip(
        np.deg2rad(np.array([10.0, 35.0, 10.0, 30.0])),
        thumb_lower,
        thumb_upper,
      )
      use_thumb_orientation = use_orientation and orientation_scales[0] > 0.0
      thumb_task_dimensions = 6 if use_thumb_orientation else 3
      thumb_identity_task = np.eye(thumb_task_dimensions)
      thumb_identity_joint = np.eye(4)
      fallback_iterations = max(12, min(60, max_iterations // 2))

      for thumb_seed in (thumb_midpoint, thumb_power_grasp):
        q = best_q.copy()
        q[thumb_indices] = thumb_seed
        (
          errors,
          error_norms,
          orientation_errors,
          orientation_error_norms,
          task_cost,
        ) = evaluate(q)
        for _ in range(fallback_iterations):
          completed_iterations += 1
          jacobian_position.fill(0.0)
          jacobian_rotation.fill(0.0)
          mujoco.mj_jacSite(
            self.model,
            self.ik_data,
            jacobian_position,
            jacobian_rotation if use_thumb_orientation else None,
            int(site_ids[0]),
          )
          thumb_jacobian = np.zeros((thumb_task_dimensions, 4), dtype=float)
          thumb_jacobian[:3] = (
            position_scales[0]
            * jacobian_position[:, dof_addresses[thumb_indices]]
          )
          thumb_jacobian[:3, thumb_joint5_index] += (
            position_scales[0] * jacobian_position[:, thumb_joint6_dof]
          )
          thumb_error = position_scales[0] * errors[0]
          if use_thumb_orientation:
            thumb_jacobian[3:] = (
              orientation_scales[0]
              * jacobian_rotation[:, dof_addresses[thumb_indices]]
            )
            thumb_jacobian[3:, thumb_joint5_index] += (
              orientation_scales[0] * jacobian_rotation[:, thumb_joint6_dof]
            )
            thumb_error = np.concatenate(
              (thumb_error, orientation_scales[0] * orientation_errors[0])
            )
          try:
            if use_thumb_orientation:
              thumb_pinv = np.linalg.solve(
                thumb_jacobian.T @ thumb_jacobian
                + (damping**2) * thumb_identity_joint,
                thumb_jacobian.T,
              )
            else:
              thumb_task_matrix = (
                thumb_jacobian @ thumb_jacobian.T
                + (damping**2) * thumb_identity_task
              )
              thumb_pinv = np.linalg.solve(thumb_task_matrix, thumb_jacobian).T
          except np.linalg.LinAlgError:
            break
          if use_thumb_orientation:
            thumb_nullspace = thumb_identity_joint - np.linalg.pinv(
              thumb_jacobian, rcond=1.0e-6
            ) @ thumb_jacobian
          else:
            thumb_nullspace = thumb_identity_joint - thumb_pinv @ thumb_jacobian
          thumb_delta = thumb_pinv @ thumb_error
          thumb_delta += (
            posture_weight
            * thumb_nullspace
            @ (q_reference[thumb_indices] - q[thumb_indices])
          )
          thumb_delta = np.clip(thumb_delta, -step_limit, step_limit)
          current_thumb_cost = float(thumb_error @ thumb_error)
          accepted = False
          for scale in (1.0, 0.5, 0.25, 0.125, 0.0625):
            candidate = q.copy()
            candidate[thumb_indices] = np.clip(
              q[thumb_indices] + scale * thumb_delta,
              thumb_lower,
              thumb_upper,
            )
            (
              candidate_errors,
              candidate_norms,
              candidate_orientation_errors,
              candidate_orientation_norms,
              candidate_cost,
            ) = evaluate(candidate)
            candidate_thumb_error = position_scales[0] * candidate_errors[0]
            if use_thumb_orientation:
              candidate_thumb_error = np.concatenate(
                (
                  candidate_thumb_error,
                  orientation_scales[0] * candidate_orientation_errors[0],
                )
              )
            candidate_thumb_cost = float(
              candidate_thumb_error @ candidate_thumb_error
            )
            if candidate_thumb_cost <= current_thumb_cost + 1.0e-14:
              q = candidate
              errors = candidate_errors
              error_norms = candidate_norms
              orientation_errors = candidate_orientation_errors
              orientation_error_norms = candidate_orientation_norms
              task_cost = candidate_cost
              accepted = True
              break
          if not accepted:
            break

          key = result_key(errors, error_norms, task_cost)
          if key < best_key:
            best_key = key
            best_q = q.copy()
            best_error_norms = error_norms.copy()
          maximum_error = float(np.max(error_norms))
          if maximum_error <= position_tolerance and not use_orientation:
            return HandIkResult(
              True,
              joint_names,
              q.copy(),
              error_norms.copy(),
              maximum_error,
              completed_iterations,
            )

    evaluate(best_q)
    best_position_error = float(np.max(best_error_norms))
    return HandIkResult(
      best_position_error <= position_tolerance,
      joint_names,
      best_q,
      best_error_norms,
      best_position_error,
      completed_iterations,
    )

  def set_fingertip_targets(
    self,
    side: str,
    target_positions: Iterable[Iterable[float]],
    *,
    target_rotations: Iterable[Iterable[Iterable[float]]] | None = None,
    seed: Iterable[float] | None = None,
    arm_joint_positions: Iterable[float] | None = None,
    apply_best_effort: bool = True,
    max_iterations: int = 100,
    position_tolerance: float = 1.5e-3,
    position_weights: float | Iterable[float] = 1.0,
    orientation_weights: float | Iterable[float] = 0.01,
    posture_weight: float = 0.01,
    damping: float = 0.003,
    step_limit: float = 0.2,
  ) -> HandIkResult:
    """Solve and command hand joints for five world-frame fingertip targets.

    Set ``apply_best_effort=False`` when an unreachable policy target should
    leave the previous hand command untouched.  With the default, the closest
    posture found is applied even when ``result.success`` is false.
    """
    result = self.solve_hand_ik(
      side,
      target_positions,
      target_rotations=target_rotations,
      seed=seed,
      arm_joint_positions=arm_joint_positions,
      max_iterations=max_iterations,
      position_tolerance=position_tolerance,
      position_weights=position_weights,
      orientation_weights=orientation_weights,
      posture_weight=posture_weight,
      damping=damping,
      step_limit=step_limit,
    )
    if result.success or apply_best_effort:
      accepted = self.set_hand_joint_targets(result.joint_names, result.joint_positions)
      if accepted != len(result.joint_names):
        raise RuntimeError(
          f"hand IK produced {len(result.joint_names)} targets, accepted {accepted}"
        )
    return result

  def set_hand_joint_targets(
    self, names: Iterable[str], positions: Iterable[float]
  ) -> int:
    accepted = 0
    for name, value in zip(names, positions, strict=False):
      if not np.isfinite(value):
        continue
      if name.startswith("hand_l_"):
        side = "left"
      elif name.startswith("hand_r_"):
        side = "right"
      else:
        continue
      if name not in self._hand_targets[side]:
        # thumb_joint4 is intentionally unactuated and thumb_joint6 is
        # constrained 1:1 to joint5 in the MJCF.
        continue
      joint_id = self._joint_id[name]
      lower, upper = self.model.jnt_range[joint_id]
      self._hand_targets[side][name] = float(np.clip(value, lower, upper))
      accepted += 1
    return accepted

  def _before_physics_step(self) -> None:
    """Optional control-only extension after servo commands have been built."""

  def step(self, steps: int = 1) -> None:
    for _ in range(max(1, int(steps))):
      # Feed the model bias forces (gravity plus velocity-dependent
      # terms) back into the actuated arm DoFs.  This is the MuJoCo
      # equivalent of the gravity feed-forward used by a real arm
      # controller; without it a finite-gain position servo visibly
      # sags under the KaiHand mass even while holding a fixed target.
      self.data.qfrc_applied.fill(0.0)
      for side in SIDES:
        dofs = self._arm_dofs[side]
        self.data.qfrc_applied[dofs] = self.data.qfrc_bias[dofs]
      max_delta = self.arm_speed_limit * self.timestep
      for side in SIDES:
        error = self._arm_goal[side] - self._arm_command[side]
        self._arm_command[side] += np.clip(error, -max_delta, max_delta)
        self.data.ctrl[self._arm_actuators[side]] = self._arm_command[side]
        for name, target in self._hand_targets[side].items():
          qpos = self.data.qpos[self._qpos_address[name]]
          velocity_target = self.hand_position_gain * (target - qpos)
          actuator_id = self._hand_actuators[side][name]
          control_range = self.model.actuator_ctrlrange[actuator_id]
          self.data.ctrl[actuator_id] = np.clip(
            velocity_target, control_range[0], control_range[1]
          )
      self._before_physics_step()
      mujoco.mj_step(self.model, self.data)

  def joint_state(self) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    position = np.array(
      [self.data.qpos[self._qpos_address[name]] for name in self.joint_names]
    )
    velocity = np.array(
      [self.data.qvel[self._qvel_address[name]] for name in self.joint_names]
    )
    return self.joint_names, position, velocity

  @property
  def contact_count(self) -> int:
    return int(self.data.ncon)
