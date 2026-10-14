"""Physically consistent reset state for an already installed USB plug."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import mujoco
import numpy as np

from kaihand_tactile_env.shared.simulation import ArmHandSimulation

from . import config

_WRAP_JOINT = "right_arm_joint5"


def _unwrap_home_wrist_coordinate(simulation: ArmHandSimulation) -> dict[str, Any]:
  """Choose the equivalent positive J5 branch without changing robot pose."""
  model, data = simulation.model, simulation.data
  joint = model.joint(_WRAP_JOINT)
  qpos = int(joint.qposadr[0])
  actuator = model.actuator(_WRAP_JOINT).id
  right_index = tuple(simulation._arm_joint_ids["right"]).index(joint.id)
  original = float(data.qpos[qpos])
  wrapped = original + 2 * np.pi if original < 0.0 else original
  model.jnt_range[joint.id, 1] = max(model.jnt_range[joint.id, 1], 2 * np.pi)
  model.actuator_ctrlrange[actuator, 1] = max(
    model.actuator_ctrlrange[actuator, 1], 2 * np.pi
  )
  data.qpos[qpos] = wrapped
  simulation._arm_goal["right"][right_index] = wrapped
  simulation._arm_command["right"][right_index] = wrapped
  data.ctrl[actuator] = wrapped
  return {
    "joint": _WRAP_JOINT,
    "original_coordinate_rad": original,
    "wrapped_coordinate_rad": wrapped,
    "physical_pose_changed": False,
  }


def initialize_installed_state(simulation: ArmHandSimulation) -> dict[str, Any]:
  """Load the seated plug and compressed passive shoes before the first step.

  This is an episode initial condition, not an execution primitive.  It is
  deliberately rejected after simulation time advances.  No socket mechanical
  parameter, equality, collision mask, or external wrench is changed.  As in
  the insertion expert, the periodic wrist coordinate is put on an equivalent
  short-path branch and its scalar coordinate range is extended accordingly.
  """
  if simulation.scene != config.SCENE_NAME:
    raise ValueError("USB unplug initialization requires scene='usb-unplug'")
  if float(simulation.data.time) != 0.0:
    raise ValueError(
      "USB unplug initialization is only allowed immediately after reset"
    )

  model, data = simulation.model, simulation.data
  plug_joint = model.joint("usb_plug_freejoint")
  plug_qpos = int(plug_joint.qposadr[0])
  plug_dof = int(plug_joint.dofadr[0])
  mouth = model.site("usb_socket_mouth").id
  socket = model.body("usb_socket").id
  socket_rotation = data.site_xmat[mouth].reshape(3, 3).copy()
  plug_position = data.site_xpos[mouth] + socket_rotation @ (
    np.array([config.INITIAL_INSERTION_DEPTH_M, 0.0, 0.0]) - config.PLUG_TIP_LOCAL_M
  )
  plug_quaternion = data.xquat[socket].copy()

  data.qpos[plug_qpos : plug_qpos + 3] = plug_position
  data.qpos[plug_qpos + 3 : plug_qpos + 7] = plug_quaternion
  data.qvel[plug_dof : plug_dof + 6] = 0.0

  spring_positions: dict[str, float] = {}
  for side in ("top", "bottom"):
    joint = model.joint(f"usb_socket_spring_{side}_slide")
    qpos = int(joint.qposadr[0])
    dof = int(joint.dofadr[0])
    lower, upper = model.jnt_range[joint.id]
    displacement = float(config.INITIAL_SPRING_DISPLACEMENT_M)
    if not lower <= displacement <= upper:
      raise ValueError(f"initial {side} spring displacement is outside its joint range")
    data.qpos[qpos] = displacement
    data.qvel[dof] = 0.0
    spring_positions[side] = displacement

  wrist_wrap = _unwrap_home_wrist_coordinate(simulation)

  # Reset any stale assistance if a caller reuses an MjData at time zero.
  data.xfrc_applied[model.body("usb_plug").id] = 0.0
  data.qfrc_applied[plug_dof : plug_dof + 6] = 0.0
  mujoco.mj_forward(model, data)
  return {
    "schema": "usb_unplug_installed_reset_v1",
    "time_s": float(data.time),
    "plug_pose_wxyz": data.qpos[plug_qpos : plug_qpos + 7].tolist(),
    "insertion_depth_m": config.INITIAL_INSERTION_DEPTH_M,
    "spring_slide_qpos_m": spring_positions,
    "wrist_coordinate_wrap": wrist_wrap,
    "velocity_zeroed": True,
    "socket_mechanics_changed": False,
    "object_assistance_applied": False,
  }


class UsbUnplugSimulation(ArmHandSimulation):
  """Arm/hand simulation whose public reset always installs a valid task state."""

  def __init__(
    self,
    *,
    seed: int = 0,
    add_genesis_probes: bool = False,
    **kwargs,
  ) -> None:
    self._usb_unplug_seed = int(seed)
    self.initialization: dict[str, Any] | None = None
    self.physics_observer: Callable[[UsbUnplugSimulation], None] | None = None
    super().__init__(
      scene=config.SCENE_NAME,
      add_genesis_probes=add_genesis_probes,
      **kwargs,
    )
    # Base construction dispatches through this subclass's reset once; the
    # requested seed is already available through ``_usb_unplug_seed``.

  def reset(self, *, seed: int | None = None, **kwargs) -> None:
    if seed is None:
      seed = getattr(self, "_usb_unplug_seed", 0)
    self._usb_unplug_seed = int(seed)
    # Independent seated state has no free tabletop jitter at reset.
    kwargs.pop("object_xy_jitter", None)
    kwargs.pop("object_yaw_jitter", None)
    kwargs.pop("randomized_objects", None)
    super().reset(seed=seed, randomized_objects=(), **kwargs)
    self.initialization = initialize_installed_state(self)

  def step(self, steps: int = 1) -> None:
    """Advance and publish each post-forward physics state independently."""
    for _ in range(max(1, int(steps))):
      super().step(1)
      mujoco.mj_forward(self.model, self.data)
      if self.physics_observer is not None:
        self.physics_observer(self)


Simulation = UsbUnplugSimulation

__all__ = ["Simulation", "UsbUnplugSimulation", "initialize_installed_state"]
