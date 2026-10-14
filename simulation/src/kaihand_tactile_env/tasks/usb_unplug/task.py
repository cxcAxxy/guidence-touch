"""Read-only physical state and ordered completion monitor for USB unplugging."""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product

import mujoco
import numpy as np

from kaihand_tactile_env.shared.simulation import ArmHandSimulation

from . import config


@dataclass(frozen=True)
class UsbUnplugState:
  timestamp: float
  insertion_depth_m: float
  lateral_error_m: tuple[float, float]
  orientation_error_rad: float
  shell_fits_aperture: bool
  socket_contact_count: int
  socket_force_on_plug_n: tuple[float, float, float]
  signed_axial_force_n: float
  insertion_resistance_n: float
  withdrawal_resistance_n: float
  socket_normal_load_n: float
  maximum_socket_penetration_m: float
  backstop_contact_count: int
  spring_contact_count: int
  wall_contact_count: int
  tongue_contact_count: int
  backstop_normal_load_n: float
  spring_normal_load_n: float
  wall_normal_load_n: float
  tongue_normal_load_n: float
  spring_slide_qpos_m: tuple[float, float]
  linear_speed_m_s: float
  angular_speed_rad_s: float
  axial_speed_m_s: float
  grip_force_n: tuple[float, float]
  bilateral_grasp: bool
  right_hand_contact_count: int
  table_contact_count: int
  table_normal_load_n: float
  home_xy_error_m: float
  home_height_m: float
  home_orientation_error_rad: float
  initially_seated_now: bool
  fully_clear_now: bool
  at_home_now: bool
  wrist_object_distance_m: float
  initial_stable_duration_s: float = 0.0
  grasp_stable_duration_s: float = 0.0
  clear_duration_s: float = 0.0
  home_stable_duration_s: float = 0.0
  release_duration_s: float = 0.0
  retreat_duration_s: float = 0.0
  initially_seated_verified: bool = False
  grasp_verified: bool = False
  fully_extracted: bool = False
  placed_on_table: bool = False
  released: bool = False
  hand_retreated: bool = False
  success: bool = False


def _angle_between(first: np.ndarray, second: np.ndarray) -> float:
  relative = first.T @ second
  return float(np.arccos(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)))


class UsbUnplugMonitor:
  """Measure contacts and require ordered, continuous physical milestones."""

  def __init__(self, simulation: ArmHandSimulation) -> None:
    if simulation.scene != config.SCENE_NAME:
      raise ValueError("UsbUnplugMonitor requires scene='usb-unplug'")
    self.simulation = simulation
    model = simulation.model
    self._plug_body = model.body("usb_plug").id
    self._mouth = model.site("usb_socket_mouth").id
    self._tip = model.site("usb_plug_tip").id
    socket = model.body("usb_socket").id
    socket_bodies = {socket}
    for body in range(model.nbody):
      if int(model.body_parentid[body]) in socket_bodies:
        socket_bodies.add(body)
    self._socket_geoms = {
      geom
      for geom in range(model.ngeom)
      if int(model.geom_bodyid[geom]) in socket_bodies
    }
    self._plug_geoms = {
      geom
      for geom in range(model.ngeom)
      if int(model.geom_bodyid[geom]) == self._plug_body
    }
    self._right_hand_bodies = {
      body
      for body in range(model.nbody)
      if (model.body(body).name or "").startswith("hand_r_")
    }
    self._table_geom = model.geom("tabletop").id
    self._grip_pads = tuple(
      model.geom(name).id
      for name in (
        "hand_r_thumb_link6_tactile_pad_col",
        "hand_r_index_link4_tactile_pad_col",
      )
    )
    self._spring_qpos = tuple(
      int(model.joint(f"usb_socket_spring_{side}_slide").qposadr[0])
      for side in ("top", "bottom")
    )
    self._categories: dict[int, str] = {}
    for geom in self._socket_geoms:
      name = model.geom(geom).name or ""
      if name == "usb_socket_backstop":
        category = "backstop"
      elif name.startswith("usb_socket_spring_"):
        category = "spring"
      elif name.startswith("usb_socket_wall_"):
        category = "wall"
      elif name == "usb_socket_tongue":
        category = "tongue"
      else:
        category = "other"
      self._categories[geom] = category
    home_matrix = np.empty(9)
    mujoco.mju_quat2Mat(home_matrix, config.HOME_QUATERNION_WXYZ)
    self._home_rotation = home_matrix.reshape(3, 3)
    self.reset()

  def reset(self) -> None:
    self._last_time: float | None = None
    self._initial_s = self._grasp_s = self._clear_s = 0.0
    self._home_s = self._release_s = self._retreat_s = 0.0
    self._initial_verified = self._grasp_verified = False
    self._fully_extracted = self._placed = self._released = self._retreated = False

  def measure(self) -> UsbUnplugState:
    simulation = self.simulation
    model, data = simulation.model, simulation.data
    mujoco.mj_forward(model, data)
    socket_rotation = data.site_xmat[self._mouth].reshape(3, 3)
    tip = socket_rotation.T @ (data.site_xpos[self._tip] - data.site_xpos[self._mouth])
    plug_rotation = data.xmat[self._plug_body].reshape(3, 3)
    relative_rotation = socket_rotation.T @ plug_rotation
    orientation_error = float(
      np.arccos(np.clip((np.trace(relative_rotation) - 1.0) / 2.0, -1.0, 1.0))
    )
    inserted_length = min(max(float(tip[0]), 0.0), config.PLUG_SHELL_LENGTH_M)
    corners = np.array(
      [
        [x, y, z]
        for x, y, z in product(
          (-inserted_length, 0.0),
          (-config.PLUG_SHELL_HALF_WIDTH_M, config.PLUG_SHELL_HALF_WIDTH_M),
          (-config.PLUG_SHELL_HALF_HEIGHT_M, config.PLUG_SHELL_HALF_HEIGHT_M),
        )
      ]
    )
    points = corners @ relative_rotation.T + tip
    fits = bool(
      np.max(np.abs(points[:, 1])) <= config.SOCKET_HALF_WIDTH_M + 1e-6
      and np.max(np.abs(points[:, 2])) <= config.SOCKET_HALF_HEIGHT_M + 1e-6
    )

    category_counts = dict.fromkeys(
      ("backstop", "spring", "wall", "tongue", "other"), 0
    )
    category_loads = dict.fromkeys(category_counts, 0.0)
    socket_force_world = np.zeros(3)
    socket_load = penetration = table_load = 0.0
    grip = np.zeros(2)
    socket_count = hand_count = table_count = 0
    wrench = np.empty(6)
    for contact_id in range(data.ncon):
      contact = data.contact[contact_id]
      geom1, geom2 = int(contact.geom1), int(contact.geom2)
      body1, body2 = int(model.geom_bodyid[geom1]), int(model.geom_bodyid[geom2])
      plug_is_first = geom1 in self._plug_geoms
      plug_is_second = geom2 in self._plug_geoms
      if not (plug_is_first or plug_is_second):
        continue
      other = geom2 if plug_is_first else geom1
      other_body = body2 if plug_is_first else body1
      mujoco.mj_contactForce(model, data, contact_id, wrench)
      load = abs(float(wrench[0]))
      # mj_contactForce reports the wrench on geom2 in the contact frame.
      sign = 1.0 if plug_is_second else -1.0
      force_on_plug = sign * (contact.frame.reshape(3, 3).T @ wrench[:3])
      if other in self._socket_geoms:
        socket_count += 1
        socket_load += load
        socket_force_world += force_on_plug
        penetration = max(penetration, -float(contact.dist))
        category = self._categories[other]
        category_counts[category] += 1
        category_loads[category] += load
      if other_body in self._right_hand_bodies:
        hand_count += 1
      if other == self._table_geom:
        table_count += 1
        table_load += load
      for index, pad in enumerate(self._grip_pads):
        if other == pad:
          grip[index] += load

    force_socket = socket_rotation.T @ socket_force_world
    signed_axial = float(force_socket[0])
    twist = simulation.object_twist("usb_plug")
    local_velocity = socket_rotation.T @ twist[:3]
    position = data.xpos[self._plug_body]
    home_xy_error = float(np.linalg.norm(position[:2] - config.HOME_POSITION_M[:2]))
    home_orientation_error = _angle_between(self._home_rotation, plug_rotation)
    bilateral = bool(np.min(grip) >= config.GRASP_MIN_FORCE_N)
    initial_now = bool(
      abs(float(tip[0]) - config.INITIAL_INSERTION_DEPTH_M)
      <= config.INITIAL_DEPTH_TOLERANCE_M
      and np.linalg.norm(tip[1:]) <= config.INITIAL_LATERAL_TOLERANCE_M
      and orientation_error <= config.INITIAL_ORIENTATION_TOLERANCE_RAD
      and penetration <= config.MAX_SOCKET_PENETRATION_M
      and np.linalg.norm(twist[:3]) <= config.HOME_LINEAR_SPEED_M_S
      and np.linalg.norm(twist[3:]) <= config.HOME_ANGULAR_SPEED_RAD_S
      and category_counts["spring"] > 0
    )
    fully_clear_now = bool(
      float(tip[0]) <= config.FULL_CLEAR_DEPTH_M and socket_count == 0
    )
    at_home_now = bool(
      home_xy_error <= config.HOME_XY_TOLERANCE_M
      and config.HOME_HEIGHT_RANGE_M[0]
      <= float(position[2])
      <= config.HOME_HEIGHT_RANGE_M[1]
      and home_orientation_error <= config.HOME_ORIENTATION_TOLERANCE_RAD
      and np.linalg.norm(twist[:3]) <= config.HOME_LINEAR_SPEED_M_S
      and np.linalg.norm(twist[3:]) <= config.HOME_ANGULAR_SPEED_RAD_S
      and table_count > 0
      and socket_count == 0
    )
    wrist = simulation.current_pose_matrix("right")[0]
    return UsbUnplugState(
      timestamp=float(data.time),
      insertion_depth_m=float(tip[0]),
      lateral_error_m=(float(tip[1]), float(tip[2])),
      orientation_error_rad=orientation_error,
      shell_fits_aperture=fits,
      socket_contact_count=socket_count,
      socket_force_on_plug_n=tuple(float(value) for value in force_socket),
      signed_axial_force_n=signed_axial,
      insertion_resistance_n=max(0.0, -signed_axial),
      withdrawal_resistance_n=max(0.0, signed_axial),
      socket_normal_load_n=socket_load,
      maximum_socket_penetration_m=penetration,
      backstop_contact_count=category_counts["backstop"],
      spring_contact_count=category_counts["spring"],
      wall_contact_count=category_counts["wall"],
      tongue_contact_count=category_counts["tongue"],
      backstop_normal_load_n=category_loads["backstop"],
      spring_normal_load_n=category_loads["spring"],
      wall_normal_load_n=category_loads["wall"],
      tongue_normal_load_n=category_loads["tongue"],
      spring_slide_qpos_m=tuple(float(data.qpos[index]) for index in self._spring_qpos),
      linear_speed_m_s=float(np.linalg.norm(twist[:3])),
      angular_speed_rad_s=float(np.linalg.norm(twist[3:])),
      axial_speed_m_s=float(local_velocity[0]),
      grip_force_n=tuple(float(value) for value in grip),
      bilateral_grasp=bilateral,
      right_hand_contact_count=hand_count,
      table_contact_count=table_count,
      table_normal_load_n=table_load,
      home_xy_error_m=home_xy_error,
      home_height_m=float(position[2]),
      home_orientation_error_rad=home_orientation_error,
      initially_seated_now=initial_now,
      fully_clear_now=fully_clear_now,
      at_home_now=at_home_now,
      wrist_object_distance_m=float(np.linalg.norm(wrist - position)),
      initial_stable_duration_s=self._initial_s,
      grasp_stable_duration_s=self._grasp_s,
      clear_duration_s=self._clear_s,
      home_stable_duration_s=self._home_s,
      release_duration_s=self._release_s,
      retreat_duration_s=self._retreat_s,
      initially_seated_verified=self._initial_verified,
      grasp_verified=self._grasp_verified,
      fully_extracted=self._fully_extracted,
      placed_on_table=self._placed,
      released=self._released,
      hand_retreated=self._retreated,
      success=bool(self._retreated and at_home_now and hand_count == 0),
    )

  def update(self) -> UsbUnplugState:
    state = self.measure()
    elapsed = 0.0 if self._last_time is None else state.timestamp - self._last_time
    discontinuity = bool(
      self._last_time is not None
      and (elapsed < -1e-12 or elapsed > 1.5 * self.simulation.timestep)
    )
    self._last_time = state.timestamp
    if discontinuity:
      self.reset()
      self._last_time = state.timestamp
      elapsed = 0.0
    elapsed = elapsed if elapsed > 1e-12 else 0.0

    if not self._initial_verified:
      self._initial_s = self._initial_s + elapsed if state.initially_seated_now else 0.0
      self._initial_verified = self._initial_s >= config.INITIAL_STABLE_DWELL_S - 1e-12
    if self._initial_verified and not self._grasp_verified:
      self._grasp_s = self._grasp_s + elapsed if state.bilateral_grasp else 0.0
      self._grasp_verified = self._grasp_s >= config.GRASP_STABLE_DWELL_S - 1e-12
    if self._grasp_verified and not self._fully_extracted:
      self._clear_s = self._clear_s + elapsed if state.fully_clear_now else 0.0
      self._fully_extracted = self._clear_s >= config.FULL_CLEAR_DWELL_S - 1e-12
    if self._fully_extracted and not self._placed:
      self._home_s = self._home_s + elapsed if state.at_home_now else 0.0
      self._placed = self._home_s >= config.HOME_STABLE_DWELL_S - 1e-12
    if self._placed and not self._released:
      no_hand = state.right_hand_contact_count == 0
      self._release_s = (
        self._release_s + elapsed if no_hand and state.at_home_now else 0.0
      )
      self._released = self._release_s >= config.RELEASE_DWELL_S - 1e-12
    if self._released and not self._retreated:
      clear = (
        state.right_hand_contact_count == 0
        and state.wrist_object_distance_m >= config.RETREAT_CLEARANCE_M
        and state.at_home_now
      )
      self._retreat_s = self._retreat_s + elapsed if clear else 0.0
      self._retreated = self._retreat_s >= config.RETREAT_DWELL_S - 1e-12
    return replace(
      state,
      initial_stable_duration_s=self._initial_s,
      grasp_stable_duration_s=self._grasp_s,
      clear_duration_s=self._clear_s,
      home_stable_duration_s=self._home_s,
      release_duration_s=self._release_s,
      retreat_duration_s=self._retreat_s,
      initially_seated_verified=self._initial_verified,
      grasp_verified=self._grasp_verified,
      fully_extracted=self._fully_extracted,
      placed_on_table=self._placed,
      released=self._released,
      hand_retreated=self._retreated,
      success=bool(
        self._retreated and state.at_home_now and state.right_hand_contact_count == 0
      ),
    )


__all__ = ["UsbUnplugMonitor", "UsbUnplugState"]
