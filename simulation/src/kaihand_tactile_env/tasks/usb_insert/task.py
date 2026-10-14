"""Read-only geometric insertion and contact measurements for the USB scene.

The automatic robot policy is defined separately in execution.py. Success here
denotes a stable mechanical fit, not electrical connectivity. Only plug/socket
forces are included in the insertion load.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product

import mujoco
import numpy as np

from kaihand_tactile_env.shared.simulation import ArmHandSimulation

from . import config


@dataclass(frozen=True)
class UsbInsertionState:
  timestamp: float
  insertion_depth_m: float
  lateral_error_m: tuple[float, float]
  orientation_error_rad: float
  shell_fits_aperture: bool
  socket_contact_count: int
  socket_force_on_plug_n: tuple[float, float, float]
  axial_resistance_n: float
  socket_normal_load_n: float
  maximum_socket_penetration_m: float
  linear_speed_m_s: float
  angular_speed_rad_s: float
  seated: bool
  axial_speed_m_s: float
  backstop_contact_count: int
  backstop_force_on_plug_n: tuple[float, float, float]
  backstop_axial_resistance_n: float
  backstop_normal_load_n: float
  backstop_contact: bool
  spring_contact_count: int
  spring_normal_load_n: float
  spring_axial_resistance_n: float
  wall_normal_load_n: float
  tongue_normal_load_n: float
  bottom_out_confirmed: bool = False
  stable_duration_s: float = 0.0
  success: bool = False


def _near_bottom(state: UsbInsertionState) -> bool:
  """Geometric validity required to retain an observed bottom-out history."""
  return bool(
    config.BACKSTOP_DEPTH_M - config.SEATED_DEPTH_TOLERANCE_M
    <= state.insertion_depth_m
    <= config.BACKSTOP_DEPTH_M + config.MAX_SOCKET_PENETRATION_M
    and state.shell_fits_aperture
    and state.orientation_error_rad <= np.deg2rad(3.0)
    and state.maximum_socket_penetration_m <= config.MAX_SOCKET_PENETRATION_M
  )


def _stationary(state: UsbInsertionState) -> bool:
  return bool(
    state.linear_speed_m_s <= config.SEATED_LINEAR_SPEED_M_S
    and state.angular_speed_rad_s <= 0.05
  )


def _with_seating(state: UsbInsertionState, confirmed: bool) -> UsbInsertionState:
  near_bottom = _near_bottom(state)
  retained = bool(confirmed and near_bottom)
  return replace(
    state,
    bottom_out_confirmed=retained,
    seated=bool(
      near_bottom and _stationary(state) and (state.backstop_contact or retained)
    ),
  )


class UsbInsertionMonitor:
  """Require current bottom support or retained, measured bottom-out history.

  Spring friction alone or passing the old 11 mm depth threshold cannot
  establish seating. A sustained 0.6 N / 150 ms bottom contact is retained
  while the plug stays in the aligned near-bottom region. Elastic unloading
  may transfer its weight to spring friction without undoing that evidence.

  Call ``update`` at every physics step. A time rewind or an observation gap
  greater than 1.5 physics steps clears both timers and bottom history:
  sparse snapshots do not prove continuous seating. ``reset`` is required after external state
  changes that do not rewind simulation time.
  """

  def __init__(self, simulation: ArmHandSimulation) -> None:
    if simulation.scene != config.SCENE_NAME:
      raise ValueError("UsbInsertionMonitor requires scene='usb-insert'")
    self.simulation = simulation
    model = simulation.model
    self._plug_id = model.body("usb_plug").id
    self._mouth_id = model.site("usb_socket_mouth").id
    self._tip_id = model.site("usb_plug_tip").id
    socket_id = model.body("usb_socket").id
    # Include descendants so a later fixture hierarchy remains measurable.
    socket_bodies = {socket_id}
    for body_id in range(model.nbody):
      if int(model.body_parentid[body_id]) in socket_bodies:
        socket_bodies.add(body_id)
    self._socket_geoms = {
      i for i in range(model.ngeom) if int(model.geom_bodyid[i]) in socket_bodies
    }
    self._plug_geoms = {
      i for i in range(model.ngeom) if int(model.geom_bodyid[i]) == self._plug_id
    }
    self._contact_categories = {}
    for geom_id in self._socket_geoms:
      name = model.geom(geom_id).name
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
      self._contact_categories[geom_id] = category
    self.reset()

  def reset(self) -> None:
    self._last_time: float | None = None
    self._seated_since: float | None = None
    self._bottom_out_hold_s = 0.0
    self._bottom_out_confirmed = False

  def measure(self) -> UsbInsertionState:
    """Synchronize derived state and measure without advancing physics time."""
    sim = self.simulation
    model, data = sim.model, sim.data
    # mj_step advances qpos/time after computing site/contact caches. Refresh
    # only this task's data so depth, velocity and force share the timestamp.
    mujoco.mj_forward(model, data)
    rotation = data.site_xmat[self._mouth_id].reshape(3, 3)
    tip = rotation.T @ (data.site_xpos[self._tip_id] - data.site_xpos[self._mouth_id])
    relative_rotation = rotation.T @ data.xmat[self._plug_id].reshape(3, 3)
    angle = float(np.arccos(np.clip((np.trace(relative_rotation) - 1) / 2, -1, 1)))
    # Bound the inserted portion of the shell, not just its tip centre.
    # At valid seating, both end cross sections must fit the rectangular bore.
    corners = np.array(
      [
        [x, y, z]
        for x, y, z in product(
          (-min(max(float(tip[0]), 0.0), config.PLUG_SHELL_LENGTH_M), 0.0),
          (-config.PLUG_SHELL_HALF_WIDTH_M, config.PLUG_SHELL_HALF_WIDTH_M),
          (-config.PLUG_SHELL_HALF_HEIGHT_M, config.PLUG_SHELL_HALF_HEIGHT_M),
        )
      ]
    )
    points = corners @ relative_rotation.T + tip
    fits = bool(
      np.max(np.abs(points[:, 1])) <= config.SOCKET_HALF_WIDTH_M + 1.0e-6
      and np.max(np.abs(points[:, 2])) <= config.SOCKET_HALF_HEIGHT_M + 1.0e-6
    )
    force_world = np.zeros(3)
    normal_load = 0.0
    penetration = 0.0
    count = 0
    categories = ("backstop", "spring", "wall", "tongue", "other")
    category_counts = dict.fromkeys(categories, 0)
    category_loads = dict.fromkeys(categories, 0.0)
    category_forces_world = {name: np.zeros(3) for name in categories}
    wrench = np.zeros(6)
    for contact_id in range(data.ncon):
      contact = data.contact[contact_id]
      geom1, geom2 = int(contact.geom1), int(contact.geom2)
      if geom1 in self._plug_geoms and geom2 in self._socket_geoms:
        sign = -1.0
        socket_geom = geom2
      elif geom2 in self._plug_geoms and geom1 in self._socket_geoms:
        sign = 1.0
        socket_geom = geom1
      else:
        continue
      count += 1
      mujoco.mj_contactForce(model, data, contact_id, wrench)
      contact_force = sign * (contact.frame.reshape(3, 3).T @ wrench[:3])
      contact_load = abs(float(wrench[0]))
      force_world += contact_force
      normal_load += contact_load
      category = self._contact_categories[socket_geom]
      category_counts[category] += 1
      category_loads[category] += contact_load
      category_forces_world[category] += contact_force
      penetration = max(penetration, -float(contact.dist))
    force_socket = rotation.T @ force_world
    backstop_force = rotation.T @ category_forces_world["backstop"]
    backstop_resistance = max(0.0, -float(backstop_force[0]))
    spring_force = rotation.T @ category_forces_world["spring"]
    backstop_contact = bool(
      category_counts["backstop"] > 0
      and category_loads["backstop"] >= config.SEATED_BACKSTOP_MIN_LOAD_N
      and backstop_resistance >= config.SEATED_BACKSTOP_MIN_LOAD_N
    )
    twist = sim.object_twist("usb_plug")
    linear_speed = float(np.linalg.norm(twist[:3]))
    angular_speed = float(np.linalg.norm(twist[3:]))
    axial_speed = float((rotation.T @ twist[:3])[0])
    state = UsbInsertionState(
      timestamp=float(data.time),
      insertion_depth_m=float(tip[0]),
      lateral_error_m=(float(tip[1]), float(tip[2])),
      orientation_error_rad=angle,
      shell_fits_aperture=fits,
      socket_contact_count=count,
      socket_force_on_plug_n=tuple(float(x) for x in force_socket),
      axial_resistance_n=max(0.0, -float(force_socket[0])),
      socket_normal_load_n=normal_load,
      maximum_socket_penetration_m=penetration,
      linear_speed_m_s=linear_speed,
      angular_speed_rad_s=angular_speed,
      seated=False,
      axial_speed_m_s=axial_speed,
      backstop_contact_count=category_counts["backstop"],
      backstop_force_on_plug_n=tuple(float(x) for x in backstop_force),
      backstop_axial_resistance_n=backstop_resistance,
      backstop_normal_load_n=category_loads["backstop"],
      backstop_contact=backstop_contact,
      spring_contact_count=category_counts["spring"],
      spring_normal_load_n=category_loads["spring"],
      spring_axial_resistance_n=max(0.0, -float(spring_force[0])),
      wall_normal_load_n=category_loads["wall"],
      tongue_normal_load_n=category_loads["tongue"],
    )
    # Measuring alone never acquires contact history or advances its timer.
    return _with_seating(state, self._bottom_out_confirmed)

  def update(self) -> UsbInsertionState:
    state = self.measure()
    elapsed = 0.0
    if self._last_time is not None:
      elapsed = state.timestamp - self._last_time
      if elapsed < -1.0e-12 or elapsed > 1.5 * self.simulation.timestep:
        self._seated_since = None
        self._bottom_out_hold_s = 0.0
        self._bottom_out_confirmed = False
        elapsed = 0.0
    self._last_time = state.timestamp
    if not _near_bottom(state):
      self._bottom_out_hold_s = 0.0
      self._bottom_out_confirmed = False
    qualified_bottom = bool(
      _near_bottom(state)
      and _stationary(state)
      and state.backstop_contact
      and state.backstop_axial_resistance_n >= config.BOTTOM_OUT_MIN_FORCE_N
    )
    if qualified_bottom:
      # Repeated reads at the same timestamp provide no new physical evidence.
      if elapsed > 1e-12:
        self._bottom_out_hold_s += elapsed
      if self._bottom_out_hold_s >= config.BOTTOM_OUT_HOLD_S - 1e-12:
        self._bottom_out_confirmed = True
    else:
      self._bottom_out_hold_s = 0.0
    state = _with_seating(state, self._bottom_out_confirmed)
    if not state.seated:
      self._seated_since = None
    elif self._seated_since is None:
      self._seated_since = state.timestamp
    duration = (
      0.0 if self._seated_since is None else state.timestamp - self._seated_since
    )
    return replace(
      state,
      stable_duration_s=duration,
      success=duration >= config.SEATED_DWELL_S - 1e-12,
    )
