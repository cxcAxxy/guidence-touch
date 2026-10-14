"""Geometric and solver-contact verification of a free DIMM in its socket.

The shared robot, cameras and tactile geometry are inherited unchanged.
Success describes stable mechanical seating, not electrical connectivity or
certified latch engagement. No task force or attachment is applied here.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product

import mujoco
import numpy as np

from ...shared.pickup_randomization import PickupPositionRandomization
from ...shared.simulation import ArmHandSimulation
from . import config


@dataclass(frozen=True)
class RamInstallationState:
  timestamp: float
  insertion_depth_m: float
  lateral_error_m: tuple[float, float]
  orientation_error_rad: float
  aperture_fits: bool
  socket_contact_count: int
  socket_normal_load_n: float
  axial_resistance_n: float
  backstop_load_n: float
  maximum_socket_penetration_m: float
  linear_speed_m_s: float
  angular_speed_rad_s: float
  seated: bool
  spring_normal_load_n: float = 0.0
  spring_axial_resistance_n: float = 0.0
  wall_normal_load_n: float = 0.0
  palm_down_cosine: float = 0.0
  bottom_out_duration_s: float = 0.0
  bottom_out_confirmed: bool = False
  stable_duration_s: float = 0.0
  success: bool = False


class RamInstallSimulation(ArmHandSimulation):
  def __init__(self, *, position_seed=None, **kwargs):
    self.position_seed = position_seed
    self._pickup_randomization = None
    super().__init__(scene=config.SCENE_NAME, **kwargs)
    # Task-local visual setting: moving shadow-map artifacts are particularly
    # visible in the close wrist view. Keep shared lights/cameras and shading,
    # but disable cast shadows consistently in recording and interactive views.
    self.model.light_castshadow[:] = False

  def reset(self, **kwargs):
    super().reset(**kwargs)
    if self._pickup_randomization is None:
      self._pickup_randomization = PickupPositionRandomization(
        self, "ram", support_body="ram_presentation_stand"
      )
    self.initial_position_randomization = self._pickup_randomization.apply(
      self, self.position_seed
    )


class RamInstallationMonitor:
  """Require aligned bottom contact continuously at every physics step."""

  def __init__(self, simulation: ArmHandSimulation):
    if simulation.scene != config.SCENE_NAME:
      raise ValueError("RamInstallationMonitor requires scene='install-ram'")
    self.sim = simulation
    model = simulation.model
    self._body = model.body("ram").id
    self._mouth = model.site("ram_socket_mouth").id
    self._bottom = model.site("ram_bottom").id
    self._stop = model.geom("ram_socket_backstop").id
    socket_bodies = {model.body("ram_socket").id}
    for body in range(model.nbody):
      if int(model.body_parentid[body]) in socket_bodies:
        socket_bodies.add(body)
    self._socket_geoms = {
      i for i in range(model.ngeom) if int(model.geom_bodyid[i]) in socket_bodies
    }
    self._ram_geoms = {
      i for i in range(model.ngeom) if int(model.geom_bodyid[i]) == self._body
    }
    self.reset()

  def reset(self):
    self._last_time = None
    self._stable_since = None
    self._bottom_hold_s = 0.0
    self._bottom_confirmed = False

  def measure(self) -> RamInstallationState:
    model, data = self.sim.model, self.sim.data
    # mj_step integrates after computing contact/FK caches. Recompute them so
    # all task measurements and the subsequent recorder use the same state.
    mujoco.mj_forward(model, data)
    frame = data.site_xmat[self._mouth].reshape(3, 3)
    bottom = frame.T @ (data.site_xpos[self._bottom] - data.site_xpos[self._mouth])
    relative = frame.T @ data.xmat[self._body].reshape(3, 3)
    angle = float(np.arccos(np.clip((np.trace(relative) - 1) / 2, -1, 1)))
    depth = -float(bottom[2])
    # Check the full width of the inserted edge, including tilt. A centre
    # point alone would allow one DIMM end to stand on top of the housing.
    corners = np.array(
      list(
        product(
          (-config.RAM_LENGTH_M / 2, config.RAM_LENGTH_M / 2),
          (-config.RAM_THICKNESS_M / 2, config.RAM_THICKNESS_M / 2),
          (0.0, min(max(depth, 0.0), config.TARGET_INSERTION_DEPTH_M)),
        )
      )
    )
    points = corners @ relative.T + bottom
    fits = bool(
      np.max(np.abs(points[:, 0])) <= config.SOCKET_INNER_LENGTH_M / 2 + 1e-6
      and np.max(np.abs(points[:, 1])) <= config.SOCKET_INNER_WIDTH_M / 2 + 1e-6
    )
    total = np.zeros(3)
    stop_force = np.zeros(3)
    spring_force = np.zeros(3)
    spring_load = wall_load = 0.0
    load = penetration = 0.0
    count = 0
    wrench = np.zeros(6)
    for i in range(data.ncon):
      contact = data.contact[i]
      a, b = int(contact.geom1), int(contact.geom2)
      if a in self._ram_geoms and b in self._socket_geoms:
        sign, socket_geom = -1, b
      elif b in self._ram_geoms and a in self._socket_geoms:
        sign, socket_geom = 1, a
      else:
        continue
      mujoco.mj_contactForce(model, data, i, wrench)
      force = sign * (contact.frame.reshape(3, 3).T @ wrench[:3])
      total += force
      load += abs(float(wrench[0]))
      if socket_geom == self._stop:
        stop_force += force
      elif model.geom(socket_geom).name.startswith("ram_socket_spring_"):
        spring_force += force
        spring_load += abs(float(wrench[0]))
      else:
        wall_load += abs(float(wrench[0]))
      penetration = max(penetration, -float(contact.dist))
      count += 1
    resistance = max(0.0, float((frame.T @ total)[2]))
    stop_load = max(0.0, float((frame.T @ stop_force)[2]))
    twist = self.sim.object_twist("ram")
    linear = float(np.linalg.norm(twist[:3]))
    angular = float(np.linalg.norm(twist[3:]))
    seated = bool(
      abs(depth - config.TARGET_INSERTION_DEPTH_M) <= config.SEATED_DEPTH_TOLERANCE_M
      and fits
      and angle <= np.deg2rad(1.0)
      and penetration <= config.MAX_SOCKET_PENETRATION_M
      and (stop_load > config.SEATED_BACKSTOP_MIN_LOAD_N or self._bottom_confirmed)
      and linear < config.SEATED_LINEAR_SPEED_M_S
      and angular < 0.05
    )
    return RamInstallationState(
      float(data.time),
      depth,
      tuple(float(x) for x in bottom[:2]),
      angle,
      fits,
      count,
      load,
      resistance,
      stop_load,
      penetration,
      linear,
      angular,
      seated,
      spring_normal_load_n=spring_load,
      spring_axial_resistance_n=float((frame.T @ spring_force)[2]),
      wall_normal_load_n=wall_load,
      palm_down_cosine=-float(
        data.xmat[model.body("hand_r_base_link").id].reshape(3, 3)[2, 1]
      ),
      bottom_out_duration_s=self._bottom_hold_s,
      bottom_out_confirmed=self._bottom_confirmed,
    )

  def update(self) -> RamInstallationState:
    state = self.measure()
    elapsed = 0.0
    if self._last_time is not None:
      elapsed = state.timestamp - self._last_time
      if elapsed < -1e-12 or elapsed > self.sim.timestep * 1.5:
        self._stable_since = None
        self._bottom_hold_s = 0.0
        self._bottom_confirmed = False
        elapsed = 0.0
    self._last_time = state.timestamp
    near_bottom = (
      abs(state.insertion_depth_m - config.TARGET_INSERTION_DEPTH_M)
      <= config.SEATED_DEPTH_TOLERANCE_M
      and state.aperture_fits
      and state.orientation_error_rad <= np.deg2rad(1.0)
      and state.maximum_socket_penetration_m <= config.MAX_SOCKET_PENETRATION_M
    )
    stationary = (
      state.linear_speed_m_s < config.SEATED_LINEAR_SPEED_M_S
      and state.angular_speed_rad_s < 0.05
    )
    if not near_bottom:
      self._bottom_confirmed = False
      self._bottom_hold_s = 0.0
    if (
      near_bottom
      and stationary
      and state.backstop_load_n >= config.BOTTOM_OUT_MIN_FORCE_N
    ):
      self._bottom_hold_s += max(elapsed, 0.0)
      if self._bottom_hold_s >= config.BOTTOM_OUT_HOLD_S - 1e-10:
        self._bottom_confirmed = True
    elif not self._bottom_confirmed:
      self._bottom_hold_s = 0.0
    state = replace(
      state,
      seated=bool(
        near_bottom
        and stationary
        and (
          state.backstop_load_n > config.SEATED_BACKSTOP_MIN_LOAD_N
          or self._bottom_confirmed
        )
      ),
      bottom_out_confirmed=self._bottom_confirmed,
      bottom_out_duration_s=self._bottom_hold_s,
    )
    if not state.seated:
      self._stable_since = None
    elif self._stable_since is None:
      self._stable_since = state.timestamp
    duration = (
      0.0 if self._stable_since is None else state.timestamp - self._stable_since
    )
    return replace(
      state,
      stable_duration_s=duration,
      success=self._bottom_confirmed and duration >= config.SEATED_DWELL_S - 1e-10,
    )
