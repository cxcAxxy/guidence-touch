"""Contact-gated guide for colliding rounded internal/external helical crests.

The free bulb must enter the socket and load a thread crest before a passive
helical guide is enabled. The guide constrains ideal alignment/pitch; it is
not an electrically or mechanically calibrated E27 thread model.
"""

from collections import deque
from dataclasses import dataclass, replace

import mujoco
import numpy as np

from kaihand_tactile_env.shared.pickup_randomization import PickupPositionRandomization
from kaihand_tactile_env.shared.simulation import ArmHandSimulation

from . import config


@dataclass(frozen=True)
class BulbScrewState:
  timestamp: float
  engaged: bool
  clockwise_turns: float
  insertion_depth_m: float
  lateral_error_m: float
  tilt_rad: float
  thread_error_m: float
  linear_speed_m_s: float
  angular_speed_rad_s: float
  backstop_load_n: float
  seated: bool
  stable_duration_s: float = 0.0
  success: bool = False
  axial_travel_m: float = 0.0
  thread_contact_load_n: float = 0.0
  exposed_thread_m: float = 0.0
  shoulder_gap_m: float = 0.0
  shoulder_contact_load_n: float = 0.0
  cushion_contact_load_n: float = 0.0
  bulb_lit: bool = False
  seated_duration_s: float = 0.0
  tightening_verified: bool = False
  tightening_stall_duration_s: float = 0.0
  tightening_peak_torque_nm: float = 0.0
  released: bool = False
  release_stable_duration_s: float = 0.0
  fingertip_normal_force_n: tuple[float, ...] = ()
  right_hand_contact_load_n: float = 0.0
  right_hand_contact_count: int = 0
  right_hand_nonpad_contact_count: int = 0
  robot_contact_load_n: float = 0.0
  robot_contact_count: int = 0
  five_finger_contact_duration_s: float = 0.0
  maximum_five_finger_contact_duration_s: float = 0.0
  five_finger_contact_fraction: float = 0.0
  hand_clockwise_torque_nm: float = 0.0
  bulb_pose_wxyz: tuple[float, ...] = ()
  capture_armed: bool = False
  capture_ready: bool = False
  capture_depth_margin_m: float = 0.0
  capture_lateral_margin_m: float = 0.0
  capture_tilt_margin_rad: float = 0.0
  capture_speed_margin_m_s: float = 0.0
  capture_angular_speed_margin_rad_s: float = 0.0
  capture_contact_margin_n: float = 0.0


class BulbScrewSimulation(ArmHandSimulation):
  """Free tabletop bulb with automatically gated, reversible thread capture."""

  def __init__(self, *, position_seed=None, **kwargs):
    self.position_seed = position_seed
    self._pickup_randomization = None
    super().__init__(scene=config.SCENE_NAME, **kwargs)

  def reset(self, **kwargs) -> None:
    super().reset(**kwargs)
    if self._pickup_randomization is None:
      self._pickup_randomization = PickupPositionRandomization(
        self, "bulb", support_body="bulb_pickup_cradle"
      )
    self.initial_position_randomization = self._pickup_randomization.apply(
      self, self.position_seed
    )
    self._weld = self.model.equality("bulb_thread_capture").id
    self._slide = int(self.model.joint("bulb_thread_depth").qposadr[0])
    self._hinge = int(self.model.joint("bulb_thread_angle").qposadr[0])
    self._slide_dof = int(self.model.joint("bulb_thread_depth").dofadr[0])
    self._hinge_dof = int(self.model.joint("bulb_thread_angle").dofadr[0])
    self._bulb_id = self.model.body("bulb").id
    self._bulb_geoms = {
      i for i in range(self.model.ngeom) if self.model.geom_bodyid[i] == self._bulb_id
    }
    self._right_fingertip_pads = {
      self.model.geom(f"hand_r_{finger}_link{link}_tactile_pad_col").id: index
      for index, (finger, link) in enumerate(
        (("thumb", 6), ("index", 4), ("middle", 4), ("ring", 4), ("pinky", 4))
      )
    }

    def descendants(root_name: str) -> set[int]:
      root = self.model.body(root_name).id
      bodies = set()
      for body_id in range(self.model.nbody):
        ancestor = body_id
        while ancestor > 0 and ancestor != root:
          ancestor = int(self.model.body_parentid[ancestor])
        if ancestor == root:
          bodies.add(body_id)
      return bodies

    right_hand_bodies = descendants("right_hand_mount")
    robot_bodies = descendants("left_arm_base") | descendants("right_arm_base")
    self._right_hand_geoms = {
      geom_id
      for geom_id in range(self.model.ngeom)
      if int(self.model.geom_bodyid[geom_id]) in right_hand_bodies
    }
    self._robot_geoms = {
      geom_id
      for geom_id in range(self.model.ngeom)
      if int(self.model.geom_bodyid[geom_id]) in robot_bodies
    }
    self._mouth_id = self.model.site("bulb_socket_mouth").id
    self._external_thread_geoms = {
      self.model.geom(f"bulb_thread_crest_{i:03d}").id
      for i in range(config.EXTERNAL_THREAD_SEGMENTS)
    }
    self._internal_thread_geoms = {
      self.model.geom(f"bulb_socket_thread_{i:03d}").id
      for i in range(config.INTERNAL_THREAD_SEGMENTS)
    }
    self._socket_walls = {
      self.model.geom(f"bulb_socket_wall_{i:02d}").id for i in range(16)
    }
    self._socket_cushions = {
      self.model.geom(f"bulb_socket_cushion_{i:02d}").id for i in range(16)
    }
    self._capture_armed = True
    self._capture_alignment_started_s = None
    self._capture_alignment_initial_relpose = None
    self._capture_alignment_target_relpose = None
    self._set_bulb_lit(False)
    self.model.eq_data[self._weld, 3:10] = (0, 0, 0, 1, 0, 0, 0)
    self.data.eq_active[self._weld] = False
    mujoco.mj_forward(self.model, self.data)

  @property
  def thread_engaged(self) -> bool:
    return bool(self.data.eq_active[self._weld])

  @property
  def bulb_lit(self) -> bool:
    return bool(self.model.light_active[self.model.light("bulb_glow").id])

  def _set_bulb_lit(self, lit: bool) -> None:
    """Switch visual material and local illumination, without changing physics."""
    material = "bulb_glowing" if lit else "bulb_frosted"
    self.model.geom_matid[self.model.geom("bulb_globe").id] = self.model.material(
      material
    ).id
    self.model.light_active[self.model.light("bulb_glow").id] = lit

  def confirm_tightening(self) -> None:
    """Latch the completion light after the controller verifies loaded stall."""
    if not BulbScrewMonitor(self).measure().seated:
      raise ValueError("cannot light a bulb that is not seated")
    self._set_bulb_lit(True)

  def initialize_threaded(self) -> None:
    """Explicit reset preset: insert the lead-in and touch the first crest."""
    self.reset()
    self.set_object_pose("bulb", config.THREAD_ENTRY_POSITION_M)
    if not self.try_engage_thread():
      raise RuntimeError("thread preset failed to align with the socket")

  def _alignment(self):
    position = self.data.xpos[self._bulb_id] - self.data.site_xpos[self._mouth_id]
    rotation = self.data.xmat[self._bulb_id].reshape(3, 3)
    tilt = float(np.arccos(np.clip(rotation[2, 2], -1.0, 1.0)))
    return -float(position[2]), float(np.linalg.norm(position[:2])), tilt

  def thread_contact_load(self) -> float:
    """Sum actual internal/external crest contact loads, excluding the guide."""
    load = 0.0
    wrench = np.zeros(6)
    for i, contact in enumerate(self.data.contact):
      a, b = int(contact.geom1), int(contact.geom2)
      if (a in self._external_thread_geoms and b in self._internal_thread_geoms) or (
        b in self._external_thread_geoms and a in self._internal_thread_geoms
      ):
        mujoco.mj_contactForce(self.model, self.data, i, wrench)
        load += max(0.0, float(wrench[0]))
    return load

  def hand_object_contact_metrics(
    self,
  ) -> tuple[np.ndarray, float, float, int, int, float, int]:
    """Return pad torque plus right-hand and whole-robot bulb contact.

    Tightening is defined by the five tactile pads and rejects hidden support
    from any non-pad right-hand geom. Release is stricter: any arm or hand geom
    on either side still touching the bulb prevents completion.
    """
    normal = np.zeros(len(self._right_fingertip_pads), dtype=float)
    clockwise_torque = 0.0
    right_hand_load = 0.0
    right_hand_contact_count = 0
    right_hand_nonpad_contact_count = 0
    robot_load = 0.0
    robot_contact_count = 0
    wrench = np.empty(6)
    for i in range(self.data.ncon):
      contact = self.data.contact[i]
      geom1, geom2 = int(contact.geom1), int(contact.geom2)
      if geom1 in self._bulb_geoms and geom2 in self._robot_geoms:
        robot_geom = geom2
      elif geom2 in self._bulb_geoms and geom1 in self._robot_geoms:
        robot_geom = geom1
      else:
        continue
      mujoco.mj_contactForce(self.model, self.data, i, wrench)
      load = abs(float(wrench[0]))
      robot_load += load
      robot_contact_count += 1
      if robot_geom not in self._right_hand_geoms:
        continue
      hand_geom = robot_geom
      right_hand_load += load
      right_hand_contact_count += 1
      pad = hand_geom
      if pad not in self._right_fingertip_pads:
        right_hand_nonpad_contact_count += 1
        continue
      normal[self._right_fingertip_pads[pad]] += load
      frame = np.asarray(contact.frame).reshape(3, 3)
      sign = 1 if geom2 in self._bulb_geoms else -1
      world_force = sign * (frame.T @ wrench[:3])
      world_moment = sign * (frame.T @ wrench[3:])
      clockwise_torque -= float(
        (
          np.cross(contact.pos - self.data.xpos[self._bulb_id], world_force)
          + world_moment
        )[2]
      )
    return (
      normal,
      clockwise_torque,
      right_hand_load,
      right_hand_contact_count,
      right_hand_nonpad_contact_count,
      robot_load,
      robot_contact_count,
    )

  def hand_contact_metrics(self) -> tuple[np.ndarray, float]:
    """Backward-compatible fingertip loads and clockwise hand torque."""
    normal, torque, *_ = self.hand_object_contact_metrics()
    return normal, torque

  def try_engage_thread(self) -> bool:
    """Capture only after insertion and physical thread contact.

    Never overwrite the free bulb's pose/velocity.

    The passive carriage is initialized at the measured axial/thread phase so
    enabling the weld cannot snap depth or yaw to an unrelated zero.  The
    bounded lateral/tilt residual is initially retained in the weld and then
    removed over ``CAPTURE_ALIGNMENT_S`` instead of as a one-step impulse.
    """
    if self.thread_engaged:
      return True
    mujoco.mj_forward(self.model, self.data)
    depth, lateral, tilt = self._alignment()
    twist = self.object_twist("bulb")
    if not (
      self._capture_armed
      and abs(depth - config.THREAD_ENTRY_DEPTH_M) <= config.CAPTURE_DEPTH_M
      and self.thread_contact_load() > config.THREAD_CAPTURE_LOAD_N
      and lateral <= config.CAPTURE_LATERAL_M
      and tilt <= config.CAPTURE_TILT_RAD
      and np.linalg.norm(twist[:3]) <= config.CAPTURE_SPEED_M_S
      and np.linalg.norm(twist[3:]) <= config.CAPTURE_ANGULAR_SPEED_RAD_S
    ):
      return False
    rotation = self.data.xmat[self._bulb_id].reshape(3, 3)
    yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
    depth_offset = depth - config.THREAD_ENTRY_DEPTH_M
    thread_angle = 2 * np.pi * depth_offset / config.THREAD_PITCH_M
    relative_yaw = float(
      np.arctan2(np.sin(yaw + thread_angle), np.cos(yaw + thread_angle))
    )
    self.data.qpos[[self._slide, self._hinge]] = (depth_offset, thread_angle)
    self.data.qvel[[self._slide_dof, self._hinge_dof]] = 0
    # Refresh only the passive carriage pose.  The free bulb pose/twist remain
    # untouched until the now-satisfied weld is activated.
    mujoco.mj_forward(self.model, self.data)
    carriage = self.model.body("bulb_thread_carriage").id
    carriage_rotation = self.data.xmat[carriage].reshape(3, 3)
    bulb_rotation = self.data.xmat[self._bulb_id].reshape(3, 3)
    relative_position = carriage_rotation.T @ (
      self.data.xpos[self._bulb_id] - self.data.xpos[carriage]
    )
    relative_quaternion = np.empty(4)
    mujoco.mju_mat2Quat(
      relative_quaternion,
      (carriage_rotation.T @ bulb_rotation).reshape(9),
    )
    target_quaternion = np.array(
      [np.cos(relative_yaw / 2), 0.0, 0.0, np.sin(relative_yaw / 2)]
    )
    if float(relative_quaternion @ target_quaternion) < 0:
      relative_quaternion *= -1
    initial_relpose = np.concatenate((relative_position, relative_quaternion))
    target_relpose = np.concatenate((np.zeros(3), target_quaternion))
    self.model.eq_data[self._weld, 3:10] = initial_relpose
    self._capture_alignment_started_s = float(self.data.time)
    self._capture_alignment_initial_relpose = initial_relpose
    self._capture_alignment_target_relpose = target_relpose
    self.data.eq_active[self._weld] = True
    mujoco.mj_forward(self.model, self.data)
    return True

  def _advance_capture_alignment(self) -> None:
    initial = self._capture_alignment_initial_relpose
    target = self._capture_alignment_target_relpose
    if initial is None or target is None:
      return
    elapsed = (
      float(self.data.time) + self.timestep - float(self._capture_alignment_started_s)
    )
    fraction = float(np.clip(elapsed / config.CAPTURE_ALIGNMENT_S, 0.0, 1.0))
    relpose = (1 - fraction) * initial + fraction * target
    # The capture gate limits the orientation difference to two degrees, so a
    # normalized shortest-path lerp is numerically indistinguishable from
    # slerp here and keeps the equality quaternion unit length.
    relpose[3:] /= np.linalg.norm(relpose[3:])
    self.model.eq_data[self._weld, 3:10] = relpose
    if fraction >= 1.0:
      self._capture_alignment_initial_relpose = None
      self._capture_alignment_target_relpose = None

  def _before_physics_step(self) -> None:
    # Refresh geometry because mj_step's caches precede its final integration.
    mujoco.mj_forward(self.model, self.data)
    if self.thread_engaged:
      self._advance_capture_alignment()
      if self.data.qpos[self._slide] < config.THREAD_RELEASE_DEPTH_OFFSET_M:
        self.data.eq_active[self._weld] = False
        self._capture_armed = False
        self._capture_alignment_started_s = None
        self._capture_alignment_initial_relpose = None
        self._capture_alignment_target_relpose = None
    else:
      depth, _, _ = self._alignment()
      if depth < -0.002:
        self._capture_armed = True
      self.try_engage_thread()
    if self.bulb_lit:
      depth, _, _ = self._alignment()
      if not self.thread_engaged or depth < (
        config.SEATED_DEPTH_M - config.SEATED_DEPTH_TOLERANCE_M
      ):
        self._set_bulb_lit(False)


class BulbScrewMonitor:
  """Track seated dwell, plus loaded tightening and release in policy mode."""

  def __init__(
    self,
    simulation: BulbScrewSimulation,
    *,
    auto_confirm_tightening: bool = True,
  ):
    self.simulation = simulation
    self.auto_confirm_tightening = bool(auto_confirm_tightening)
    self.reset()

  def reset(self) -> None:
    self._last_time = None
    self._seated_since = None
    self._completion_since = None
    self._five_finger_contact_since = None
    self._maximum_five_finger_contact_duration_s = 0.0
    self._contact_sample_count = 0
    self._five_finger_contact_sample_count = 0
    self._tightening_history = deque()
    self._tightening_peak_torque_nm = 0.0
    self._tightening_stall_duration_s = 0.0

  def measure(self) -> BulbScrewState:
    sim = self.simulation
    model, data = sim.model, sim.data
    mujoco.mj_forward(model, data)
    depth, lateral, tilt = sim._alignment()
    turns = float(data.qpos[sim._hinge] / (2 * np.pi)) if sim.thread_engaged else 0.0
    thread_error = depth - config.THREAD_ENTRY_DEPTH_M - config.THREAD_PITCH_M * turns
    twist = sim.object_twist("bulb")
    linear = float(np.linalg.norm(twist[:3]))
    angular = float(np.linalg.norm(twist[3:]))
    thread_contact = sim.thread_contact_load()
    (
      fingertip_loads,
      hand_torque,
      right_hand_contact_load,
      right_hand_contact_count,
      right_hand_nonpad_contact_count,
      robot_contact_load,
      robot_contact_count,
    ) = sim.hand_object_contact_metrics()
    pose = tuple(float(value) for value in sim.object_pose("bulb"))
    capture_depth_margin = config.CAPTURE_DEPTH_M - abs(
      depth - config.THREAD_ENTRY_DEPTH_M
    )
    capture_lateral_margin = config.CAPTURE_LATERAL_M - lateral
    capture_tilt_margin = config.CAPTURE_TILT_RAD - tilt
    capture_speed_margin = config.CAPTURE_SPEED_M_S - linear
    capture_angular_speed_margin = config.CAPTURE_ANGULAR_SPEED_RAD_S - angular
    capture_contact_margin = thread_contact - config.THREAD_CAPTURE_LOAD_N
    capture_ready = bool(
      sim._capture_armed
      and capture_depth_margin >= 0
      and capture_lateral_margin >= 0
      and capture_tilt_margin >= 0
      and capture_speed_margin >= 0
      and capture_angular_speed_margin >= 0
      and capture_contact_margin > 0
    )
    released = bool(
      fingertip_loads.max(initial=0.0) <= config.RELEASE_FINGERTIP_LOAD_N
      and right_hand_contact_load <= config.RELEASE_HAND_LOAD_N
      and robot_contact_load <= config.RELEASE_HAND_LOAD_N
      and robot_contact_count == 0
    )
    stop = model.geom("bulb_socket_backstop").id
    shoulder = model.geom("bulb_neck").id
    mouth_z = data.site_xpos[sim._mouth_id, 2]
    # A capsule's endpoint extent along world Z plus its radius. The highest
    # external crest must be below the physical rim, not just near turn one.
    crests = np.array(sorted(sim._external_thread_geoms))
    crest_z = (
      data.geom_xpos[crests, 2]
      + np.abs(data.geom_xmat[crests, 8]) * model.geom_size[crests, 1]
      + model.geom_size[crests, 0]
    )
    exposed_thread = max(0.0, float(np.max(crest_z) - mouth_z))
    # Highest edge of the shoulder's bottom face also detects tilted seating.
    axis_z = data.geom_xmat[shoulder, 8]
    shoulder_gap = float(
      data.geom_xpos[shoulder, 2]
      - model.geom_size[shoulder, 1] * axis_z
      + model.geom_size[shoulder, 0] * np.sqrt(max(0.0, 1 - axis_z**2))
      - mouth_z
    )
    load = 0.0
    shoulder_load = 0.0
    cushion_load = 0.0
    wrench = np.zeros(6)
    for i in range(data.ncon):
      contact = data.contact[i]
      ids = (int(contact.geom1), int(contact.geom2))
      if stop in ids and any(model.geom_bodyid[g] == sim._bulb_id for g in ids):
        mujoco.mj_contactForce(model, data, i, wrench)
        load += max(0.0, float(wrench[0]))
      if shoulder in ids and any(g in sim._socket_walls for g in ids):
        mujoco.mj_contactForce(model, data, i, wrench)
        shoulder_load += max(0.0, float(wrench[0]))
      if shoulder in ids and any(g in sim._socket_cushions for g in ids):
        mujoco.mj_contactForce(model, data, i, wrench)
        cushion_load += max(0.0, float(wrench[0]))
    seated = bool(
      sim.thread_engaged
      and abs(depth - config.SEATED_DEPTH_M) <= config.SEATED_DEPTH_TOLERANCE_M
      and abs(turns - config.TARGET_TURNS) <= 0.08
      and abs(thread_error) <= config.SEATED_DEPTH_TOLERANCE_M
      and lateral <= config.CAPTURE_LATERAL_M
      and tilt <= config.CAPTURE_TILT_RAD
      and linear <= 0.001
      and angular <= 0.05
      and load >= 0.01
      and exposed_thread <= config.SEATED_SHOULDER_GAP_M
      and abs(shoulder_gap) <= config.SEATED_SHOULDER_GAP_M
      and shoulder_load >= 0.01
    )
    return BulbScrewState(
      float(data.time),
      sim.thread_engaged,
      turns,
      depth,
      lateral,
      tilt,
      thread_error,
      linear,
      angular,
      load,
      seated,
      axial_travel_m=depth - config.THREAD_ENTRY_DEPTH_M,
      thread_contact_load_n=thread_contact,
      exposed_thread_m=exposed_thread,
      shoulder_gap_m=shoulder_gap,
      shoulder_contact_load_n=shoulder_load,
      cushion_contact_load_n=cushion_load,
      bulb_lit=sim.bulb_lit,
      tightening_verified=sim.bulb_lit,
      tightening_stall_duration_s=self._tightening_stall_duration_s,
      tightening_peak_torque_nm=self._tightening_peak_torque_nm,
      released=released,
      fingertip_normal_force_n=tuple(float(value) for value in fingertip_loads),
      right_hand_contact_load_n=float(right_hand_contact_load),
      right_hand_contact_count=right_hand_contact_count,
      right_hand_nonpad_contact_count=right_hand_nonpad_contact_count,
      robot_contact_load_n=float(robot_contact_load),
      robot_contact_count=robot_contact_count,
      hand_clockwise_torque_nm=float(hand_torque),
      bulb_pose_wxyz=pose,
      capture_armed=bool(sim._capture_armed),
      capture_ready=capture_ready,
      capture_depth_margin_m=float(capture_depth_margin),
      capture_lateral_margin_m=float(capture_lateral_margin),
      capture_tilt_margin_rad=float(capture_tilt_margin),
      capture_speed_margin_m_s=float(capture_speed_margin),
      capture_angular_speed_margin_rad_s=float(capture_angular_speed_margin),
      capture_contact_margin_n=float(capture_contact_margin),
    )

  def _update_tightening(self, state: BulbScrewState) -> BulbScrewState:
    if state.bulb_lit:
      return replace(
        state,
        tightening_verified=True,
        tightening_stall_duration_s=self._tightening_stall_duration_s,
        tightening_peak_torque_nm=self._tightening_peak_torque_nm,
      )
    if not self.auto_confirm_tightening or not state.seated:
      self._tightening_history.clear()
      self._tightening_stall_duration_s = 0.0
      return state

    rotation = self.simulation.data.xmat[self.simulation._bulb_id].reshape(3, 3)
    yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
    minimum_load = min(state.fingertip_normal_force_n, default=0.0)
    history = self._tightening_history
    history.append(
      (
        state.timestamp,
        yaw,
        state.hand_clockwise_torque_nm,
        minimum_load,
        state.seated,
        state.right_hand_nonpad_contact_count == 0
        and state.robot_contact_count == state.right_hand_contact_count,
      )
    )
    while (
      len(history) > 1 and state.timestamp - history[1][0] >= config.TIGHTENING_HOLD_S
    ):
      history.popleft()
    self._tightening_peak_torque_nm = max(
      self._tightening_peak_torque_nm,
      state.hand_clockwise_torque_nm,
    )
    duration = state.timestamp - history[0][0]
    angles = np.unwrap(np.asarray([sample[1] for sample in history]))
    displacement = float(np.ptp(angles))
    verified = bool(
      duration >= config.TIGHTENING_HOLD_S - 1e-9
      and displacement <= config.TIGHTENING_MAX_ROTATION_RAD
      and min(sample[2] for sample in history) >= config.TIGHTENING_MIN_TORQUE_NM
      and min(sample[3] for sample in history) >= 4.0
      and all(sample[4] for sample in history)
      and all(sample[5] for sample in history)
    )
    self._tightening_stall_duration_s = duration if verified else 0.0
    if verified:
      self.simulation.confirm_tightening()
    return replace(
      state,
      bulb_lit=verified,
      tightening_verified=verified,
      tightening_stall_duration_s=self._tightening_stall_duration_s,
      tightening_peak_torque_nm=self._tightening_peak_torque_nm,
    )

  def update(self) -> BulbScrewState:
    state = self.measure()
    if self._last_time is not None:
      elapsed = state.timestamp - self._last_time
      if elapsed < 0 or elapsed > 1.5 * self.simulation.timestep:
        self._seated_since = None
        self._completion_since = None
        self._five_finger_contact_since = None
        self._tightening_history.clear()
        self._tightening_stall_duration_s = 0.0
    self._last_time = state.timestamp
    state = self._update_tightening(state)

    five_finger_contact = bool(
      len(state.fingertip_normal_force_n) == 5
      and min(state.fingertip_normal_force_n) >= 0.025
    )
    self._contact_sample_count += 1
    self._five_finger_contact_sample_count += int(five_finger_contact)
    if not five_finger_contact:
      self._five_finger_contact_since = None
    elif self._five_finger_contact_since is None:
      self._five_finger_contact_since = state.timestamp
    five_finger_duration = (
      0.0
      if self._five_finger_contact_since is None
      else state.timestamp - self._five_finger_contact_since
    )
    self._maximum_five_finger_contact_duration_s = max(
      self._maximum_five_finger_contact_duration_s,
      five_finger_duration,
    )
    five_finger_fraction = (
      self._five_finger_contact_sample_count / self._contact_sample_count
    )

    if not state.seated:
      self._seated_since = None
    elif self._seated_since is None:
      self._seated_since = state.timestamp
    seated_duration = (
      0.0 if self._seated_since is None else state.timestamp - self._seated_since
    )

    completion_candidate = bool(
      state.seated and state.tightening_verified and state.released
    )
    if not completion_candidate:
      self._completion_since = None
    elif self._completion_since is None:
      self._completion_since = state.timestamp
    completion_duration = (
      0.0
      if self._completion_since is None
      else state.timestamp - self._completion_since
    )
    success_duration = (
      completion_duration if self.auto_confirm_tightening else seated_duration
    )
    return replace(
      state,
      seated_duration_s=seated_duration,
      five_finger_contact_duration_s=five_finger_duration,
      maximum_five_finger_contact_duration_s=self._maximum_five_finger_contact_duration_s,
      five_finger_contact_fraction=five_finger_fraction,
      stable_duration_s=success_duration,
      release_stable_duration_s=completion_duration,
      success=success_duration >= config.SEATED_DWELL_S - 1e-12,
    )
