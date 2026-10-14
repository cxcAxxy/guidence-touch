"""Task adapters for model-only closed-loop evaluation of the newer tasks."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .evaluation_guards import (
  TaskGuardViolation,
  configure_evaluation_guards,
  ensure_evaluation_physics,
  evaluation_guard_report,
  record_guard_event,
  task_guard,
)

TASK_INSTRUCTIONS = {
  "usb-unplug": (
    "Grasp the already inserted USB plug with the right hand, pull it straight "
    "out of the socket, return it to the table, release it, and withdraw the hand."
  ),
  "bulb-unscrew": (
    "Grasp the already seated light bulb with the right hand, unscrew it "
    "counterclockwise, lift it clear of the socket, place it stably on the "
    "ordinary tabletop, release it, and return the hand and arm to the initial posture."
  ),
  "whiteboard-wipe-horizontal": (
    "Pick up the eraser with the right hand, wipe all ink from the horizontal "
    "face-up whiteboard with loaded sliding contact, then return and release the eraser "
    "and return the open hand and arm to the initial posture."
  ),
  "whiteboard-wipe-horizontal-line": (
    "Pick up the eraser with the right hand, wipe the single top horizontal "
    "ink line from the horizontal face-up whiteboard with loaded sliding "
    "contact, then return and release the eraser and return the open hand "
    "and arm to the initial posture."
  ),
  "bulb-screw": (
    "Pick up the light bulb with the right hand, align it with the socket, "
    "screw it clockwise until mechanically seated, then release it."
  ),
  "install-ram": (
    "Pick up the RAM module with the right hand, align its keyed edge with "
    "the socket, press it straight down until seated, then release it."
  ),
  "vase-wipe": (
    "Pick up the sponge with the right hand and wipe the marked dirt from "
    "the inside wall of the vase until it is clean."
  ),
  "whiteboard-wipe": (
    "Pick up the eraser with the right hand, wipe all ink from the tilted "
    "whiteboard with loaded sliding contact, then return and release the eraser."
  ),
  "sponge-grasp": (
    "Pick up the upright sponge with the right hand, carry it to the plate on "
    "the robot's right, place it inside the plate, and release it."
  ),
}

HORIZONTAL_WHITEBOARD_TASKS = (
  "whiteboard-wipe-horizontal", "whiteboard-wipe-horizontal-line",
)
WHITEBOARD_TASKS = ("whiteboard-wipe", *HORIZONTAL_WHITEBOARD_TASKS)


@dataclass(frozen=True)
class WhiteboardPolicyState:
  cleaned_fraction: float
  maximum_ink_remaining: float
  board_normal_force_n: float
  board_tangent_force_n: float
  table_support_force_n: float
  fingertip_normal_force_n: tuple[float, ...]
  eraser_linear_speed_m_s: float
  maximum_lift_m: float
  released_on_table: bool
  success: bool
  home_return_verified: bool = False
  pickup_verified: bool = False


@dataclass(frozen=True)
class SpongePolicyState:
  timestamp: float
  sponge_bottom_z_m: float
  sponge_centroid_xyz_m: tuple[float, float, float]
  sponge_plate_center_offset_m: float
  sponge_maximum_plate_radius_m: float
  table_support_force_n: float
  plate_base_support_force_n: float
  plate_base_mean_support_force_n: float
  fingertip_normal_force_n: tuple[float, ...]
  fingertip_tangent_force_n: tuple[float, ...]
  maximum_lift_clearance_m: float
  five_finger_grasp_stable_s: float
  grasp_established: bool
  lift_established: bool
  transport_contact_valid: bool
  minimum_carry_clearance_m: float
  maximum_carry_table_support_force_n: float
  recent_plate_base_mean_support_force_n: float
  release_allowed: bool
  released: bool
  settle_stable_s: float
  settle_pin_height_range_m: float
  settle_centroid_height_range_m: float
  settle_bottom_height_range_m: float
  grasp_patch_enabled: bool
  success: bool


class TaskPolicyAdapter:
  """Own task reset, per-physics-step metrics, guards, and success semantics."""

  def __init__(
    self,
    task: str,
    seed: int,
    *,
    guard_mode: str = "strict",
    diagnostic_relax_whiteboard_contact_guards: bool = False,
  ):
    if task not in TASK_INSTRUCTIONS:
      raise ValueError(f"unsupported shared policy task: {task!r}")
    if seed < 0:
      raise ValueError("seed must be nonnegative")
    if diagnostic_relax_whiteboard_contact_guards and task != "whiteboard-wipe":
      raise ValueError(
        "diagnostic whiteboard contact-guard relaxation is only valid for "
        "whiteboard-wipe"
      )
    self.task = task
    self.seed = seed
    self.instruction = TASK_INSTRUCTIONS[task]
    self.diagnostic_relax_whiteboard_contact_guards = bool(
      diagnostic_relax_whiteboard_contact_guards
    )
    self.whiteboard_force_guard_observed = False
    self.whiteboard_collision_guard_observed = False
    self._success = False
    if task == "usb-unplug":
      from kaihand_tactile_env.tasks.usb_unplug.setup import UsbUnplugSimulation
      from kaihand_tactile_env.tasks.usb_unplug.task import UsbUnplugMonitor

      self.simulation = UsbUnplugSimulation(seed=seed)
      self.monitor = UsbUnplugMonitor(self.simulation)
      self.state = self.monitor.measure()
      self.object_name = "usb_plug"
    elif task == "bulb-unscrew":
      from kaihand_tactile_env.tasks.bulb_unscrew.task import (
        BulbUnscrewMonitor,
        BulbUnscrewSimulation,
      )

      self.simulation = BulbUnscrewSimulation(position_seed=seed)
      self.monitor = BulbUnscrewMonitor(self.simulation)
      self.state = self.monitor.measure()
      self.object_name = "bulb"
    elif task == "bulb-screw":
      from kaihand_tactile_env.tasks.bulb_screw.task import (
        BulbScrewMonitor,
        BulbScrewSimulation,
      )

      self.simulation = BulbScrewSimulation(position_seed=seed)
      self.monitor = BulbScrewMonitor(self.simulation, auto_confirm_tightening=True)
      self.state = self.monitor.measure()
      self.object_name = "bulb"
    elif task == "install-ram":
      from kaihand_tactile_env.tasks.install_ram.task import (
        RamInstallationMonitor,
        RamInstallSimulation,
      )

      self.simulation = RamInstallSimulation(position_seed=seed)
      self.monitor = RamInstallationMonitor(self.simulation)
      self.state = self.monitor.measure()
      self.object_name = "ram"
    elif task == "vase-wipe":
      from kaihand_tactile_env.tasks.vase_wipe.task import VaseWipeSimulation

      self.simulation = VaseWipeSimulation(stain_seed=seed)
      self.monitor = None
      self.state = self.simulation.measure()
      self.object_name = "sponge"
    elif task in WHITEBOARD_TASKS:
      from kaihand_tactile_env.tasks.whiteboard_wipe.task import (
        WhiteboardWipeSimulation,
      )

      if task == "whiteboard-wipe-horizontal-line":
        from kaihand_tactile_env.tasks.whiteboard_wipe_horizontal_line.task import (
          HorizontalLineWhiteboardWipeSimulation,
        )

        self.simulation = HorizontalLineWhiteboardWipeSimulation(ink_seed=seed)
      elif task == "whiteboard-wipe-horizontal":
        from kaihand_tactile_env.tasks.whiteboard_wipe_horizontal.task import (
          HorizontalWhiteboardWipeSimulation,
        )

        self.simulation = HorizontalWhiteboardWipeSimulation(ink_seed=seed)
      else:
        self.simulation = WhiteboardWipeSimulation(ink_seed=seed)
      self.monitor = None
      self.object_name = "eraser"
      self.maximum_lift_m = 0.0
      self.release_stable_s = 0.0
      self.horizontal_opposed_grasp_verified = False
      self.horizontal_pickup_verified = False
    else:
      from kaihand_tactile_env.tasks.sponge_grasp.task import SpongeGraspSimulation

      self.simulation = SpongeGraspSimulation(
        position_seed=seed,
        randomize_xy=True,
      )
      self.monitor = None
      self.object_name = "sponge"
      self.sponge_grasp_stable_s = 0.0
      self.sponge_grasp_established = False
      self.sponge_lift_established = False
      self.sponge_transport_contact_valid = True
      self.sponge_minimum_carry_clearance_m = np.inf
      self.sponge_maximum_carry_table_support_n = 0.0
      self.sponge_placement_started = False
      self.sponge_release_allowed = False
      self.sponge_release_stable_s = 0.0
      self.sponge_released = False
      self.sponge_maximum_clearance_m = 0.0
      self.sponge_next_audit_sample_s = 0.0
      self.sponge_plate_impulse_history = deque()
      self.sponge_recent_plate_mean_support_n = 0.0
      self.sponge_settle_start_s = None
      self.sponge_settle_start_impulse_ns = 0.0
      self.sponge_settle_min_heights = None
      self.sponge_settle_max_heights = None
    configure_evaluation_guards(self.simulation, mode=guard_mode)
    self.guard_mode = guard_mode
    self.initial_object_pose = self.simulation.object_pose(self.object_name).copy()
    if task in WHITEBOARD_TASKS:
      self.state = self._whiteboard_state()
    elif task == "sponge-grasp":
      self.state = self._sponge_state(self._sponge_sample())

  @property
  def success(self) -> bool:
    return self._success

  def enable_staged_evaluation(self, version: str = "staged_v1") -> None:
    """Attach read-only observers after reset; original mechanics stay intact."""
    from .staged_evaluation import StagedEvaluationMonitor

    self.staged_evaluation = StagedEvaluationMonitor(self.task, self.simulation, version=version)
    self.whiteboard_geometry_metrics = None
    if self.task in WHITEBOARD_TASKS:
      from .whiteboard_geometry_metrics import WhiteboardGeometryMetrics

      self.whiteboard_geometry_metrics = WhiteboardGeometryMetrics(self.simulation)
      self._staged_prior_contact_observer = self.simulation.evaluation_contact_observer

      def capture_integrated_contacts(simulation, before_time):
        if self._staged_prior_contact_observer is not None:
          self._staged_prior_contact_observer(simulation, before_time)
        self.staged_evaluation.capture_integrated_contacts(simulation, before_time)

      self._staged_contact_callback = capture_integrated_contacts
      self.simulation.evaluation_contact_observer = capture_integrated_contacts
    self._update_staged_evaluation()

  def detach_staged_contact_observer(self) -> None:
    callback = getattr(self, "_staged_contact_callback", None)
    if callback is not None and self.simulation.evaluation_contact_observer is callback:
      self.simulation.evaluation_contact_observer = self._staged_prior_contact_observer

  def _update_staged_evaluation(self) -> None:
    observer = getattr(self, "staged_evaluation", None)
    if observer is None:
      return
    observer.update(self.simulation, task_state=self.state, return_snapshot=False)
    geometry = self.whiteboard_geometry_metrics
    if geometry is not None:
      observer.attach_continuous_metrics(geometry.update(self.simulation))

  def stage(self) -> str:
    if self._success:
      return "success"
    if self.task in {"usb-unplug", "bulb-unscrew"}:
      state = self.state
      if state.released:
        return "retreat_and_home"
      if state.placed_on_table:
        return "release"
      if state.fully_extracted:
        return "return_to_table"
      if self.task == "bulb-unscrew" and state.thread_disengaged:
        return "extract_clear"
      if state.grasp_verified:
        return "withdraw_plug" if self.task == "usb-unplug" else "unscrew_counterclockwise"
      verified = (
        state.initially_seated_verified if self.task == "usb-unplug"
        else state.initially_installed_verified
      )
      return "grasp_installed_object" if verified else "verify_installed_initial_state"
    if self.task == "bulb-screw":
      from kaihand_tactile_env.tasks.bulb_screw import config

      state = self.state
      if state.tightening_verified:
        return "verify_released" if state.released else "release_bulb"
      if state.seated:
        return "tighten"
      if state.engaged:
        return "screw_thread" if state.clockwise_turns > 0.01 else "thread_engaged"
      if state.capture_ready:
        return "capture_ready"
      if state.thread_contact_load_n > 0:
        return "thread_contact"
      near_axis = bool(
        state.lateral_error_m <= 0.005 and state.tilt_rad <= np.deg2rad(10)
      )
      if (
        near_axis
        and state.insertion_depth_m
        < config.THREAD_ENTRY_DEPTH_M - config.CAPTURE_DEPTH_M
      ):
        return "aligned_above_socket"
      position = np.asarray(state.bulb_pose_wxyz[:3], dtype=float)
      if (
        state.lateral_error_m <= 0.030
        and position[2] >= config.THREAD_ENTRY_POSITION_M[2] - 0.020
      ):
        return "insert_to_thread"
      if position[2] - self.initial_object_pose[2] > 0.030:
        return "transport_to_socket"
      if max(state.fingertip_normal_force_n, default=0.0) > 0.05:
        return "grasp_and_lift"
      return "pick_bulb"
    if self.task == "install-ram":
      if self.state.seated:
        return "seated"
      if self.state.insertion_depth_m > 0:
        return "inserting"
      if self.state.aperture_fits:
        return "aligned"
      return "pick_and_align"
    if self.task in WHITEBOARD_TASKS:
      if self.task in HORIZONTAL_WHITEBOARD_TASKS and self.state.released_on_table:
        return "retreat_and_home"
      if self.state.maximum_ink_remaining <= 1e-9:
        return "return_and_release"
      if self.state.board_normal_force_n > 0.05:
        return "wiping"
      if self.state.maximum_lift_m > 0.05:
        return "carry_eraser"
      if max(self.state.fingertip_normal_force_n, default=0.0) > 0.05:
        return "eraser_grasped"
      return "pick_eraser"
    if self.task == "sponge-grasp":
      if self.state.settle_stable_s > 0:
        return "settle_in_plate"
      if self.state.released:
        return "released"
      if self.state.plate_base_support_force_n > 0.08:
        return "lower_into_plate"
      if self.state.lift_established:
        return "carry_to_plate"
      if self.state.grasp_established:
        return "lift"
      if max(self.state.fingertip_normal_force_n, default=0.0) > 0.05:
        return "grasp"
      return "pick_sponge"
    if self.state.wall_normal_force_n > 0:
      return "wiping"
    if max(self.state.fingertip_normal_force_n, default=0.0) > 0.05:
      return "sponge_grasped"
    return "pick_sponge"

  def step(self) -> None:
    if self.task in WHITEBOARD_TASKS:
      self.simulation.phase = "policy"
    before_time = float(self.simulation.data.time)
    try:
      self.simulation.step()
    except TaskGuardViolation:
      # A MuJoCo warning/nonfinite state from the same step takes precedence
      # over a recoverable task guard in strict mode.
      ensure_evaluation_physics(self.simulation, before_time=before_time)
      raise
    ensure_evaluation_physics(self.simulation, before_time=before_time)
    contact_recorder = getattr(self, "_contact_intensity_recorder", None)
    if contact_recorder is not None:
      contact_recorder.commit_step()
    if self.task in WHITEBOARD_TASKS:
      if self.task in HORIZONTAL_WHITEBOARD_TASKS:
        self._update_horizontal_pickup()
      self.maximum_lift_m = max(
        self.maximum_lift_m,
        float(self.simulation.object_pose("eraser")[2] - self.initial_object_pose[2]),
      )
      instant_release = self._whiteboard_release_condition()
      self.release_stable_s = (
        self.release_stable_s + self.simulation.timestep if instant_release else 0.0
      )
      self.state = self._whiteboard_state()
      self._success = self.state.success
    elif self.monitor is None:
      if self.task == "sponge-grasp":
        self._update_sponge()
        self._success = self.state.success
      else:
        self.state = self.simulation.measure()
        self._success = bool(self.simulation.cleaning.success)
    else:
      self.state = self.monitor.update()
      self._success = bool(self.state.success)
    self._guard()
    self._update_staged_evaluation()

  def _whiteboard_state(self) -> WhiteboardPolicyState:
    sim = self.simulation
    remaining = np.asarray(sim.remaining, dtype=float)
    fingertip = np.asarray(sim.forces.read(sim.data).normal_force_n[5:], dtype=float)
    linear_speed = float(np.linalg.norm(sim.object_twist("eraser")[:3]))
    maximum_lift = float(getattr(self, "maximum_lift_m", 0.0))
    released = bool(getattr(self, "release_stable_s", 0.0) >= 0.1)
    maximum_remaining = float(remaining.max(initial=0.0))
    home_verified = bool(
      released
      and np.max(abs(
        sim.data.qpos[sim._arm_qpos["right"]] - sim.task_config.ARM_HOME["right"]
      )) < 0.01
    )
    if self.task in HORIZONTAL_WHITEBOARD_TASKS:
      home_verified = bool(
        home_verified
        and np.max(abs(
          sim.data.qpos[sim._hand_qpos["right"]] - sim.spread_grip
        )) < sim.task_config.OPEN_HAND_MAX_ERROR_RAD
      )
    return WhiteboardPolicyState(
      cleaned_fraction=float(np.mean(1.0 - remaining)),
      maximum_ink_remaining=maximum_remaining,
      board_normal_force_n=float(sim.board_force),
      board_tangent_force_n=float(sim.board_tangent_force),
      table_support_force_n=float(sim.table_force),
      fingertip_normal_force_n=tuple(float(value) for value in fingertip),
      eraser_linear_speed_m_s=linear_speed,
      maximum_lift_m=maximum_lift,
      released_on_table=released,
      success=bool(
        maximum_lift > 0.12 and maximum_remaining <= 1e-9 and released
        and (self.task == "whiteboard-wipe" or (
          home_verified and self.horizontal_pickup_verified
        ))
      ),
      home_return_verified=home_verified,
      pickup_verified=bool(getattr(self, "horizontal_pickup_verified", False)),
    )

  def _update_horizontal_pickup(self) -> None:
    sim = self.simulation
    loads = np.asarray(sim.forces.read(sim.data).normal_force_n[5:], dtype=float)
    prior_grasp = self.horizontal_opposed_grasp_verified
    self.horizontal_opposed_grasp_verified |= bool(loads[0] >= 0.05 and loads[1:].sum() >= 0.15)
    if (
      prior_grasp and sim.table_force <= 0.01
      and sim.object_pose("eraser")[2] - self.initial_object_pose[2] >= 0.12
    ):
      self.horizontal_pickup_verified = True

  def _whiteboard_release_condition(self) -> bool:
    sim = self.simulation
    if self.task in HORIZONTAL_WHITEBOARD_TASKS and (
      not self.horizontal_pickup_verified or np.max(sim.remaining, initial=0.0) > 1e-9
    ):
      return False
    fingertip = np.asarray(sim.forces.read(sim.data).normal_force_n[5:], dtype=float)
    linear_speed = float(np.linalg.norm(sim.object_twist("eraser")[:3]))
    return bool(
      sim.table_force > 0.2 and linear_speed < 0.01 and fingertip.sum() < 0.05
    )

  def _sponge_sample(self) -> dict[str, Any]:
    from kaihand_tactile_env.tasks.sponge_grasp import config
    from kaihand_tactile_env.tasks.vase_wipe.sponge import PIN

    sim = self.simulation
    vertices = sim.data.flexvert_xpos[
      sim._flex_start : sim._flex_start + sim._flex_count
    ]
    force = sim.forces.read(sim.data)
    normal = np.asarray(force.normal_force_n[5:], dtype=float)
    tangent = np.linalg.norm(np.asarray(force.tangent_force_n[5:], dtype=float), axis=1)
    centroid = vertices.mean(axis=0)
    radial = np.linalg.norm(vertices[:, :2] - config.PLATE_TABLE_XY, axis=1)
    return {
      "timestamp": float(sim.data.time),
      "bottom_z": float(vertices[:, 2].min()),
      "pin_z": float(vertices[PIN, 2]),
      "centroid": centroid.copy(),
      "plate_offset": float(np.linalg.norm(centroid[:2] - config.PLATE_TABLE_XY)),
      "maximum_plate_radius": float(radial.max()),
      "normal": normal,
      "tangent": tangent,
      "table_support": float(sim.table_support_force_n),
      "plate_base_support": float(sim.plate_base_support_force_n),
    }

  def _reset_sponge_settle_window(self) -> None:
    self.sponge_settle_start_s = None
    self.sponge_settle_start_impulse_ns = 0.0
    self.sponge_settle_min_heights = None
    self.sponge_settle_max_heights = None

  def _sponge_settle_statistics(self) -> tuple[float, float, np.ndarray]:
    if self.sponge_settle_start_s is None:
      return 0.0, 0.0, np.zeros(3, dtype=float)
    elapsed = float(self.simulation.data.time - self.sponge_settle_start_s)
    mean_support = (
      float(
        self.simulation.plate_base_support_impulse_ns
        - self.sponge_settle_start_impulse_ns
      )
      / elapsed
      if elapsed > 0
      else 0.0
    )
    height_range = self.sponge_settle_max_heights - self.sponge_settle_min_heights
    return elapsed, mean_support, height_range

  def _sponge_state(self, sample: dict[str, Any]) -> SpongePolicyState:
    elapsed, mean_support, height_range = self._sponge_settle_statistics()
    centroid = sample["centroid"]
    return SpongePolicyState(
      timestamp=sample["timestamp"],
      sponge_bottom_z_m=sample["bottom_z"],
      sponge_centroid_xyz_m=tuple(float(value) for value in centroid),
      sponge_plate_center_offset_m=sample["plate_offset"],
      sponge_maximum_plate_radius_m=sample["maximum_plate_radius"],
      table_support_force_n=sample["table_support"],
      plate_base_support_force_n=sample["plate_base_support"],
      plate_base_mean_support_force_n=mean_support,
      fingertip_normal_force_n=tuple(float(value) for value in sample["normal"]),
      fingertip_tangent_force_n=tuple(float(value) for value in sample["tangent"]),
      maximum_lift_clearance_m=self.sponge_maximum_clearance_m,
      five_finger_grasp_stable_s=self.sponge_grasp_stable_s,
      grasp_established=self.sponge_grasp_established,
      lift_established=self.sponge_lift_established,
      transport_contact_valid=self.sponge_transport_contact_valid,
      minimum_carry_clearance_m=(
        float(self.sponge_minimum_carry_clearance_m)
        if np.isfinite(self.sponge_minimum_carry_clearance_m)
        else 0.0
      ),
      maximum_carry_table_support_force_n=(self.sponge_maximum_carry_table_support_n),
      recent_plate_base_mean_support_force_n=(self.sponge_recent_plate_mean_support_n),
      release_allowed=self.sponge_release_allowed,
      released=self.sponge_released,
      settle_stable_s=elapsed,
      settle_pin_height_range_m=float(height_range[0]),
      settle_centroid_height_range_m=float(height_range[1]),
      settle_bottom_height_range_m=float(height_range[2]),
      grasp_patch_enabled=bool(self.simulation.grasp_patch.enabled),
      success=self._success,
    )

  def _update_sponge(self) -> None:
    from kaihand_tactile_env.tasks.sponge_grasp import config

    sim = self.simulation
    now = float(sim.data.time)
    if now + 1.0e-12 < self.sponge_next_audit_sample_s:
      return
    while self.sponge_next_audit_sample_s <= now + 1.0e-12:
      self.sponge_next_audit_sample_s += 0.01
    sample = self._sponge_sample()
    dt = 0.01
    clearance = sample["bottom_z"] - float(sim.table_height)
    self.sponge_maximum_clearance_m = max(self.sponge_maximum_clearance_m, clearance)

    if not self.sponge_grasp_established and not self.sponge_released:
      if np.all(sample["normal"] >= config.MINIMUM_FIVE_FINGER_NORMAL_FORCE_N):
        self.sponge_grasp_stable_s += dt
        if self.sponge_grasp_stable_s >= config.SETTLED_GRASP_CONTACT_SECONDS:
          self.sponge_grasp_established = True
          sim.grasp_patch.enabled = True
      else:
        self.sponge_grasp_stable_s = 0.0

    history = self.sponge_plate_impulse_history
    history.append((now, float(sim.plate_base_support_impulse_ns)))
    cutoff = now - 0.25
    while len(history) > 1 and history[1][0] <= cutoff:
      history.popleft()
    duration = now - history[0][0]
    self.sponge_recent_plate_mean_support_n = (
      (history[-1][1] - history[0][1]) / duration if duration >= 0.24 else 0.0
    )

    normal_max = float(sample["normal"].max(initial=0.0))
    if self.sponge_grasp_established and not self.sponge_release_allowed:
      if np.any(sample["normal"] < config.MINIMUM_FIVE_FINGER_NORMAL_FORCE_N):
        self.sponge_transport_contact_valid = False
      if (
        not self.sponge_lift_established
        and clearance >= 0.12
        and sample["table_support"] <= 0.01
      ):
        self.sponge_lift_established = True
        self.sponge_minimum_carry_clearance_m = clearance
      if self.sponge_lift_established and not self.sponge_placement_started:
        if sample["plate_offset"] <= 0.05 or sample["plate_base_support"] > 0.02:
          self.sponge_placement_started = True
        else:
          self.sponge_minimum_carry_clearance_m = min(
            self.sponge_minimum_carry_clearance_m, clearance
          )
          self.sponge_maximum_carry_table_support_n = max(
            self.sponge_maximum_carry_table_support_n,
            sample["table_support"],
          )
      self.sponge_release_allowed = bool(
        self.sponge_lift_established
        and self.sponge_transport_contact_valid
        and self.sponge_placement_started
        and self.sponge_minimum_carry_clearance_m >= 0.08
        and self.sponge_maximum_carry_table_support_n < 0.05
        and self.sponge_recent_plate_mean_support_n >= 0.08
      )

    if self.sponge_release_allowed and not self.sponge_released:
      if normal_max < 0.10:
        self.sponge_release_stable_s += dt
        if self.sponge_release_stable_s >= 0.10:
          self.sponge_released = True
          sim.grasp_patch.enabled = False
      else:
        self.sponge_release_stable_s = 0.0

    inside_plate = bool(
      sample["plate_offset"] <= 0.035
      and sample["maximum_plate_radius"] <= config.PLATE_INTERIOR_RADIUS_M
      and config.PLATE_INTERIOR_TOP_Z_M - 0.005
      <= sample["bottom_z"]
      <= config.PLATE_INTERIOR_TOP_Z_M + 0.015
    )
    settle_candidate = bool(
      self.sponge_released
      and not sim.grasp_patch.enabled
      and inside_plate
      and sample["table_support"] < 0.05
      and normal_max < 0.10
    )
    heights = np.array(
      [sample["pin_z"], sample["centroid"][2], sample["bottom_z"]],
      dtype=float,
    )
    if settle_candidate:
      if self.sponge_settle_start_s is None:
        self.sponge_settle_start_s = sample["timestamp"]
        self.sponge_settle_start_impulse_ns = float(sim.plate_base_support_impulse_ns)
        self.sponge_settle_min_heights = heights.copy()
        self.sponge_settle_max_heights = heights.copy()
      else:
        self.sponge_settle_min_heights = np.minimum(
          self.sponge_settle_min_heights, heights
        )
        self.sponge_settle_max_heights = np.maximum(
          self.sponge_settle_max_heights, heights
        )
        if np.any(
          self.sponge_settle_max_heights - self.sponge_settle_min_heights > 0.003
        ):
          self.sponge_settle_start_s = sample["timestamp"]
          self.sponge_settle_start_impulse_ns = float(sim.plate_base_support_impulse_ns)
          self.sponge_settle_min_heights = heights.copy()
          self.sponge_settle_max_heights = heights.copy()
    else:
      self._reset_sponge_settle_window()

    elapsed, mean_support, height_range = self._sponge_settle_statistics()
    self._success = bool(
      self.sponge_grasp_established
      and self.sponge_lift_established
      and self.sponge_transport_contact_valid
      and self.sponge_release_allowed
      and elapsed >= config.SETTLE_SECONDS
      and mean_support >= 0.20
      and np.all(height_range <= 0.003)
    )
    self.state = self._sponge_state(sample)

  def _guard(self) -> None:
    sim = self.simulation
    if self.task in {"bulb-screw", "bulb-unscrew"}:
      from kaihand_tactile_env.tasks.bulb_screw import config

      task_guard(
        sim,
        max(self.state.fingertip_normal_force_n, default=0.0)
        > config.MAX_FINGERTIP_LOAD_N,
        "force_guard",
        "bulb fingertip normal force exceeded task limit",
        measured=max(self.state.fingertip_normal_force_n, default=0.0),
        threshold=config.MAX_FINGERTIP_LOAD_N,
        recovery="continue_physics",
      )
      bulb_height = float(sim.object_pose("bulb")[2])
      task_guard(
        sim,
        bulb_height < 0.5,
        "object_dropped",
        "bulb fell below the work surface",
        measured=bulb_height,
        threshold=0.5,
        recovery="continue_physics",
      )
      if self.task == "bulb-unscrew" and self.state.failure_reason is not None:
        task_guard(
          sim,
          True,
          "task_failure",
          str(self.state.failure_reason),
          recovery="continue_physics",
        )
    elif self.task == "usb-unplug":
      from kaihand_tactile_env.tasks.usb_unplug import config

      state = self.state
      for measured, threshold, kind, message in (
        (
          state.withdrawal_resistance_n,
          config.MAX_WITHDRAWAL_RESISTANCE_N,
          "force_guard",
          "USB withdrawal resistance exceeded task limit",
        ),
        (
          state.insertion_resistance_n,
          config.MAX_INSERTION_RESISTANCE_N,
          "force_guard",
          "USB inward resistance exceeded task limit",
        ),
        (
          state.socket_normal_load_n,
          config.MAX_SOCKET_NORMAL_LOAD_N,
          "force_guard",
          "USB socket normal force exceeded task limit",
        ),
        (
          state.wall_normal_load_n,
          config.MAX_SOCKET_WALL_LOAD_N,
          "force_guard",
          "USB socket wall force exceeded task limit",
        ),
        (
          state.maximum_socket_penetration_m,
          config.MAX_SOCKET_PENETRATION_M,
          "penetration_guard",
          "USB socket penetration exceeded task limit",
        ),
      ):
        task_guard(
          sim,
          measured > threshold,
          kind,
          message,
          measured=measured,
          threshold=threshold,
          recovery="continue_physics",
        )
      plug_height = float(sim.object_pose("usb_plug")[2])
      task_guard(
        sim,
        plug_height < 0.5,
        "object_dropped",
        "USB plug fell below the work surface",
        measured=plug_height,
        threshold=0.5,
        recovery="continue_physics",
      )
    elif self.task == "install-ram":
      from kaihand_tactile_env.tasks.install_ram import config

      task_guard(
        sim,
        self.state.axial_resistance_n > config.INSERTION_FORCE_LIMIT_N,
        "force_guard",
        "RAM insertion resistance exceeded task limit",
        measured=self.state.axial_resistance_n,
        threshold=config.INSERTION_FORCE_LIMIT_N,
        recovery="continue_physics",
      )
      task_guard(
        sim,
        self.state.maximum_socket_penetration_m > config.MAX_SOCKET_PENETRATION_M,
        "penetration_guard",
        "RAM socket penetration exceeded task limit",
        measured=self.state.maximum_socket_penetration_m,
        threshold=config.MAX_SOCKET_PENETRATION_M,
        recovery="continue_physics",
      )
      ram_height = float(sim.object_pose("ram")[2])
      task_guard(
        sim,
        ram_height < 0.5,
        "object_dropped",
        "RAM fell below the work surface",
        measured=ram_height,
        threshold=0.5,
        recovery="continue_physics",
      )
    elif self.task == "vase-wipe":
      from kaihand_tactile_env.tasks.vase_wipe import config

      task_guard(
        sim,
        self.state.wall_normal_force_n > config.MAX_CLEAN_NORMAL_N,
        "force_guard",
        "vase wall normal force exceeded cleaning limit",
        measured=self.state.wall_normal_force_n,
        threshold=config.MAX_CLEAN_NORMAL_N,
        recovery="continue_physics",
      )
      sponge_height = float(sim.object_pose("sponge")[2])
      task_guard(
        sim,
        sponge_height < 0.4,
        "object_dropped",
        "sponge fell below the work surface",
        measured=sponge_height,
        threshold=0.4,
        recovery="continue_physics",
      )
    elif self.task == "sponge-grasp":
      maximum_normal = max(self.state.fingertip_normal_force_n, default=0.0)
      maximum_tangent = max(self.state.fingertip_tangent_force_n, default=0.0)
      task_guard(
        sim,
        maximum_normal > 5.0,
        "force_guard",
        "sponge fingertip normal force exceeded task limit",
        measured=maximum_normal,
        threshold=5.0,
        recovery="continue_physics",
      )
      task_guard(
        sim,
        maximum_tangent > 3.0,
        "force_guard",
        "sponge fingertip tangent force exceeded task limit",
        measured=maximum_tangent,
        threshold=3.0,
        recovery="continue_physics",
      )
      task_guard(
        sim,
        self.state.sponge_bottom_z_m < 0.4,
        "object_dropped",
        "sponge fell below the work surface",
        measured=self.state.sponge_bottom_z_m,
        threshold=0.4,
        recovery="continue_physics",
      )
    else:
      from kaihand_tactile_env.tasks.whiteboard_wipe import cleaning

      force_exceeded = bool(
        self.state.board_normal_force_n > cleaning.BOARD_NORMAL_FORCE_GUARD_N
      )
      collision_observed = bool(sim.direct_hand_board_force > 0.5)
      self.whiteboard_force_guard_observed |= force_exceeded
      self.whiteboard_collision_guard_observed |= collision_observed
      guard_function = (
        record_guard_event
        if self.diagnostic_relax_whiteboard_contact_guards
        else task_guard
      )
      if guard_function is record_guard_event:
        if force_exceeded:
          guard_function(
            sim,
            "force_guard",
            "whiteboard normal force exceeded evaluation guard threshold",
            measured=self.state.board_normal_force_n,
            threshold=cleaning.BOARD_NORMAL_FORCE_GUARD_N,
            recovery="continue_physics",
          )
        if collision_observed:
          guard_function(
            sim,
            "collision_guard",
            "hand struck the whiteboard",
            measured=sim.direct_hand_board_force,
            threshold=0.5,
            recovery="continue_physics",
          )
      else:
        guard_function(
          sim,
          force_exceeded,
          "force_guard",
          "whiteboard normal force exceeded evaluation guard threshold",
          measured=self.state.board_normal_force_n,
          threshold=cleaning.BOARD_NORMAL_FORCE_GUARD_N,
          recovery="continue_physics",
        )
        guard_function(
          sim,
          collision_observed,
          "collision_guard",
          "hand struck the whiteboard",
          measured=sim.direct_hand_board_force,
          threshold=0.5,
          recovery="continue_physics",
        )
      eraser_height = float(sim.object_pose("eraser")[2])
      task_guard(
        sim,
        eraser_height < 0.5,
        "object_dropped",
        "eraser fell below the work surface",
        measured=eraser_height,
        threshold=0.5,
        recovery="continue_physics",
      )

  def snapshot(self, _simulation: Any = None) -> dict[str, Any]:
    values = asdict(self.state)
    compact: dict[str, Any] = {"success": self._success, "stage": self.stage()}
    preferred_by_task = {
      "bulb-screw": (
        "engaged",
        "clockwise_turns",
        "insertion_depth_m",
        "lateral_error_m",
        "tilt_rad",
        "thread_error_m",
        "linear_speed_m_s",
        "angular_speed_rad_s",
        "thread_contact_load_n",
        "backstop_load_n",
        "shoulder_contact_load_n",
        "cushion_contact_load_n",
        "bulb_pose_wxyz",
        "fingertip_normal_force_n",
        "right_hand_contact_load_n",
        "right_hand_contact_count",
        "right_hand_nonpad_contact_count",
        "robot_contact_load_n",
        "robot_contact_count",
        "five_finger_contact_duration_s",
        "maximum_five_finger_contact_duration_s",
        "five_finger_contact_fraction",
        "hand_clockwise_torque_nm",
        "capture_armed",
        "capture_ready",
        "capture_depth_margin_m",
        "capture_lateral_margin_m",
        "capture_tilt_margin_rad",
        "capture_speed_margin_m_s",
        "capture_angular_speed_margin_rad_s",
        "capture_contact_margin_n",
        "seated",
        "seated_duration_s",
        "tightening_verified",
        "tightening_stall_duration_s",
        "tightening_peak_torque_nm",
        "bulb_lit",
        "released",
        "release_stable_duration_s",
      ),
      "install-ram": (
        "insertion_depth_m",
        "orientation_error_rad",
        "axial_resistance_n",
        "seated",
      ),
      "vase-wipe": (
        "cleaned_fraction",
        "wall_normal_force_n",
        "wall_tangent_force_n",
        "deformation_mm",
      ),
      "whiteboard-wipe": (
        "cleaned_fraction",
        "maximum_ink_remaining",
        "board_normal_force_n",
        "table_support_force_n",
        "maximum_lift_m",
        "released_on_table",
      ),
      "sponge-grasp": (
        "sponge_plate_center_offset_m",
        "sponge_maximum_plate_radius_m",
        "plate_base_mean_support_force_n",
        "maximum_lift_clearance_m",
        "grasp_established",
        "lift_established",
        "transport_contact_valid",
        "minimum_carry_clearance_m",
        "recent_plate_base_mean_support_force_n",
        "release_allowed",
        "released",
        "settle_stable_s",
      ),
    }
    for task in HORIZONTAL_WHITEBOARD_TASKS:
      preferred_by_task[task] = (
        *preferred_by_task["whiteboard-wipe"], "pickup_verified", "home_return_verified",
      )
    preferred_by_task["usb-unplug"] = (
      "insertion_depth_m", "signed_axial_force_n", "withdrawal_resistance_n",
      "socket_normal_load_n", "grip_force_n", "initially_seated_verified",
      "grasp_verified", "fully_extracted", "placed_on_table", "released", "hand_retreated",
    )
    preferred_by_task["bulb-unscrew"] = (
      "engaged", "bulb_lit", "turns_removed", "insertion_depth_m", "thread_error_m",
      "fingertip_normal_force_n", "initially_installed_verified", "grasp_verified",
      "breakaway_verified", "thread_disengaged", "fully_extracted", "placed_on_table",
      "released", "hand_retreated",
    )
    preferred = preferred_by_task[self.task]
    compact.update({key: values[key] for key in preferred})
    observer = getattr(self, "staged_evaluation", None)
    if observer is not None:
      staged = observer.snapshot()
      compact = {**staged["stages"], "staged_highest_stage": staged["highest_stage"],
                 **compact, "staged_evaluation": staged}
    return compact

  def metadata(self) -> dict[str, Any]:
    guard_metadata = evaluation_guard_report(self.simulation)
    metadata = {
      "name": f"{self.task}-policy-outcome",
      "guard_mode": guard_metadata["guard_mode"],
      "diagnostic_only": guard_metadata["diagnostic_only"],
      "formal_metrics_valid": guard_metadata["formal_metrics_valid"],
      "success_source": (
        "stable five-finger pickup, lifted transport, released inside the "
        "plate, and plate-base support"
        if self.task == "sponge-grasp"
        else (
          "whiteboard cleaned plus physical lift and stable table release"
          if self.task in WHITEBOARD_TASKS
          else (
            "task monitor stable success"
            if self.monitor is not None
            else "ground-truth cleaning completion"
          )
        )
      ),
    }
    if self.task == "bulb-screw":
      metadata["success_source"] = (
        "mechanically seated, loaded tightening verified, released, and stable"
      )
    elif self.task in HORIZONTAL_WHITEBOARD_TASKS:
      metadata["success_source"] = (
        "whiteboard cleaned plus physical lift, stable table release, and arm home return"
      )
    if self.task in WHITEBOARD_TASKS:
      relaxed = self.diagnostic_relax_whiteboard_contact_guards
      metadata.update(
        diagnostic_only=bool(metadata["diagnostic_only"] or relaxed),
        formal_metrics_valid=bool(metadata["formal_metrics_valid"] and not relaxed),
        whiteboard_contact_guards_enabled=not relaxed,
        cleaning_model_has_upper_normal_force_gate=False,
        cleaning_can_accumulate_above_board_force_guard=True,
      )
      if self.task == "whiteboard-wipe-horizontal-line":
        metadata.update(
          ink_pattern=self.simulation.task_config.INK_PATTERN,
          ink_patch_count=self.simulation.task_config.INK_COUNT,
        )
    return metadata

  def report(self) -> dict[str, Any]:
    report = {
      "task": self.task,
      "success": self._success,
      "stage": self.stage(),
      "state": asdict(self.state),
      "initial_object_pose_wxyz": self.initial_object_pose.tolist(),
      "final_object_pose_wxyz": self.simulation.object_pose(self.object_name).tolist(),
      "initial_randomization": getattr(
        self.simulation, "initial_position_randomization", None
      ),
      "sponge_xy_offset_m": (
        self.simulation.sponge_xy_offset_m.tolist()
        if hasattr(self.simulation, "sponge_xy_offset_m")
        else None
      ),
      "stain_randomization": getattr(self.simulation, "stain_randomization", None),
      "ink_randomization": getattr(self.simulation, "ink_randomization", None),
    }
    report.update(evaluation_guard_report(self.simulation))
    observer = getattr(self, "staged_evaluation", None)
    if observer is not None:
      report["staged_evaluation"] = observer.report()
    if self.task in WHITEBOARD_TASKS:
      from kaihand_tactile_env.tasks.whiteboard_wipe import cleaning

      sim = self.simulation
      report["contact_guard_diagnostics"] = {
        "mode": (
          "observe_only"
          if self.diagnostic_relax_whiteboard_contact_guards
          else "enforce"
        ),
        "board_normal_force_guard_threshold_n": (cleaning.BOARD_NORMAL_FORCE_GUARD_N),
        "board_normal_force_exceeded": self.whiteboard_force_guard_observed,
        "peak_board_normal_force_n": float(sim.peak_board_force),
        "direct_hand_board_force_guard_threshold_n": 0.5,
        "direct_hand_board_contact_observed": (
          self.whiteboard_collision_guard_observed
        ),
        "peak_direct_hand_board_force_n": float(sim.peak_direct_hand_board_force),
        "cleaning_model_has_upper_normal_force_gate": False,
        "cleaning_can_accumulate_above_board_force_guard": True,
        "cleaning_force_semantics": (
          "8 N is an evaluation safety/out-of-distribution guard, not a "
          "material erasure ceiling; pad/board loaded sliding above it can "
          "still accumulate measured friction work, stroke, and contact time"
        ),
        "nonfinite_state_guard_enabled": True,
        "object_drop_guard_enabled": True,
        "solver_warning_guard_enabled": True,
      }
    return report


def create_task_policy_adapter(
  task: str,
  seed: int,
  *,
  guard_mode: str = "strict",
  diagnostic_relax_whiteboard_contact_guards: bool = False,
) -> TaskPolicyAdapter:
  return TaskPolicyAdapter(
    task,
    seed,
    guard_mode=guard_mode,
    diagnostic_relax_whiteboard_contact_guards=(
      diagnostic_relax_whiteboard_contact_guards
    ),
  )
