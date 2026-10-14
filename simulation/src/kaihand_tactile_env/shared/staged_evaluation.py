"""Versioned, read-only outcome measurements, independent of task mechanics.

Call ``update`` after *every* physics substep and after the native task observer
has refreshed its state.  This module never advances physics, resets an object,
changes controls, runs an expert, or replaces the native ``task_success`` label.
Missing observations are null, not guessed from controller phases.  Whiteboard
pickup is a stage, not a substitute for its separate continuous wipe metrics.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Mapping

import mujoco
import numpy as np

EVALUATION_VERSION = "staged_v1"
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
WHITEBOARD_TASKS = (
  "whiteboard-wipe", "whiteboard-wipe-horizontal", "whiteboard-wipe-horizontal-line",
)
STAGE_NAMES = {
  "bulb-screw": ("picked_up", "held_near_socket", "threaded_and_seated"),
  "bulb-unscrew": ("fully_extracted", "released_stably_on_table"),
  "install-ram": ("picked_up_and_held_near_slot", "inserted_and_seated"),
  "usb-insert": ("picked_up", "held_near_socket", "inserted_and_seated"),
  "usb-unplug": ("fully_extracted", "released_stably_on_table"),
  "poker-draw": ("dragged_on_table", "stable_grasp", "terminal_hold_over_3s"),
  **{task: ("picked_up",) for task in WHITEBOARD_TASKS},
}
OBJECT_NAMES = {
  "bulb-screw": "bulb", "bulb-unscrew": "bulb", "install-ram": "ram",
  "usb-insert": "usb_plug", "usb-unplug": "usb_plug", "poker-draw": "card",
  **{task: "eraser" for task in WHITEBOARD_TASKS},
}


@dataclass(frozen=True)
class StageThresholds:
  """Fixed SI-unit thresholds; serialize alongside every rollout's result."""

  lift_m: float = 0.02
  near_mouth_m: float = 0.05
  grasp_hold_s: float = 0.15
  finger_load_n: float = 0.025
  maximum_opposition_cosine: float = 0.50
  near_hold_s: float = 0.15
  seating_hold_s: float = 0.20
  clear_hold_s: float = 0.10
  table_placement_hold_s: float = 0.05
  release_hold_s: float = 0.20
  minimum_table_support_n: float = 0.01
  maximum_release_linear_speed_m_s: float = 0.02
  maximum_release_angular_speed_rad_s: float = 0.20
  drag_displacement_m: float = 0.02
  drag_hold_s: float = 0.05
  maximum_drag_clearance_m: float = 0.005
  minimum_transport_m: float = 0.02
  poker_terminal_hold_s: float = 3.0
  poker_terminal_position_tolerance_m: float = 0.03
  poker_terminal_face_cosine: float = 0.80
  poker_terminal_linear_speed_m_s: float = 0.02
  poker_terminal_angular_speed_rad_s: float = 0.20
  observation_gap_steps: float = 1.5

  def __post_init__(self):
    for name, value in asdict(self).items():
      if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    if self.maximum_opposition_cosine > 1 or self.poker_terminal_face_cosine > 1:
      raise ValueError("cosine thresholds must not exceed one")


def _json_safe(value):
  if is_dataclass(value) and not isinstance(value, type):
    value = asdict(value)
  if isinstance(value, Mapping):
    return {str(key): _json_safe(item) for key, item in value.items()}
  if isinstance(value, (list, tuple, np.ndarray)):
    return [_json_safe(item) for item in value]
  if isinstance(value, (bool, np.bool_)):
    return bool(value)
  if isinstance(value, (int, np.integer)):
    return int(value)
  if isinstance(value, (float, np.floating)):
    return float(value) if np.isfinite(value) else None
  if value is None or isinstance(value, str):
    return value
  raise TypeError(f"cannot persist observation of type {type(value).__name__}")


def _all(*conditions):
  """Three-valued AND: an absent observation must not become success."""
  if any(condition is False for condition in conditions):
    return False
  return None if any(condition is None for condition in conditions) else True


def _ge(measurements, key, threshold):
  value = measurements.get(key)
  return None if value is None else bool(value >= threshold)


def _le(measurements, key, threshold):
  value = measurements.get(key)
  return None if value is None else bool(value <= threshold)


class _SimulationMeasurements:
  """Only existing MuJoCo caches and solver contact forces are read here.

  In particular, there is deliberately no mj_forward: refreshing derived data
  belongs to the simulation's existing observer, not to an outcome metric.
  Native seating predicates carry the exact scene-specific bore/crest geometry.
  """

  def __init__(self, task, simulation, thresholds):
    self.task, self.sim, self.thresholds = task, simulation, thresholds
    self.model = simulation.model
    self.object_name = OBJECT_NAMES[task]
    self.object_body = self.model.body(self.object_name).id
    object_bodies = self._descendants(self.object_body)
    self.object_geoms = {
      geom for geom in range(self.model.ngeom)
      if int(self.model.geom_bodyid[geom]) in object_bodies
    }
    self.geom_names = {
      geom: self.model.geom(geom).name or "" for geom in range(self.model.ngeom)
    }
    self.body_names = {
      body: self.model.body(body).name or "" for body in range(self.model.nbody)
    }
    robot_bodies = set()
    for name in ("left_arm_base", "right_arm_base", "right_hand_mount"):
      root = self._named_id("body", name)
      if root is not None:
        robot_bodies.update(self._descendants(root))
    self.robot_geoms = {
      geom for geom in range(self.model.ngeom)
      if int(self.model.geom_bodyid[geom]) in robot_bodies
      or self.body_names[int(self.model.geom_bodyid[geom])].startswith(("hand_r_", "hand_l_"))
    }
    self.hand_geoms = {
      geom for geom in self.robot_geoms
      if self.body_names[int(self.model.geom_bodyid[geom])].startswith("hand_r_")
    }
    self.finger_geoms = {}
    for geom, name in self.geom_names.items():
      if "tactile_pad_col" not in name:
        continue
      for finger in FINGERS:
        if name.startswith(f"hand_r_{finger}_"):
          self.finger_geoms[geom] = finger
          break
    self.table_geoms = {
      geom for geom, name in self.geom_names.items()
      if name in {"tabletop", "poker_table_top"}
    }
    self.initial_support_geoms = {
      geom for geom in range(self.model.ngeom)
      if self.body_names[int(self.model.geom_bodyid[geom])].startswith(
        ("bulb_pickup_cradle", "ram_presentation_stand")
      )
    }
    self.socket_geoms = {
      geom for geom, name in self.geom_names.items()
      if name.startswith(("usb_socket_", "ram_socket_", "bulb_socket_"))
    }
    site_names = {
      "bulb-screw": ("bulb_tip", "bulb_socket_mouth"),
      "bulb-unscrew": ("bulb_tip", "bulb_socket_mouth"),
      "install-ram": ("ram_bottom", "ram_socket_mouth"),
      "usb-insert": ("usb_plug_tip", "usb_socket_mouth"),
      "usb-unplug": ("usb_plug_tip", "usb_socket_mouth"),
    }.get(task, (None, None))
    self.tip = self._named_id("site", site_names[0])
    self.mouth = self._named_id("site", site_names[1])
    self.card_geom = self._named_id("geom", "card_core_geom")
    # Poker has its own raised playing surface in addition to the workcell
    # tabletop.  Never use unordered aggregate contact geoms as its height
    # reference: the lower worktop makes an on-table card appear airborne.
    self.poker_table_geom = self._named_id("geom", "poker_table_top")
    self.head_camera = self._named_id("camera", "head")
    self.initial_position = np.asarray(simulation.object_pose(self.object_name)[:3]).copy()
    self.initial_time_s = float(simulation.data.time)
    self._integrated_contacts = None

  def _named_id(self, kind, name):
    if name is None:
      return None
    try:
      return int(getattr(self.model, kind)(name).id)
    except (KeyError, ValueError):
      return None

  def _descendants(self, root):
    selected = {root}
    for body in range(root + 1, self.model.nbody):
      if int(self.model.body_parentid[body]) in selected:
        selected.add(body)
    return selected

  def capture_integrated_contacts(self, simulation, before_time):
    """Cache integrated contacts from the existing pre-forward observer hook."""
    if simulation is not self.sim:
      raise ValueError("contact sample belongs to another simulation")
    end_time = float(simulation.data.time)
    duration = end_time - float(before_time)
    if not np.isfinite(duration) or abs(duration - simulation.timestep) > 1e-10:
      raise ValueError("integrated contact sample must cover exactly one physics step")
    measured = self.measure(_integrated_capture=True)
    fields = (
      "finger_load_n", "finger_contact_normal_vectors_n", "opposition_cosine",
      "opposed_grasp_now", "contacted_by_hand_now", "support_free_now",
      *(f"{name}_{suffix}" for name in ("hand", "robot", "table", "initial_support", "socket")
        for suffix in ("contact_count", "normal_load_n")),
    )
    self._integrated_contacts = {
      "end_time_s": end_time, "start_time_s": float(before_time),
      "fields": {name: measured[name] for name in fields},
    }

  def measure(self, task_state=None, *, _integrated_capture=False):
    sim, model, data = self.sim, self.model, self.sim.data
    pose = np.asarray(sim.object_pose(self.object_name), dtype=float)
    twist = np.asarray(sim.object_twist(self.object_name), dtype=float)
    names = ("hand", "robot", "table", "initial_support", "socket")
    counts = dict.fromkeys(names, 0)
    loads = dict.fromkeys(names, 0.0)
    finger_loads = dict.fromkeys(FINGERS, 0.0)
    finger_vectors = {finger: np.zeros(3) for finger in FINGERS}
    wrench = np.empty(6)
    for index in range(data.ncon):
      contact = data.contact[index]
      geom1, geom2 = int(contact.geom1), int(contact.geom2)
      if geom1 in self.object_geoms:
        other, sign = geom2, -1.0
      elif geom2 in self.object_geoms:
        other, sign = geom1, 1.0
      else:
        continue
      mujoco.mj_contactForce(model, data, index, wrench)
      load = abs(float(wrench[0]))
      for name, selected in (
        ("hand", self.hand_geoms), ("robot", self.robot_geoms),
        ("table", self.table_geoms), ("initial_support", self.initial_support_geoms),
        ("socket", self.socket_geoms),
      ):
        if other in selected:
          counts[name] += 1
          loads[name] += load
      finger = self.finger_geoms.get(other)
      if finger is not None:
        finger_loads[finger] += load
        finger_vectors[finger] += sign * np.asarray(contact.frame).reshape(3, 3)[0] * load
    thumb = finger_vectors["thumb"]
    opposing = sum((finger_vectors[finger] for finger in FINGERS[1:]), np.zeros(3))
    denominator = float(np.linalg.norm(thumb) * np.linalg.norm(opposing))
    opposition_cosine = None if denominator <= 1e-12 else float(thumb @ opposing / denominator)
    opposed = bool(
      finger_loads["thumb"] >= self.thresholds.finger_load_n
      and sum(finger_loads[finger] for finger in FINGERS[1:]) >= self.thresholds.finger_load_n
      and opposition_cosine is not None
      and opposition_cosine <= self.thresholds.maximum_opposition_cosine
    )
    observation = {
      "time_s": float(data.time), "simulation_time_s": float(data.time),
      "object_name": self.object_name,
      "object_pose_wxyz": pose.tolist(), "object_position_xyz_m": pose[:3].tolist(),
      "initial_object_position_xyz_m": self.initial_position.tolist(),
      "lift_from_initial_m": float(pose[2] - self.initial_position[2]),
      "planar_displacement_m": float(np.linalg.norm(pose[:2] - self.initial_position[:2])),
      "linear_speed_m_s": float(np.linalg.norm(twist[:3])),
      "angular_speed_rad_s": float(np.linalg.norm(twist[3:])),
      "object_twist_linear_angular": twist.tolist(),
      "finger_load_n": finger_loads,
      "finger_contact_normal_vectors_n": {finger: vector.tolist() for finger, vector in finger_vectors.items()},
      "opposition_cosine": opposition_cosine,
      "opposed_grasp_now": opposed,
      "contacted_by_hand_now": bool(counts["hand"] > 0 and loads["hand"] >= self.thresholds.finger_load_n),
      "support_free_now": bool(counts["table"] == 0 and counts["initial_support"] == 0),
      "work_end_xyz_m": None, "mouth_position_xyz_m": None,
      "relative_work_end_socket_xyz_m": None, "near_mouth_distance_m": None,
      "table_clearance_m": None, "table_clearance_reference_geom": None,
      "seated_now": None, "fully_clear_now": None,
      "thread_engaged_now": None, "native_task_state": None,
      "poker_terminal_position_error_m": None,
      "poker_face_to_head_cosine": None, "poker_face_to_robot_cosine": None,
      "measurement_source": "mujoco_solver_and_geometry_read_only",
      "contact_force_source": "current_mujoco_solver_cache",
      "contact_interval_start_s": None, "contact_interval_end_s": None,
      "missing_measurements": [],
    }
    for name in names:
      observation[f"{name}_contact_count"] = counts[name]
      observation[f"{name}_normal_load_n"] = loads[name]
    if self.task in WHITEBOARD_TASKS and not _integrated_capture:
      sample = self._integrated_contacts
      if sample is not None and abs(sample["end_time_s"] - float(data.time)) < 1e-12:
        observation.update(sample["fields"])
        observation.update(
          contact_force_source="integrated_solver_contacts_before_post_step_mj_forward",
          contact_interval_start_s=sample["start_time_s"],
          contact_interval_end_s=sample["end_time_s"],
        )
      elif float(data.time) > self.initial_time_s + 1e-12:
        # A refreshed instantaneous force is not evidence for the whole step.
        observation["opposed_grasp_now"] = None
        observation["missing_measurements"].append("integrated_whiteboard_grasp_contacts")
    if self.tip is not None and self.mouth is not None:
      tip, mouth = np.asarray(data.site_xpos[self.tip]), np.asarray(data.site_xpos[self.mouth])
      local = np.asarray(data.site_xmat[self.mouth]).reshape(3, 3).T @ (tip - mouth)
      observation.update(
        work_end_xyz_m=tip.tolist(), mouth_position_xyz_m=mouth.tolist(),
        relative_work_end_socket_xyz_m=local.tolist(),
        near_mouth_distance_m=float(np.linalg.norm(tip - mouth)),
      )
    if self.task.startswith("bulb-"):
      observation["thread_engaged_now"] = bool(sim.thread_engaged)
    if task_state is not None:
      state = _json_safe(task_state)
      if not isinstance(state, dict):
        raise TypeError("task_state must be a dataclass or mapping")
      timestamp = state.get("timestamp", float(data.time))
      if timestamp is not None and abs(timestamp - float(data.time)) <= 1.5 * sim.timestep + 1e-12:
        observation["native_task_state"] = state
        for field, target in (("seated", "seated_now"), ("fully_clear_now", "fully_clear_now")):
          if field in state:
            observation[target] = bool(state[field]) if state[field] is not None else None
      else:
        observation["missing_measurements"].append("native_task_state_timestamp_mismatch")
    if observation["fully_clear_now"] is None and self.task in {"usb-unplug", "bulb-unscrew"}:
      local = observation["relative_work_end_socket_xyz_m"]
      if local is not None:
        if self.task == "usb-unplug":
          from ..tasks.usb_unplug import config
          observation["fully_clear_now"] = bool(local[0] <= config.FULL_CLEAR_DEPTH_M and counts["socket"] == 0)
        else:
          from ..tasks.bulb_unscrew import config
          observation["fully_clear_now"] = bool(
            not sim.thread_engaged and -local[2] <= config.FULL_CLEAR_DEPTH_M
            and counts["socket"] == 0 and loads["socket"] <= config.FULL_CLEAR_SOCKET_LOAD_N
          )
    if self.task == "poker-draw":
      rotation = np.asarray(data.xmat[self.object_body]).reshape(3, 3)
      normal = -rotation[:, 2]
      observation["poker_terminal_position_error_m"] = float(np.linalg.norm(pose[:3] - [0.55, -0.10, 1.05]))
      observation["poker_face_to_robot_cosine"] = float(-normal[0])
      if self.head_camera is not None:
        direction = np.asarray(data.cam_xpos[self.head_camera]) - pose[:3]
        observation["poker_face_to_head_cosine"] = float(normal @ direction / max(float(np.linalg.norm(direction)), 1e-12))
      if self.card_geom is None:
        observation["missing_measurements"].append("card_core_geom")
      if self.poker_table_geom is None:
        observation["missing_measurements"].append("poker_table_top_geom")
      if self.card_geom is not None and self.poker_table_geom is not None:
        table = self.poker_table_geom
        card_rotation = np.asarray(data.geom_xmat[self.card_geom]).reshape(3, 3)
        half_extent = float(np.abs(card_rotation[2]) @ model.geom_size[self.card_geom])
        table_rotation = np.asarray(data.geom_xmat[table]).reshape(3, 3)
        table_top = float(data.geom_xpos[table, 2] + np.abs(table_rotation[2]) @ model.geom_size[table])
        observation["table_clearance_m"] = float(data.geom_xpos[self.card_geom, 2] - half_extent - table_top)
        observation["table_clearance_reference_geom"] = "poker_table_top"
    if self.task in {"bulb-screw", "install-ram", "usb-insert"} and observation["seated_now"] is None:
      observation["missing_measurements"].append("native_seating_predicate")
    return _json_safe(observation)


class StagedEvaluationMonitor:
  """Ordered, cumulative stage achievement with continuous physics-time holds.

  ``simulation=None`` enables offline replay through ``observe``; runtime
  ``update`` requires a real simulation.  Sparse replay cannot prove a hold.
  Attach area/cleaning reports using ``attach_continuous_metrics`` separately.
  """

  def __init__(self, task, simulation=None, *, version=EVALUATION_VERSION, thresholds=None):
    if task not in STAGE_NAMES:
      raise ValueError(f"unsupported staged evaluation task: {task!r}")
    if version != EVALUATION_VERSION:
      raise ValueError(f"unsupported staged evaluation version: {version!r}")
    self.task, self.simulation, self.version = task, simulation, version
    self.thresholds = thresholds if isinstance(thresholds, StageThresholds) else StageThresholds(**(thresholds or {}))
    self._adapter = None if simulation is None else _SimulationMeasurements(task, simulation, self.thresholds)
    self.stage_names = STAGE_NAMES[task]
    self.stages = {name: False for name in self.stage_names}
    self.first_times = {name: None for name in self.stage_names}
    self._last_time = None
    self._since = {}
    self._durations = {}
    self._conditions = {}
    self._measurements = {}
    self._continuous_metrics = {}
    self._held_free_origin = None
    self._transport_verified = False
    self._placement_verified = False
    self._maximum_transport_m = 0.0
    self.observation_gap_count = 0
    self.observations = 0

  def attach_continuous_metrics(self, metrics: Mapping[str, Any]):
    """Attach another read-only observer's report without changing stages."""
    self._continuous_metrics = _json_safe(metrics)

  def capture_integrated_contacts(self, simulation, before_time):
    """Read-only hook, chain after an existing whiteboard contact recorder."""
    if self._adapter is None:
      raise ValueError("integrated contact capture requires a real simulation")
    self._adapter.capture_integrated_contacts(simulation, before_time)

  def update(self, simulation=None, *, task_state=None, return_snapshot=True):
    if simulation is not None and simulation is not self.simulation:
      raise ValueError("monitor cannot move between simulations or episodes")
    if self._adapter is None:
      raise ValueError("runtime update requires a simulation; use observe for replay")
    return self.observe(
      self._adapter.measure(task_state), timestep=self.simulation.timestep,
      return_snapshot=return_snapshot,
    )

  def _hold(self, name, condition, time_s):
    self._conditions[name] = condition
    if condition is not True:
      self._since.pop(name, None)
      self._durations[name] = 0.0
    else:
      self._since.setdefault(name, time_s)
      self._durations[name] = max(0.0, time_s - self._since[name])
    return self._durations[name]

  def _latch(self, index, condition, time_s):
    name = self.stage_names[index]
    predecessors = all(self.stages[previous] for previous in self.stage_names[:index])
    if condition is True and predecessors and not self.stages[name]:
      self.stages[name] = True
      self.first_times[name] = time_s

  def observe(self, measurements: Mapping[str, Any], *, timestep: float, return_snapshot=True):
    """Replay/test entry point accepting persisted *physical* measurements."""
    m = _json_safe(measurements)
    time_s = m.get("time_s")
    if time_s is None or not np.isfinite(timestep) or timestep <= 0:
      raise ValueError("a finite physics timestamp and positive timestep are required")
    if self._last_time is not None:
      delta = time_s - self._last_time
      if delta < -1e-12:
        raise ValueError("physics time rewound; create a new monitor for a new episode")
      if delta > self.thresholds.observation_gap_steps * timestep + 1e-12:
        self._since.clear()
        self._durations.clear()
        self._held_free_origin = None
        self.observation_gap_count += 1
    self._last_time, self._measurements = time_s, m
    self.observations += 1
    t = self.thresholds
    grasp = m.get("opposed_grasp_now")
    grasp_s = self._hold("stable_grasp", grasp, time_s)
    stable_grasp = bool(grasp is True and grasp_s >= t.grasp_hold_s - 1e-10)
    lifted = _all(grasp, _ge(m, "lift_from_initial_m", t.lift_m), m.get("support_free_now"))
    lift_s = self._hold("stable_lift", lifted, time_s)
    picked_up = bool(lifted is True and lift_s >= t.grasp_hold_s - 1e-10)
    near = _all(grasp, _le(m, "near_mouth_distance_m", t.near_mouth_m))
    near_s = self._hold("held_near_mouth", near, time_s)
    seated = m.get("seated_now")
    seated_s = self._hold("mechanically_seated", seated, time_s)

    if self.task in {"bulb-screw", "usb-insert"}:
      self._latch(0, picked_up, time_s)
      self._latch(1, bool(stable_grasp and near is True and near_s >= t.near_hold_s - 1e-10), time_s)
      engaged = m.get("thread_engaged_now") if self.task == "bulb-screw" else True
      self._latch(2, bool(seated is True and engaged is True and seated_s >= t.seating_hold_s - 1e-10), time_s)
    elif self.task == "install-ram":
      self._hold("ram_pickup_latched", picked_up or self._conditions.get("ram_pickup_latched") is True, time_s)
      lifted_before = self._conditions["ram_pickup_latched"] is True
      self._latch(0, bool(lifted_before and stable_grasp and near is True and near_s >= t.near_hold_s - 1e-10), time_s)
      self._latch(1, bool(seated is True and seated_s >= t.seating_hold_s - 1e-10), time_s)
    elif self.task in {"bulb-unscrew", "usb-unplug"}:
      clear = m.get("fully_clear_now")
      clear_s = self._hold("fully_clear", clear, time_s)
      self._latch(0, bool(clear is True and clear_s >= t.clear_hold_s - 1e-10), time_s)
      carried = _all(grasp, clear, m.get("support_free_now"))
      carry_s = self._hold("held_free_after_extraction", carried, time_s)
      position = m.get("object_position_xyz_m")
      if carried is not True:
        self._held_free_origin = None
      if carried is True and carry_s >= t.grasp_hold_s - 1e-10 and position is not None:
        if self._held_free_origin is None:
          self._held_free_origin = np.asarray(position, dtype=float)
        travelled = float(np.linalg.norm(np.asarray(position) - self._held_free_origin))
        self._maximum_transport_m = max(self._maximum_transport_m, travelled)
        self._transport_verified |= travelled >= t.minimum_transport_m
      supported_while_held = _all(
        clear, grasp, bool(self._transport_verified),
        _ge(m, "table_contact_count", 1),
        _ge(m, "table_normal_load_n", t.minimum_table_support_n),
        _le(m, "linear_speed_m_s", t.maximum_release_linear_speed_m_s),
        _le(m, "angular_speed_rad_s", t.maximum_release_angular_speed_rad_s),
      )
      placement_s = self._hold("controlled_table_placement", supported_while_held, time_s)
      if supported_while_held is True and placement_s >= t.table_placement_hold_s - 1e-10:
        self._placement_verified = True
      release = _all(
        clear, bool(self.stages[self.stage_names[0]] and self._transport_verified and self._placement_verified),
        _ge(m, "table_contact_count", 1),
        _ge(m, "table_normal_load_n", t.minimum_table_support_n),
        _le(m, "robot_contact_count", 0),
        _le(m, "linear_speed_m_s", t.maximum_release_linear_speed_m_s),
        _le(m, "angular_speed_rad_s", t.maximum_release_angular_speed_rad_s),
      )
      release_s = self._hold("stable_table_release", release, time_s)
      self._latch(1, bool(release is True and release_s >= t.release_hold_s - 1e-10), time_s)
    elif self.task == "poker-draw":
      drag = _all(
        m.get("contacted_by_hand_now"), _ge(m, "table_contact_count", 1),
        _ge(m, "planar_displacement_m", t.drag_displacement_m),
        _le(m, "table_clearance_m", t.maximum_drag_clearance_m),
      )
      drag_s = self._hold("effective_table_drag", drag, time_s)
      self._latch(0, bool(drag is True and drag_s >= t.drag_hold_s - 1e-10), time_s)
      self._latch(1, stable_grasp, time_s)
      terminal = _all(
        bool(self.stages[self.stage_names[1]]), grasp,
        _ge(m, "table_clearance_m", t.lift_m),
        _le(m, "table_contact_count", 0),
        _le(m, "poker_terminal_position_error_m", t.poker_terminal_position_tolerance_m),
        _ge(m, "poker_face_to_head_cosine", t.poker_terminal_face_cosine),
        _ge(m, "poker_face_to_robot_cosine", t.poker_terminal_face_cosine),
        _le(m, "linear_speed_m_s", t.poker_terminal_linear_speed_m_s),
        _le(m, "angular_speed_rad_s", t.poker_terminal_angular_speed_rad_s),
      )
      hold_s = self._hold("terminal_hold", terminal, time_s)
      self._latch(2, bool(terminal is True and hold_s > t.poker_terminal_hold_s + 1e-10), time_s)
    else:
      self._latch(0, picked_up, time_s)
    return self.snapshot() if return_snapshot else None

  def snapshot(self):
    highest = sum(bool(value) for value in self.stages.values())
    success = None if self.task in WHITEBOARD_TASKS else bool(self.stages[self.stage_names[-1]])
    return deepcopy({
      "version": self.version, "task": self.task, "stage_count": len(self.stage_names),
      "stages": {
        f"stage{index + 1}": self.stages[name] for index, name in enumerate(self.stage_names)
      },
      "stage_definitions": {
        f"stage{index + 1}": name for index, name in enumerate(self.stage_names)
      },
      "named_stages": dict(self.stages), "stage_flags": {
        f"stage{index + 1}": self.stages[name] for index, name in enumerate(self.stage_names)
      },
      "highest_stage": highest, "success": success, "complete_success": success,
      "first_times": {
        f"stage{index + 1}": self.first_times[name] for index, name in enumerate(self.stage_names)
      },
      "named_first_times": dict(self.first_times),
      "thresholds": asdict(self.thresholds),
      "time_s": self._last_time, "simulation_time_s": self._last_time,
      "measurements": self._measurements,
      "instantaneous_conditions": dict(self._conditions),
      "continuous_durations_s": dict(self._durations),
      "continuous_metrics": self._continuous_metrics,
      "transport_verified": bool(self._transport_verified),
      "controlled_table_placement_verified": bool(self._placement_verified),
      "maximum_held_transport_m": self._maximum_transport_m,
      "observations": self.observations, "observation_gap_count": self.observation_gap_count,
      "native_task_success_preserved": True, "expert_trajectory_used": False,
      "simulation_state_mutated": False,
      "stage_achievement_policy": "ordered_cumulative_ever_reached",
      "time_basis": "mujoco_data_time_seconds",
      "whiteboard_complete_success_defined": False if self.task in WHITEBOARD_TASKS else None,
      "failure_stage": None if success is True else (
        self.stage_names[highest] if highest < len(self.stage_names) else None
      ),
    })

  def report(self):
    return self.snapshot()
