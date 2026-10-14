"""Synchronized HDF5 episode recording and schema validation."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .config import (
  OBJECT_NAMES,
  SIDES,
  WRIST_FORCE_SENSOR_NAMES,
  WRIST_FT_SITE_NAMES,
  WRIST_TORQUE_SENSOR_NAMES,
  WorkcellConfig,
  model_fingerprint,
)
from .physics_integrity import check_physics_clock
from .rendering import WorkcellRenderer
from .runtime_contract import simulation_runtime_contract
from .simulation import ArmHandSimulation
from .tactile import (
  GenesisProbeTactileProvider,
  SolverContactTactileProvider,
  TactileProvider,
  raw_contact_snapshot,
)

SCHEMA_VERSION = "kaihand_tactile_episode_v1"
WRIST_WRENCH_SCHEMA_VERSION = "kaihand_bimanual_wrist_wrench_v1"
TASK_ISOLATED_MODEL_LAYOUT = "task-isolated-v1"
LEGACY_COMBINED_MODEL_LAYOUT = "legacy-combined-v1"
POKER_FORCE_RECORDING_CONTRACT = "poker_draw_per_finger_press_shear_v5"
TERMINAL_LINEAR_SPEED_THRESHOLD = 0.02
TERMINAL_ANGULAR_SPEED_THRESHOLD = 0.2
TERMINAL_STABLE_DURATION = 0.1
_MIN_TERMINAL_WINDOW_SAMPLES = 5
_TERMINAL_VALUE_TOLERANCE = 1.0e-9
_CONTACT_FORCE_SHAPES = {
  "normal_force_n": (),
  "normal_force_world_n": (3,),
  "force_world_n": (3,),
  "tangent_force_world_n": (3,),
  "tangent_force_n": (2,),
  "tangent_load_n": (),
  "normal_taxel_force_n": (7, 5),
  "tangent_taxel_force_n": (7, 5, 2),
  "tangent_taxel_load_n": (7, 5),
  "tangent_basis_world": (2, 3),
  "normal_axis_world": (3,),
  "contact_count": (),
}


@dataclass(frozen=True)
class ValidationReport:
  valid: bool
  errors: tuple[str, ...]
  warnings: tuple[str, ...]
  state_samples: int
  camera_samples: dict[str, int]
  duration_seconds: float


@dataclass(frozen=True)
class TerminalStability:
  """Event-driven result after an object remains below velocity thresholds."""

  elapsed_seconds: float
  stable_seconds: float
  linear_speed: float
  angular_speed: float
  steps: int


@dataclass(frozen=True)
class WristWrenchSample:
  """Raw bimanual wrist load in sensor-local and world coordinates."""

  origin_world_m: np.ndarray
  world_from_sensor_rotation: np.ndarray
  force_local_n: np.ndarray
  torque_local_nm: np.ndarray
  force_world_n: np.ndarray
  torque_world_nm: np.ndarray


class WristWrenchSensor:
  """Read the shared left/right MuJoCo force and torque sensors."""

  def __init__(self, model: Any) -> None:
    sensor_addresses: list[tuple[int, int]] = []
    site_ids: list[int] = []
    for side, site_name, force_name, torque_name in zip(
      SIDES,
      WRIST_FT_SITE_NAMES,
      WRIST_FORCE_SENSOR_NAMES,
      WRIST_TORQUE_SENSOR_NAMES,
      strict=True,
    ):
      site = model.site(site_name)
      force = model.sensor(force_name)
      torque = model.sensor(torque_name)
      if int(force.dim[0]) != 3 or int(torque.dim[0]) != 3:
        raise RuntimeError(f"{side} wrist F/T sensors must each have dimension 3")
      site_ids.append(int(site.id))
      sensor_addresses.append((int(force.adr[0]), int(torque.adr[0])))
    self.site_ids = np.asarray(site_ids, dtype=np.int32)
    self.sensor_addresses = tuple(sensor_addresses)

  def read(self, data: Any) -> WristWrenchSample:
    rotations = np.asarray(data.site_xmat[self.site_ids]).reshape(2, 3, 3).copy()
    origins = np.asarray(data.site_xpos[self.site_ids]).copy()
    force_local = np.empty((2, 3), dtype=np.float64)
    torque_local = np.empty((2, 3), dtype=np.float64)
    for index, (force_address, torque_address) in enumerate(self.sensor_addresses):
      force_local[index] = data.sensordata[force_address : force_address + 3]
      torque_local[index] = data.sensordata[torque_address : torque_address + 3]
    force_world = np.einsum("sij,sj->si", rotations, force_local)
    torque_world = np.einsum("sij,sj->si", rotations, torque_local)
    values = (origins, rotations, force_local, torque_local, force_world, torque_world)
    if not all(np.isfinite(value).all() for value in values):
      raise RuntimeError("wrist F/T sensor sample contains non-finite values")
    return WristWrenchSample(
      origin_world_m=origins,
      world_from_sensor_rotation=rotations,
      force_local_n=force_local,
      torque_local_nm=torque_local,
      force_world_n=force_world,
      torque_world_nm=torque_world,
    )


def create_wrist_wrench_group(file: Any, string_dtype: Any) -> Any:
  """Create common metadata shared by standard and native-rate recorders."""
  group = file.create_group("wrist_wrench")
  group.attrs["schema_version"] = WRIST_WRENCH_SCHEMA_VERSION
  group.attrs["timestamp_reference"] = "/state/timestamp"
  group.attrs["sample_clock"] = "same simulation state epoch as state/timestamp"
  group.attrs["force_unit"] = "N"
  group.attrs["torque_unit"] = "N_m"
  group.attrs["local_frame"] = "respective wrist_ft_site frame"
  group.attrs["world_frame"] = "MuJoCo world frame"
  group.attrs["torque_reference"] = "respective wrist_ft_site origin"
  group.attrs["compensation"] = "raw MuJoCo sensor; no gravity or bias subtraction"
  group.create_dataset("side_names", data=SIDES, dtype=string_dtype)
  group.create_dataset("site_names", data=WRIST_FT_SITE_NAMES, dtype=string_dtype)
  group.create_dataset(
    "force_sensor_names", data=WRIST_FORCE_SENSOR_NAMES, dtype=string_dtype
  )
  group.create_dataset(
    "torque_sensor_names", data=WRIST_TORQUE_SENSOR_NAMES, dtype=string_dtype
  )
  return group


def wait_until_object_stable(
  simulation: ArmHandSimulation,
  object_name: str,
  *,
  observer: Callable[[ArmHandSimulation, str], None] | None = None,
  phase: str = "terminal_settle",
  linear_speed_threshold: float = TERMINAL_LINEAR_SPEED_THRESHOLD,
  angular_speed_threshold: float = TERMINAL_ANGULAR_SPEED_THRESHOLD,
  stable_duration: float = TERMINAL_STABLE_DURATION,
  maximum_duration: float = 5.0,
) -> TerminalStability:
  """Advance physics until low object velocity persists for a stable window.

  The stable window is a physical exit condition rather than a fixed delay:
  any threshold violation resets the consecutive-step counter.  The maximum
  duration is only a safety bound for objects that never settle.
  """

  if not np.isfinite((linear_speed_threshold, angular_speed_threshold)).all():
    raise ValueError("terminal velocity thresholds must be finite")
  if linear_speed_threshold <= 0.0 or angular_speed_threshold <= 0.0:
    raise ValueError("terminal velocity thresholds must be positive")
  if not np.isfinite((stable_duration, maximum_duration)).all():
    raise ValueError("terminal stability durations must be finite")
  if stable_duration <= 0.0:
    raise ValueError("stable_duration must be positive")
  if maximum_duration < stable_duration:
    raise ValueError("maximum_duration must be at least stable_duration")

  timestep = float(simulation.timestep)
  if not np.isfinite(timestep) or timestep <= 0.0:
    raise ValueError("simulation timestep must be finite and positive")
  required_stable_steps = max(1, int(np.ceil(stable_duration / timestep)))
  maximum_steps = max(1, int(np.ceil(maximum_duration / timestep)))
  started_at = float(simulation.data.time)
  stable_steps = 0
  linear_speed = float("inf")
  angular_speed = float("inf")

  for step_count in range(1, maximum_steps + 1):
    simulation.step()
    if observer is not None:
      observer(simulation, phase)
    twist = simulation.object_twist(object_name)
    linear_speed = float(np.linalg.norm(twist[:3]))
    angular_speed = float(np.linalg.norm(twist[3:]))
    below_threshold = (
      linear_speed < linear_speed_threshold and angular_speed < angular_speed_threshold
    )
    stable_steps = stable_steps + 1 if below_threshold else 0
    if stable_steps >= required_stable_steps:
      return TerminalStability(
        elapsed_seconds=float(simulation.data.time) - started_at,
        stable_seconds=stable_steps * timestep,
        linear_speed=linear_speed,
        angular_speed=angular_speed,
        steps=step_count,
      )

  raise RuntimeError(
    f"{object_name} did not remain below terminal velocity thresholds "
    f"({linear_speed_threshold:.3f} m/s, {angular_speed_threshold:.3f} rad/s) "
    f"for {stable_duration:.3f} s within {maximum_duration:.3f} s; "
    f"last speeds=({linear_speed:.3f} m/s, {angular_speed:.3f} rad/s)"
  )


class EpisodeRecorder:
  """Append physics-derived streams on independent state and camera clocks."""

  def __init__(
    self,
    path: str | Path,
    simulation: ArmHandSimulation,
    config: WorkcellConfig,
    *,
    tactile_provider: TactileProvider | None = None,
    renderer: WorkcellRenderer | None = None,
    metadata: dict[str, Any] | None = None,
    capture_taskspace: bool = False,
    buffer_rows: int | None = None,
  ) -> None:
    h5py = _import_h5py()
    self.h5py = h5py
    self.path = Path(path).expanduser().resolve()
    self.path.parent.mkdir(parents=True, exist_ok=True)
    self.partial_path = self.path.with_suffix(self.path.suffix + ".partial")
    if self.partial_path.exists():
      self.partial_path.unlink()
    self.sim = simulation
    self.config = config
    # Keep the historical poker environment switch while allowing task-local
    # recorders to select the same shared, lossless append adapter explicitly.
    if buffer_rows is None:
      buffer_rows = (
        int(os.environ.get("KAIHAND_POKER_HDF5_BUFFER_ROWS", "0"))
        if simulation.scene == "poker-draw"
        else 0
      )
    if type(buffer_rows) is not int or not 0 <= buffer_rows <= 256:
      raise ValueError("buffer_rows must be an integer in 0..256")
    self.taskspace_capture = None
    if capture_taskspace:
      from .taskspace_recording import TaskspaceCapture

      self.taskspace_capture = TaskspaceCapture(simulation)
    if tactile_provider is not None:
      self.tactile_provider = tactile_provider
    elif config.tactile_provider == GenesisProbeTactileProvider.source:
      self.tactile_provider = GenesisProbeTactileProvider(
        simulation.model, simulation.genesis_probe_layout
      )
    else:
      self.tactile_provider = SolverContactTactileProvider(simulation.model)
    if not self.tactile_provider.available:
      raise RuntimeError(
        f"tactile provider {self.tactile_provider.source!r} unavailable"
      )
    # Keep the historical Genesis/solver streams intact.  The new poker-only
    # stream is an explicitly labelled spatial estimate of physical contact
    # forces; visual hiding of the left hand must not remove its recorded data.
    self.contact_force_provider = None
    if simulation.scene == "poker-draw":
      from .contact_tactile import SolverDistributedTactileProvider

      self.contact_force_provider = SolverDistributedTactileProvider(
        simulation.model,
        simulation.genesis_probe_layout,
        target_geom_names=("card_core_geom",),
      )
    self.renderer = renderer
    self._owns_renderer = False
    if renderer is None and config.cameras:
      self.renderer = WorkcellRenderer(
        simulation.model, config.cameras, shadows=not capture_taskspace
      )
      self._owns_renderer = True
    self._file = h5py.File(self.partial_path, "w")
    if buffer_rows:
      from .buffered_h5 import BufferedH5File

      self._file = BufferedH5File(self._file, rows=buffer_rows)
      self._file.attrs["hdf5_buffer_rows"] = buffer_rows
      self._file.attrs["hdf5_buffer_source_sha256"] = _sha256(
        Path(__file__).with_name("buffered_h5.py")
      )
      self._file.attrs["hdf5_buffer_policy"] = (
        "copied non-camera appends; camera immediate; flush at progress/trace/close; "
        "SIGKILL/power loss may lose pending tail; partials never success"
      )
    self._closed = False
    self._finalized = False
    self._next_state_time = float(simulation.data.time)
    self._next_camera_time = float(simulation.data.time)
    self._last_observed_time = float(simulation.data.time)
    self._outcome: dict[str, Any] = {}
    self._state_samples = 0
    self._camera_samples = {camera.name: 0 for camera in config.cameras}
    self._last_state_time: float | None = None
    self._last_camera_time: float | None = None
    self._recording_started = time.monotonic()
    self._state_write_seconds = 0.0
    self._camera_write_seconds = 0.0
    self._next_progress_time = float(simulation.data.time) + 5.0
    self._initialize(metadata or {})

  def __enter__(self) -> EpisodeRecorder:
    return self

  def __exit__(self, exception_type: object, *_: object) -> None:
    self.close(finalize=exception_type is None)

  def _initialize(self, metadata: dict[str, Any]) -> None:
    file = self._file
    active_objects = tuple(self.sim.object_names)
    model_objects = tuple(self.sim.model_object_names)
    if not active_objects or not set(active_objects).issubset(model_objects):
      raise RuntimeError(
        "simulation active objects must be a non-empty subset of model objects"
      )
    model_layout = (
      TASK_ISOLATED_MODEL_LAYOUT
      if active_objects == model_objects
      else LEGACY_COMBINED_MODEL_LAYOUT
    )
    recording_metadata = dict(metadata)
    # These fields describe the simulation that is actually being recorded;
    # callers cannot accidentally publish contradictory task/model metadata.
    recording_metadata.update(
      {
        "scene": self.sim.scene,
        "model_id": _model_id(self.sim.model),
        "model_layout": model_layout,
        "active_objects": list(active_objects),
      }
    )
    file.attrs["schema_version"] = SCHEMA_VERSION
    file.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()
    file.attrs["model_path"] = str(self.sim.model_path)
    # Keep the historical root-file digest for existing converters.  The
    # dependency fingerprint additionally invalidates resume when a shared
    # recursively included MJCF source changes.
    file.attrs["model_sha256"] = _sha256(self.sim.model_path)
    file.attrs["model_fingerprint"] = model_fingerprint(self.sim.model_path)
    file.attrs["physics_hz"] = self.config.physics_hz
    file.attrs["control_hz"] = self.config.control_hz
    file.attrs["camera_hz"] = self.config.camera_hz
    file.attrs["tactile_source"] = self.tactile_provider.source
    file.attrs["runtime_contract_json"] = json.dumps(
      simulation_runtime_contract(
        self.sim,
        tactile_provider=self.tactile_provider,
        contact_force_provider=self.contact_force_provider,
      ),
      ensure_ascii=False,
      sort_keys=True,
      allow_nan=False,
    )
    backend_info = getattr(self.renderer, "backend_info", None)
    if backend_info:
      file.attrs["render_backend_json"] = json.dumps(backend_info, sort_keys=True)
    if self.taskspace_capture is not None:
      file.attrs["render_shadows"] = False
      file.attrs["taskspace_capture_source_sha256"] = _sha256(
        Path(__file__).with_name("taskspace_recording.py")
      )
    if self.contact_force_provider is not None:
      file.attrs["contact_force_source"] = self.contact_force_provider.source
    file.attrs["metadata_json"] = json.dumps(
      recording_metadata, ensure_ascii=False, sort_keys=True
    )

    string_dtype = self.h5py.string_dtype(encoding="utf-8")
    model_group = file.create_group("model")
    model_group.create_dataset(
      "body_names",
      data=tuple(
        self.sim.model.body(index).name or "" for index in range(self.sim.model.nbody)
      ),
      dtype=string_dtype,
    )
    model_group.create_dataset(
      "geom_names",
      data=tuple(
        self.sim.model.geom(index).name or "" for index in range(self.sim.model.ngeom)
      ),
      dtype=string_dtype,
    )
    model_group.attrs["segmentation_channels"] = "mjtObj_type, object_id"
    state = file.create_group("state")
    state.create_dataset("joint_names", data=self.sim.joint_names, dtype=string_dtype)
    state.create_dataset(
      "full_qpos_names",
      data=tuple(_qpos_names(self.sim.model)),
      dtype=string_dtype,
    )
    state.create_dataset(
      "full_qvel_names",
      data=tuple(_qvel_names(self.sim.model)),
      dtype=string_dtype,
    )
    _stream(state, "timestamp", (), np.float64)
    _stream(state, "qpos", (self.sim.model.nq,), np.float64)
    _stream(state, "qvel", (self.sim.model.nv,), np.float64)
    _stream(state, "robot_joint_position", (len(self.sim.joint_names),), np.float64)
    _stream(state, "robot_joint_velocity", (len(self.sim.joint_names),), np.float64)
    _stream(state, "robot_joint_effort", (len(self.sim.joint_names),), np.float64)

    self._initialize_wrist_wrench(string_dtype)

    commands = file.create_group("commands")
    arm_target, hand_target, hand_names = self.sim.command_state()
    commands.create_dataset("arm_joint_names", data=_arm_names(), dtype=string_dtype)
    commands.create_dataset("hand_joint_names", data=hand_names, dtype=string_dtype)
    _stream(commands, "arm_joint_target", arm_target.shape, np.float64)
    _stream(commands, "hand_joint_target", hand_target.shape, np.float64)
    _stream(commands, "phase", (), string_dtype)
    if self.taskspace_capture is not None:
      commands.create_dataset(
        "actuator_names",
        data=[self.sim.model.actuator(i).name for i in range(self.sim.model.nu)],
        dtype=string_dtype,
      )
      _stream(commands, "actuator_control", (self.sim.model.nu,), np.float64)
      _stream(commands, "drive_budget_active", (), np.bool_)
      for name in ("drive_requested_fx_n", "drive_actual_fx_n", "drive_limit_n"):
        _stream(commands, name, (), np.float64)

    objects = file.create_group("objects")
    for object_name in active_objects:
      group = objects.create_group(object_name)
      _stream(group, "pose_wxyz", (7,), np.float64)
      _stream(group, "twist_linear_angular", (6,), np.float64)

    self._tactile_group_name = getattr(
      self.tactile_provider, "dataset_group", "tactile_proxy"
    )
    tactile = file.create_group(self._tactile_group_name)
    tactile.attrs["source"] = self.tactile_provider.source
    tactile.attrs["force_unit"] = getattr(
      self.tactile_provider, "force_unit", "unspecified"
    )
    tactile.attrs["is_genesis_probe_truth"] = bool(
      getattr(self.tactile_provider, "is_genesis_probe_truth", False)
    )
    tactile.attrs["provider_status"] = getattr(
      self.tactile_provider, "status", "unspecified"
    )
    tactile.create_dataset(
      "link_names", data=self.tactile_provider.link_names, dtype=string_dtype
    )
    link_count = len(self.tactile_provider.link_names)
    _stream(tactile, "contact", (link_count,), np.bool_)
    _stream(tactile, "normal_force", (link_count,), np.float64)
    _stream(tactile, "contact_count", (link_count,), np.int32)
    for name in ("force_world", "torque_world", "force_local", "centroid_world"):
      _stream(tactile, name, (link_count, 3), np.float64)
    if isinstance(self.tactile_provider, GenesisProbeTactileProvider):
      layout = self.tactile_provider.layout
      tactile.attrs["contact_semantics"] = (
        "link contact_count > bool_count_threshold; derived from debounced "
        "probe_contact"
      )
      tactile.attrs["probe_contact_semantics"] = (
        "threshold hysteresis with simulation-time release debounce"
      )
      tactile.attrs["probe_contact_instantaneous_semantics"] = (
        "probe_depth >= contact_threshold_m and probe_radius > 0"
      )
      tactile.attrs["probe_depth_semantics"] = (
        "instantaneous nonnegative compression including target geom margin"
      )
      tactile.attrs["centroid_world_missing_value"] = "NaN"
      tactile.attrs["centroid_world_validity"] = "all three coordinates finite"
      tactile.attrs["contact_threshold_m"] = self.tactile_provider.contact_threshold_m
      tactile.attrs["release_threshold_m"] = self.tactile_provider.release_threshold_m
      tactile.attrs["bool_count_threshold"] = self.tactile_provider.bool_count_threshold
      tactile.attrs["release_debounce_seconds"] = (
        self.tactile_provider.release_debounce_steps * self.sim.timestep
      )
      tactile.create_dataset(
        "probe_link_names", data=layout.body_names, dtype=string_dtype
      )
      tactile.create_dataset("probe_local_pos", data=layout.local_pos)
      tactile.create_dataset("probe_local_normal", data=layout.local_normal)
      tactile.create_dataset("probe_radius", data=layout.probe_radius)
      _stream(tactile, "probe_contact", (layout.count,), np.bool_)
      _stream(tactile, "probe_contact_instantaneous", (layout.count,), np.bool_)
      _stream(tactile, "probe_depth", (layout.count,), np.float64)
      _stream(tactile, "probe_target_geom_id", (layout.count,), np.int32)

    if self.contact_force_provider is not None:
      provider = self.contact_force_provider
      force_group = file.create_group("tactile_contact_force")
      force_group.attrs["force_unit"] = "N"
      force_group.attrs["taxel_unit"] = "N_per_taxel"
      force_group.attrs["is_spatial_estimate"] = True
      force_group.attrs["timestamp_reference"] = "/state/timestamp"
      if self.taskspace_capture is not None:
        _stream(force_group, "timestamp", (), np.float64)
        force_group.attrs["timestamp_reference"] = "/tactile_contact_force/timestamp"
        force_group.attrs["clock_semantics"] = (
          "solver evaluation time before integration; state/timestamp is post-integration"
        )
      force_group.attrs["metadata_json"] = json.dumps(
        provider.metadata(), sort_keys=True
      )
      force_group.create_dataset(
        "link_names", data=provider.link_names, dtype=string_dtype
      )
      for name in (
        "taxel_positions_local_m",
        "normal_axis_local",
        "tangent_basis_local",
      ):
        force_group.create_dataset(name, data=getattr(provider, name))
      for name, tail_shape in _CONTACT_FORCE_SHAPES.items():
        dtype = np.int32 if name == "contact_count" else np.float64
        _stream(force_group, name, (len(provider.link_names), *tail_shape), dtype)

    contacts = file.create_group("contacts")
    _stream(contacts, "frame_start", (), np.int64)
    _stream(contacts, "frame_count", (), np.int32)
    events = contacts.create_group("events")
    events.attrs["wrench_force_unit"] = "N"
    events.attrs["wrench_torque_unit"] = "N_m"
    events.attrs["wrench_action"] = "on geom2; negate for action on geom1"
    events.attrs["contact_wrench_components"] = "Fn,Ft1,Ft2,Tn,Tt1,Tt2"
    events.attrs["wrench_torque_reference"] = "contact point, not world origin"
    _stream(events, "state_index", (), np.int64)
    _stream(events, "geom1_id", (), np.int32)
    _stream(events, "geom2_id", (), np.int32)
    _stream(events, "body1_id", (), np.int32)
    _stream(events, "body2_id", (), np.int32)
    _stream(events, "distance", (), np.float64)
    _stream(events, "position_world", (3,), np.float64)
    _stream(events, "frame_world", (3, 3), np.float64)
    _stream(events, "wrench_contact_on_geom2", (6,), np.float64)
    _stream(events, "wrench_world_on_geom2", (6,), np.float64)

    cameras = file.create_group("cameras")
    cameras.attrs["configured_names_json"] = json.dumps(
      [camera.name for camera in self.config.cameras],
      ensure_ascii=False,
    )
    for camera in self.config.cameras:
      group = cameras.create_group(camera.name)
      group.attrs["width"] = camera.width
      group.attrs["height"] = camera.height
      group.attrs["segmentation_channels"] = "mjtObj_type, object_id"
      _stream(group, "timestamp", (), np.float64)
      _stream(group, "state_index", (), np.int64)
      _stream(group, "world_from_camera", (4, 4), np.float64)
      if self.taskspace_capture is not None:
        _stream(group, "pose_timestamp", (), np.float64)
        _stream(group, "world_from_wrist", (2, 4, 4), np.float64)
        _stream(group, "world_from_fingertip", (2, 5, 4, 4), np.float64)
        group.attrs["taskspace_schema"] = "kaihand-native-site-se3-v1"
        group.attrs["side_names_json"] = json.dumps(["left", "right"])
        group.attrs["finger_names_json"] = json.dumps(
          ["thumb", "index", "middle", "ring", "little"]
        )
        group.attrs["wrist_sites_json"] = json.dumps(self.taskspace_capture.wrist_names)
        group.attrs["fingertip_sites_json"] = json.dumps(
          self.taskspace_capture.finger_names
        )
        group.attrs["pose_clock_semantics"] = (
          "cached FK and actual rendered RGB at pose_timestamp; timestamp is "
          "post-integration; no additional mj_forward"
        )
      if camera.rgb:
        _image_stream(group, "rgb", (camera.height, camera.width, 3), np.uint8)
      if camera.depth:
        _image_stream(group, "depth", (camera.height, camera.width), np.float32)
      if camera.segmentation:
        _image_stream(group, "segmentation", (camera.height, camera.width, 2), np.int32)

  def _initialize_wrist_wrench(self, string_dtype: Any) -> None:
    """Create the shared, state-clocked bimanual wrist F/T stream."""
    self.wrist_wrench_sensor = None
    try:
      self.wrist_wrench_sensor = WristWrenchSensor(self.sim.model)
    except KeyError:
      # Keep custom and historical non-workcell MJCF models recordable. Every
      # supported task model is separately required to contain all four shared
      # sensors by the scene contract tests.
      return
    group = create_wrist_wrench_group(self._file, string_dtype)
    _stream(group, "timestamp", (), np.float64)
    _stream(group, "origin_world_m", (2, 3), np.float64)
    _stream(group, "world_from_sensor_rotation", (2, 3, 3), np.float64)
    _stream(group, "force_local_n", (2, 3), np.float64)
    _stream(group, "torque_local_nm", (2, 3), np.float64)
    _stream(group, "force_world_n", (2, 3), np.float64)
    _stream(group, "torque_world_nm", (2, 3), np.float64)

  def _record_wrist_wrench(self, timestamp: float) -> None:
    if self.wrist_wrench_sensor is None:
      return
    sample = self.wrist_wrench_sensor.read(self.sim.data)
    group = self._file["wrist_wrench"]
    _append(group["timestamp"], timestamp)
    for name in WristWrenchSample.__dataclass_fields__:
      _append(group[name], getattr(sample, name))

  def set_outcome(self, outcome: dict[str, Any]) -> None:
    self._outcome = dict(outcome)

  def write_precontact_noise_trace(
    self, trace: dict[str, np.ndarray], *, metadata: dict[str, Any]
  ) -> None:
    """Archive the full physics-clock control audit independently of state sampling."""
    if self._closed:
      raise RuntimeError("cannot write precontact trace after recorder closes")
    if not trace:
      raise ValueError("precontact trace must contain named arrays")
    arrays = {}
    count = None
    for name, values in trace.items():
      array = np.asarray(values)
      if not name or "/" in name or array.ndim < 1:
        raise ValueError("precontact trace fields must be named time-series arrays")
      if array.dtype.kind not in "bifu" or not np.all(np.isfinite(array)):
        raise ValueError(
          f"precontact trace {name!r} must contain finite numeric values"
        )
      if count is not None and array.shape[0] != count:
        raise ValueError("precontact trace fields must have identical sample counts")
      count = array.shape[0]
      arrays[name] = array
    metadata_json = json.dumps(metadata, ensure_ascii=False, sort_keys=True)
    group = self._file.require_group("control").create_group("precontact_noise")
    group.attrs["metadata_json"] = metadata_json
    for name, array in arrays.items():
      options = {"compression": "gzip", "shuffle": True} if array.size else {}
      group.create_dataset(name, data=array, **options)
    self._file.flush()

  def record_initial(self, phase: str = "reset") -> None:
    check_physics_clock(self._last_observed_time, float(self.sim.data.time))
    self._last_observed_time = float(self.sim.data.time)
    self._record_state(phase)
    self._next_state_time += 1.0 / self.config.control_hz
    self._capture_cameras()
    self._next_camera_time += 1.0 / self.config.camera_hz

  def record_terminal(self, phase: str = "terminal_settle") -> None:
    """Synchronize exact terminal state/tactile and camera observations.

    A scheduled observer sample may already exist at the current physics time.
    In that case it is retained rather than appending a duplicate timestamp.
    """

    if self._closed:
      raise RuntimeError("cannot record terminal state after recorder is closed")
    timestamp = float(self.sim.data.time)
    check_physics_clock(self._last_observed_time, timestamp)
    self._last_observed_time = timestamp
    state_appended = False
    if self._last_state_time is None or timestamp > self._last_state_time:
      self._record_state(phase)
      self._next_state_time = timestamp + 1.0 / self.config.control_hz
      state_appended = True
    elif timestamp < self._last_state_time:
      raise RuntimeError("simulation time moved backwards before terminal record")
    else:
      # The observer already sampled this exact physics state.  Mark that
      # existing sample terminal without duplicating its timestamp.
      self._file["commands/phase"][-1] = phase

    if not self.config.cameras:
      return
    if self._last_camera_time is None or timestamp > self._last_camera_time:
      self._capture_cameras()
      self._next_camera_time = timestamp + 1.0 / self.config.camera_hz
    elif timestamp < self._last_camera_time:
      raise RuntimeError("simulation time moved backwards before terminal camera")
    elif state_appended:
      # Camera capture is ordered after scheduled state sampling in observe(),
      # but an off-rate terminal time can already have a camera and no state.
      # Its pixels represent the unchanged current physics data, so retarget
      # the existing frame to the exact state appended above.
      terminal_index = self._state_samples - 1
      for camera in self.config.cameras:
        self._file[f"cameras/{camera.name}/state_index"][-1] = terminal_index

  def observe(self, simulation: ArmHandSimulation, phase: str) -> None:
    if simulation is not self.sim:
      raise ValueError("recorder received a different simulation instance")
    timestamp = float(self.sim.data.time)
    check_physics_clock(self._last_observed_time, timestamp)
    self._last_observed_time = timestamp
    tolerance = 0.5 * self.sim.timestep
    if self.sim.data.time + tolerance >= self._next_state_time:
      self._record_state(phase)
      self._next_state_time = _advance_sample_deadline(
        self._next_state_time,
        1.0 / self.config.control_hz,
        float(self.sim.data.time),
        tolerance,
      )
    if self.config.cameras and self.sim.data.time + tolerance >= self._next_camera_time:
      self._capture_cameras()
      self._next_camera_time = _advance_sample_deadline(
        self._next_camera_time,
        1.0 / self.config.camera_hz,
        float(self.sim.data.time),
        tolerance,
      )
    if (
      self.taskspace_capture is not None
      and self.sim.data.time >= self._next_progress_time
    ):
      # Periodic flush makes interrupted captures diagnosable; they still cannot
      # be published without successful finalization and terminal validation.
      self._file.flush()
      print(
        f"recording sim={self.sim.data.time:.3f}s phase={phase} "
        f"states={self._state_samples} cameras={self._camera_samples} "
        f"wall={time.monotonic() - self._recording_started:.1f}s "
        f"state_write={self._state_write_seconds:.1f}s "
        f"camera_write={self._camera_write_seconds:.1f}s",
        flush=True,
      )
      self._next_progress_time = float(self.sim.data.time) + 5.0

  def _record_state(self, phase: str) -> None:
    started_at = time.monotonic()
    index = self._state_samples
    data = self.sim.data
    timestamp = float(data.time)
    if self._last_state_time is not None and timestamp <= self._last_state_time:
      raise RuntimeError("state timestamps must be strictly increasing")
    state = self._file["state"]
    _append(state["timestamp"], timestamp)
    _append(state["qpos"], data.qpos)
    _append(state["qvel"], data.qvel)
    joint_names, joint_position, joint_velocity = self.sim.joint_state()
    del joint_names
    joint_effort = np.array(
      [
        data.qfrc_actuator[self.sim._qvel_address[name]]
        for name in self.sim.joint_names
      ]
    )
    _append(state["robot_joint_position"], joint_position)
    _append(state["robot_joint_velocity"], joint_velocity)
    _append(state["robot_joint_effort"], joint_effort)
    self._record_wrist_wrench(timestamp)

    arm_target, hand_target, _ = self.sim.command_state()
    commands = self._file["commands"]
    _append(commands["arm_joint_target"], arm_target)
    _append(commands["hand_joint_target"], hand_target)
    _append(commands["phase"], phase)
    if self.taskspace_capture is not None:
      _append(commands["actuator_control"], data.ctrl)
      drive_limit = getattr(self.sim, "drive_limit_n", None)
      active = drive_limit is not None and hasattr(self.sim, "drive_state")
      _append(commands["drive_budget_active"], active)
      drive = self.sim.drive_state() if active else {}
      for name in ("drive_requested_fx_n", "drive_actual_fx_n"):
        _append(commands[name], drive[name] if active else 0.0)
      _append(commands["drive_limit_n"], drive_limit if active else 0.0)

    for object_name in self.sim.object_names:
      group = self._file[f"objects/{object_name}"]
      _append(group["pose_wxyz"], self.sim.object_pose(object_name))
      _append(group["twist_linear_angular"], self.sim.object_twist(object_name))

    sample = self.tactile_provider.read(data)
    tactile = self._file[self._tactile_group_name]
    for name in (
      "contact",
      "normal_force",
      "contact_count",
      "force_world",
      "torque_world",
      "force_local",
      "centroid_world",
    ):
      _append(tactile[name], getattr(sample, name))
    if self.contact_force_provider is not None:
      force_sample = self.contact_force_provider.read(data)
      force_group = self._file["tactile_contact_force"]
      if self.taskspace_capture is not None:
        _append(force_group["timestamp"], float(self.sim.observation_time))
      for name in _CONTACT_FORCE_SHAPES:
        _append(force_group[name], getattr(force_sample, name))
    if isinstance(self.tactile_provider, GenesisProbeTactileProvider):
      _append(tactile["probe_contact"], self.tactile_provider.probe_contact)
      _append(
        tactile["probe_contact_instantaneous"],
        self.tactile_provider.probe_contact_instantaneous,
      )
      _append(tactile["probe_depth"], self.tactile_provider.probe_depth)
      _append(
        tactile["probe_target_geom_id"],
        self.tactile_provider.probe_target_geom_id,
      )

    contacts = raw_contact_snapshot(self.sim.model, data)
    contact_group = self._file["contacts"]
    events = contact_group["events"]
    event_start = int(events["geom1_id"].shape[0])
    event_count = int(contacts["geom1_id"].shape[0])
    _append(contact_group["frame_start"], event_start)
    _append(contact_group["frame_count"], event_count)
    if event_count:
      _append_many(events["state_index"], np.full(event_count, index, dtype=np.int64))
      for name, values in contacts.items():
        _append_many(events[name], values)
    self._state_samples += 1
    self._last_state_time = timestamp
    self._state_write_seconds += time.monotonic() - started_at

  def _capture_cameras(self) -> None:
    if self.renderer is None:
      return
    started_at = time.monotonic()
    timestamp = float(self.sim.data.time)
    if self._last_camera_time is not None and timestamp <= self._last_camera_time:
      raise RuntimeError("camera timestamps must be strictly increasing")
    state_index = max(self._state_samples - 1, 0)
    for camera in self.config.cameras:
      group = self._file[f"cameras/{camera.name}"]
      calibration = self.renderer.calibration(self.sim.data, camera)
      if "intrinsic" not in group:
        group.create_dataset("intrinsic", data=calibration.intrinsic)
        group.attrs["fovy_degrees"] = calibration.fovy_degrees
      capture = self.renderer.capture(self.sim.data, camera)
      _append(group["timestamp"], timestamp)
      _append(group["state_index"], state_index)
      _append(group["world_from_camera"], calibration.world_from_camera)
      if self.taskspace_capture is not None:
        pose_time, wrists, fingertips = self.taskspace_capture.read()
        _append(group["pose_timestamp"], pose_time)
        _append(group["world_from_wrist"], wrists)
        _append(group["world_from_fingertip"], fingertips)
      for name, image in capture.items():
        _append(group[name], image)
      self._camera_samples[camera.name] += 1
    self._last_camera_time = timestamp
    self._camera_write_seconds += time.monotonic() - started_at

  def close(self, *, finalize: bool = True) -> None:
    if self._closed:
      return
    self._closed = True
    try:
      self._file.attrs["outcome_json"] = json.dumps(
        self._outcome, ensure_ascii=False, sort_keys=True
      )
      self._file.flush()
      if hasattr(self._file, "statistics"):
        self._file.attrs["hdf5_buffer_statistics_json"] = json.dumps(
          self._file.statistics(), sort_keys=True
        )
    finally:
      try:
        self._file.close()
      finally:
        if self._owns_renderer and self.renderer is not None:
          self.renderer.close()
    if not finalize:
      return
    os.replace(self.partial_path, self.path)
    manifest = {
      "schema_version": SCHEMA_VERSION,
      "episode": self.path.name,
      "sha256": _sha256(self.path),
      "state_samples": self._state_samples,
      "camera_samples": self._camera_samples,
      "tactile_source": self.tactile_provider.source,
      "outcome": self._outcome,
    }
    self.path.with_suffix(".json").write_text(
      json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
      encoding="utf-8",
    )
    self._finalized = True


def validate_episode(path: str | Path) -> ValidationReport:
  h5py = _import_h5py()
  errors: list[str] = []
  warnings: list[str] = []
  camera_samples: dict[str, int] = {}
  state_samples = 0
  duration = 0.0
  with h5py.File(Path(path).expanduser(), "r") as file:
    if file.attrs.get("schema_version") != SCHEMA_VERSION:
      errors.append("unsupported or missing schema_version")
    tactile_group = "tactile_genesis" if "tactile_genesis" in file else "tactile_proxy"
    required = (
      "state/timestamp",
      "state/qpos",
      "state/qvel",
      "commands/phase",
      f"{tactile_group}/normal_force",
      "contacts/frame_start",
      "contacts/frame_count",
    )
    for key in required:
      if key not in file:
        errors.append(f"missing dataset {key}")
    if errors:
      return ValidationReport(False, tuple(errors), tuple(warnings), 0, {}, 0.0)
    timestamp = np.asarray(file["state/timestamp"])
    state_samples = len(timestamp)
    if state_samples == 0:
      errors.append("episode contains no state samples")
    elif not np.all(np.diff(timestamp) > 0.0):
      errors.append("state timestamps are not strictly increasing")
    else:
      duration = float(timestamp[-1] - timestamp[0])
    for key in ("state/qpos", "state/qvel", f"{tactile_group}/normal_force"):
      dataset = file[key]
      if dataset.shape[0] != state_samples:
        errors.append(f"{key} sample count differs from state/timestamp")
      if not np.all(np.isfinite(dataset)):
        errors.append(f"{key} contains non-finite values")
    _validate_wrist_wrench(file, timestamp, errors)
    if file["contacts/frame_count"].shape[0] != state_samples:
      errors.append("contact frame count differs from state sample count")
    try:
      recording_metadata = json.loads(file.attrs.get("metadata_json", "{}"))
    except (TypeError, json.JSONDecodeError):
      recording_metadata = {}
      errors.append("metadata_json is not valid JSON")
    if not isinstance(recording_metadata, dict):
      recording_metadata = {}
      errors.append("metadata_json must contain an object")
    force_stream_required = (
      "contact_force_source" in file.attrs
      or recording_metadata.get("recording_contract") == POKER_FORCE_RECORDING_CONTRACT
    )
    if force_stream_required and "tactile_contact_force" not in file:
      errors.append("force-recording contract is missing tactile_contact_force")
    if "tactile_contact_force" in file:
      force_group = file["tactile_contact_force"]
      if force_group.attrs.get("force_unit") != "N":
        errors.append("tactile_contact_force force_unit must be N")
      force_reference = force_group.attrs.get("timestamp_reference")
      if force_reference == "/tactile_contact_force/timestamp":
        if "timestamp" not in force_group:
          errors.append("tactile_contact_force is missing its solver timestamps")
        else:
          force_times = np.asarray(force_group["timestamp"])
          if force_times.shape != timestamp.shape or not np.all(
            np.isfinite(force_times)
          ):
            errors.append("invalid solver timestamp shape or values")
          elif np.any(np.diff(force_times) < 0) or np.any(
            force_times > timestamp + 1e-9
          ):
            errors.append("solver timestamps must be causal and nondecreasing")
      elif force_reference != "/state/timestamp":
        errors.append("tactile_contact_force has an unknown timestamp reference")
      try:
        force_metadata = json.loads(force_group.attrs.get("metadata_json", "{}"))
      except (TypeError, json.JSONDecodeError):
        force_metadata = {}
      if not isinstance(force_metadata, dict):
        force_metadata = {}
      expected_force_source = file.attrs.get("contact_force_source")
      if force_metadata.get("source") != expected_force_source:
        errors.append("tactile_contact_force source differs from the file contract")
      if recording_metadata.get("scene") == "poker-draw" and (
        force_metadata.get("target_geom_names") != ["card_core_geom"]
      ):
        errors.append("poker tactile_contact_force must select only the card geometry")
      if "link_names" not in force_group:
        errors.append("tactile_contact_force is missing link_names")
      else:
        link_count = len(force_group["link_names"])
        layout_valid = True
        for name, tail in (
          ("taxel_positions_local_m", (35, 3)),
          ("normal_axis_local", (3,)),
          ("tangent_basis_local", (2, 3)),
        ):
          if name not in force_group or force_group[name].shape != (link_count, *tail):
            errors.append(f"tactile_contact_force/{name} has missing/invalid layout")
            layout_valid = False
          elif not np.all(np.isfinite(force_group[name])):
            errors.append(f"tactile_contact_force/{name} contains non-finite values")
            layout_valid = False
        if layout_valid:
          basis = np.concatenate(
            (
              np.asarray(force_group["normal_axis_local"])[:, None, :],
              np.asarray(force_group["tangent_basis_local"]),
            ),
            axis=1,
          )
          if not np.allclose(basis @ basis.transpose(0, 2, 1), np.eye(3), atol=1e-8):
            errors.append("tactile_contact_force local basis is not orthonormal")
        force_shapes_valid = True
        for name, tail_shape in _CONTACT_FORCE_SHAPES.items():
          expected_shape = (state_samples, link_count, *tail_shape)
          if name not in force_group or force_group[name].shape != expected_shape:
            errors.append(
              f"tactile_contact_force/{name} shape differs from {expected_shape}"
            )
            force_shapes_valid = False
          elif not np.all(np.isfinite(force_group[name])):
            errors.append(f"tactile_contact_force/{name} contains non-finite values")
            force_shapes_valid = False
        if force_shapes_valid:
          normal_grid = np.asarray(force_group["normal_taxel_force_n"])
          tangent_grid = np.asarray(force_group["tangent_taxel_force_n"])
          if np.any(normal_grid < -1e-12):
            errors.append("tactile_contact_force has negative normal taxel forces")
          for name in ("tangent_load_n", "tangent_taxel_load_n", "contact_count"):
            if np.any(np.asarray(force_group[name]) < 0):
              errors.append(f"tactile_contact_force/{name} has negative magnitudes")
          if not np.allclose(
            normal_grid.sum(axis=(2, 3)), force_group["normal_force_n"], atol=1e-9
          ):
            errors.append(
              "normal taxel force does not conserve per-finger normal force"
            )
          if not np.allclose(
            tangent_grid.sum(axis=(2, 3)), force_group["tangent_force_n"], atol=1e-9
          ):
            errors.append(
              "tangent taxel force does not conserve per-finger tangent force"
            )
          if not np.allclose(
            np.asarray(force_group["tangent_taxel_load_n"]).sum(axis=(2, 3)),
            force_group["tangent_load_n"],
            atol=1e-9,
          ):
            errors.append("tangent taxel load does not conserve per-finger load")
          if not np.allclose(
            np.asarray(force_group["normal_force_world_n"])
            + np.asarray(force_group["tangent_force_world_n"]),
            force_group["force_world_n"],
            atol=1e-9,
          ):
            errors.append("world contact force differs from normal plus tangent force")
          if not np.allclose(
            np.einsum(
              "tlij,tlj->tli",
              force_group["tangent_basis_world"],
              force_group["tangent_force_world_n"],
            ),
            force_group["tangent_force_n"],
            atol=1e-9,
          ):
            errors.append("tangent force differs from its sensor-basis projection")
    if "cameras" in file:
      for camera_name, group in file["cameras"].items():
        if "timestamp" not in group:
          errors.append(f"camera {camera_name} is missing timestamps")
          continue
        camera_timestamp = np.asarray(group["timestamp"], dtype=float)
        count = len(camera_timestamp)
        camera_samples[camera_name] = count
        if count and not np.all(np.diff(camera_timestamp) > 0.0):
          errors.append(f"camera {camera_name} timestamps are not increasing")
        if count == 0:
          warnings.append(f"camera {camera_name} has no samples")
        for synchronized_name in ("state_index", "world_from_camera"):
          if synchronized_name not in group:
            errors.append(f"camera {camera_name} is missing {synchronized_name}")
          elif group[synchronized_name].shape[0] != count:
            errors.append(f"camera {camera_name}/{synchronized_name} count mismatch")
        if "state_index" in group and group["state_index"].shape[0] == count:
          state_indices = np.asarray(group["state_index"], dtype=np.int64)
          indices_valid = bool(
            np.all(state_indices >= 0) and np.all(state_indices < state_samples)
          )
          if not indices_valid:
            errors.append(f"camera {camera_name} state_index is out of range")
          elif count and np.any(timestamp[state_indices] > camera_timestamp + 1.0e-12):
            errors.append(f"camera {camera_name} refers to a future state")
        for modality in ("rgb", "depth", "segmentation"):
          if modality in group and group[modality].shape[0] != count:
            errors.append(f"camera {camera_name}/{modality} count mismatch")
    source = str(file.attrs.get("tactile_source", ""))
    if source == "solver_contact_proxy_v1":
      warnings.append(
        "tactile uses the link-level solver-contact provider, not spherical probes"
      )
    elif source == GenesisProbeTactileProvider.source:
      if f"{tactile_group}/probe_depth" not in file:
        errors.append("Genesis tactile stream is missing per-probe depth")
      elif file[f"{tactile_group}/probe_depth"].shape[0] != state_samples:
        errors.append("Genesis probe-depth sample count differs from state timestamps")
      if file[tactile_group].attrs.get("provider_status") != "accepted":
        warnings.append("Genesis probe layout is generated but not visually accepted")
    _validate_terminal_state(file, state_samples, errors, warnings)
  return ValidationReport(
    valid=not errors,
    errors=tuple(errors),
    warnings=tuple(warnings),
    state_samples=state_samples,
    camera_samples=camera_samples,
    duration_seconds=duration,
  )


def _validate_wrist_wrench(
  file: Any, state_timestamp: np.ndarray, errors: list[str]
) -> None:
  """Validate the additive wrist-wrench extension when it is present."""
  if "wrist_wrench" not in file:
    return  # Historical v1 episodes remain readable and valid.
  group = file["wrist_wrench"]
  if group.attrs.get("schema_version") != WRIST_WRENCH_SCHEMA_VERSION:
    errors.append("wrist_wrench has an unsupported or missing schema_version")
  if group.attrs.get("timestamp_reference") != "/state/timestamp":
    errors.append("wrist_wrench must reference /state/timestamp")
  for attribute, expected in (("force_unit", "N"), ("torque_unit", "N_m")):
    if group.attrs.get(attribute) != expected:
      errors.append(f"wrist_wrench {attribute} must be {expected}")
  for name, expected in (
    ("side_names", SIDES),
    ("site_names", WRIST_FT_SITE_NAMES),
    ("force_sensor_names", WRIST_FORCE_SENSOR_NAMES),
    ("torque_sensor_names", WRIST_TORQUE_SENSOR_NAMES),
  ):
    if name not in group:
      errors.append(f"wrist_wrench is missing {name}")
    else:
      values = tuple(
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in group[name][:]
      )
      if values != expected:
        errors.append(f"wrist_wrench/{name} has an invalid order or value")
  count = len(state_timestamp)
  expected_shapes = {
    "timestamp": (count,),
    "origin_world_m": (count, 2, 3),
    "world_from_sensor_rotation": (count, 2, 3, 3),
    "force_local_n": (count, 2, 3),
    "torque_local_nm": (count, 2, 3),
    "force_world_n": (count, 2, 3),
    "torque_world_nm": (count, 2, 3),
  }
  valid = True
  for name, shape in expected_shapes.items():
    if name not in group or group[name].shape != shape:
      errors.append(f"wrist_wrench/{name} shape differs from {shape}")
      valid = False
    elif not np.all(np.isfinite(group[name])):
      errors.append(f"wrist_wrench/{name} contains non-finite values")
      valid = False
  if not valid:
    return
  if not np.array_equal(np.asarray(group["timestamp"]), state_timestamp):
    errors.append("wrist_wrench timestamps differ from state timestamps")
  rotations = np.asarray(group["world_from_sensor_rotation"])
  identity = np.einsum("tsji,tsjk->tsik", rotations, rotations)
  if not np.allclose(identity, np.eye(3), atol=1e-9):
    errors.append("wrist_wrench sensor rotations are not orthonormal")
  if np.any(np.linalg.det(rotations) <= 0.0):
    errors.append("wrist_wrench sensor rotations are not proper rotations")
  for quantity in ("force", "torque"):
    unit = "n" if quantity == "force" else "nm"
    expected_world = np.einsum(
      "tsij,tsj->tsi", rotations, group[f"{quantity}_local_{unit}"]
    )
    if not np.allclose(
      expected_world,
      group[f"{quantity}_world_{unit}"],
      atol=1e-10,
    ):
      errors.append(f"wrist_wrench {quantity} world/local transform is inconsistent")


def _validate_terminal_state(
  file: Any,
  state_samples: int,
  errors: list[str],
  warnings: list[str],
) -> None:
  """Validate the strict terminal contract while retaining v1 compatibility."""

  if state_samples == 0:
    return

  raw_outcome = file.attrs.get("outcome_json", "{}")
  try:
    outcome = json.loads(raw_outcome)
  except (TypeError, json.JSONDecodeError):
    errors.append("outcome_json is not valid JSON")
    return
  if not isinstance(outcome, dict):
    errors.append("outcome_json must contain an object")
    return
  object_name = str(outcome.get("object_name", "cylinder"))
  if object_name not in OBJECT_NAMES:
    errors.append(f"terminal outcome has unknown object_name {object_name!r}")
    return
  twist_key = f"objects/{object_name}/twist_linear_angular"
  if twist_key not in file:
    return
  twist_dataset = file[twist_key]
  if twist_dataset.shape[0] != state_samples:
    errors.append(f"{object_name} twist sample count differs from state timestamps")
    return
  final_twist = np.asarray(twist_dataset[-1], dtype=float)
  if final_twist.shape != (6,) or not np.all(np.isfinite(final_twist)):
    errors.append(f"terminal {object_name} twist is missing or non-finite")
    return
  linear_speed = float(np.linalg.norm(final_twist[:3]))
  angular_speed = float(np.linalg.norm(final_twist[3:]))

  if "terminal_stability" not in outcome:
    if (
      linear_speed >= TERMINAL_LINEAR_SPEED_THRESHOLD
      or angular_speed >= TERMINAL_ANGULAR_SPEED_THRESHOLD
    ):
      warnings.append(
        f"legacy episode ends before {object_name} is stable: "
        f"linear_speed={linear_speed:.3f} m/s, "
        f"angular_speed={angular_speed:.3f} rad/s"
      )
    return

  terminal_stability = outcome["terminal_stability"]
  if not isinstance(terminal_stability, dict):
    errors.append("terminal_stability outcome must contain an object")
  else:
    _validate_terminal_stability_metadata(
      terminal_stability,
      linear_speed,
      angular_speed,
      errors,
    )
  if "commands/phase" not in file or file["commands/phase"].shape[0] == 0:
    errors.append("terminal episode is missing its final command phase")
  else:
    final_phase = file["commands/phase"][-1]
    if isinstance(final_phase, bytes):
      final_phase = final_phase.decode("utf-8")
    if str(final_phase) != "terminal_settle":
      errors.append("terminal episode does not end in terminal_settle phase")
  if (
    linear_speed >= TERMINAL_LINEAR_SPEED_THRESHOLD
    or angular_speed >= TERMINAL_ANGULAR_SPEED_THRESHOLD
  ):
    errors.append(
      f"terminal {object_name} velocity exceeds stability thresholds: "
      f"linear_speed={linear_speed:.3f} m/s, "
      f"angular_speed={angular_speed:.3f} rad/s"
    )

  _validate_terminal_velocity_window(
    file,
    twist_dataset,
    errors,
    object_name=object_name,
  )
  _validate_terminal_camera_sync(file, state_samples, errors)

  _validate_outcome_vector(
    outcome,
    "final_object_twist",
    final_twist,
    errors,
  )
  pose_key = f"objects/{object_name}/pose_wxyz"
  if pose_key not in file:
    errors.append(f"terminal episode is missing {object_name} pose samples")
    return
  pose_dataset = file[pose_key]
  if pose_dataset.shape[0] != state_samples:
    errors.append(f"{object_name} pose sample count differs from state timestamps")
    return
  _validate_outcome_vector(
    outcome,
    "final_object_pose",
    np.asarray(pose_dataset[-1], dtype=float),
    errors,
  )


def _validate_terminal_stability_metadata(
  stability: dict[str, Any],
  recorded_linear_speed: float,
  recorded_angular_speed: float,
  errors: list[str],
) -> None:
  values = {
    name: _finite_terminal_number(stability, name, errors)
    for name in (
      "elapsed_seconds",
      "stable_seconds",
      "linear_speed",
      "angular_speed",
      "steps",
    )
  }
  elapsed_seconds = values["elapsed_seconds"]
  stable_seconds = values["stable_seconds"]
  steps = values["steps"]
  if (
    stable_seconds is not None
    and stable_seconds + _TERMINAL_VALUE_TOLERANCE < TERMINAL_STABLE_DURATION
  ):
    errors.append(
      "terminal_stability stable_seconds must be at least "
      f"{TERMINAL_STABLE_DURATION:.3f}"
    )
  if (
    elapsed_seconds is not None
    and stable_seconds is not None
    and elapsed_seconds + _TERMINAL_VALUE_TOLERANCE < stable_seconds
  ):
    errors.append("terminal_stability elapsed_seconds must be at least stable_seconds")
  if steps is not None and (steps <= 0.0 or not steps.is_integer()):
    errors.append("terminal_stability steps must be a positive integer")

  _validate_terminal_reported_speed(
    "linear_speed",
    values["linear_speed"],
    recorded_linear_speed,
    TERMINAL_LINEAR_SPEED_THRESHOLD,
    errors,
  )
  _validate_terminal_reported_speed(
    "angular_speed",
    values["angular_speed"],
    recorded_angular_speed,
    TERMINAL_ANGULAR_SPEED_THRESHOLD,
    errors,
  )


def _finite_terminal_number(
  stability: dict[str, Any],
  name: str,
  errors: list[str],
) -> float | None:
  value = stability.get(name)
  if isinstance(value, bool):
    errors.append(f"terminal_stability {name} is missing or non-finite")
    return None
  try:
    number = float(value)
  except (TypeError, ValueError):
    errors.append(f"terminal_stability {name} is missing or non-finite")
    return None
  if not np.isfinite(number):
    errors.append(f"terminal_stability {name} is missing or non-finite")
    return None
  return number


def _validate_terminal_reported_speed(
  name: str,
  reported: float | None,
  recorded: float,
  threshold: float,
  errors: list[str],
) -> None:
  if reported is None:
    return
  if reported < 0.0 or reported >= threshold:
    errors.append(f"terminal_stability {name} is outside the stability threshold")
  if not np.isclose(
    reported,
    recorded,
    rtol=_TERMINAL_VALUE_TOLERANCE,
    atol=_TERMINAL_VALUE_TOLERANCE,
  ):
    errors.append(f"terminal_stability {name} is not synchronized with the final state")


def _validate_terminal_velocity_window(
  file: Any,
  twist_dataset: Any,
  errors: list[str],
  *,
  object_name: str = "cylinder",
) -> None:
  timestamps = np.asarray(file["state/timestamp"], dtype=float)
  if timestamps.size == 0 or not np.isfinite(timestamps[-1]):
    return
  terminal_time = float(timestamps[-1])
  try:
    physics_hz = float(file.attrs.get("physics_hz", float("nan")))
  except (TypeError, ValueError):
    physics_hz = float("nan")
  # A stability run with N accepted physics steps spans only (N - 1)
  # timestamp intervals.  Keep the nominal t - 0.1 boundary outside the
  # sampled window even when accumulated simulation times put that sample a
  # few ulps above the arithmetically computed boundary.
  boundary_guard = (
    0.5 / physics_hz
    if np.isfinite(physics_hz) and physics_hz > 0.0
    else _TERMINAL_VALUE_TOLERANCE
  )
  window_start = terminal_time - TERMINAL_STABLE_DURATION + boundary_guard
  window_indices = np.flatnonzero(timestamps > window_start)
  if window_indices.size < _MIN_TERMINAL_WINDOW_SAMPLES:
    errors.append(
      "terminal stability window contains fewer than "
      f"{_MIN_TERMINAL_WINDOW_SAMPLES} state samples"
    )
    return
  window_twist = np.asarray(twist_dataset[window_indices], dtype=float)
  if window_twist.shape != (window_indices.size, 6) or not np.all(
    np.isfinite(window_twist)
  ):
    errors.append(f"terminal stability window contains invalid {object_name} twists")
    return
  linear_speeds = np.linalg.norm(window_twist[:, :3], axis=1)
  angular_speeds = np.linalg.norm(window_twist[:, 3:], axis=1)
  if np.any(linear_speeds >= TERMINAL_LINEAR_SPEED_THRESHOLD) or np.any(
    angular_speeds >= TERMINAL_ANGULAR_SPEED_THRESHOLD
  ):
    errors.append(
      f"terminal stability window contains {object_name} velocity above thresholds"
    )


def _validate_terminal_camera_sync(
  file: Any,
  state_samples: int,
  errors: list[str],
) -> None:
  if "cameras" not in file or state_samples == 0:
    return
  terminal_timestamp = float(file["state/timestamp"][-1])
  terminal_state_index = state_samples - 1
  cameras = file["cameras"]
  configured_names = _configured_camera_names(cameras, errors)
  if configured_names is None:
    return
  for camera_name in configured_names:
    if camera_name not in cameras:
      errors.append(f"terminal camera {camera_name} group is missing")
      continue
    group = cameras[camera_name]
    if "timestamp" not in group or "state_index" not in group:
      continue
    if group["timestamp"].shape[0] == 0 or group["state_index"].shape[0] == 0:
      errors.append(f"terminal camera {camera_name} has no synchronized sample")
      continue
    camera_timestamp = float(group["timestamp"][-1])
    if not np.isclose(
      camera_timestamp,
      terminal_timestamp,
      rtol=0.0,
      atol=1.0e-12,
    ):
      errors.append(
        f"terminal camera {camera_name} timestamp is not synchronized "
        "with the final state"
      )
    if int(group["state_index"][-1]) != terminal_state_index:
      errors.append(
        f"terminal camera {camera_name} state_index does not refer to the final state"
      )


def _configured_camera_names(
  cameras: Any,
  errors: list[str],
) -> tuple[str, ...] | None:
  raw_names = cameras.attrs.get("configured_names_json")
  if raw_names is None:
    # Older v1 episodes did not persist an independent camera manifest.  Keep
    # validating their extant groups without rejecting them for that omission.
    return tuple(str(name) for name in cameras.keys())
  try:
    names = json.loads(raw_names)
  except (TypeError, json.JSONDecodeError):
    errors.append("configured camera names are not valid JSON")
    return None
  if (
    not isinstance(names, list)
    or any(not isinstance(name, str) or not name for name in names)
    or len(names) != len(set(names))
  ):
    errors.append("configured camera names must be a unique string list")
    return None
  return tuple(names)


def _validate_outcome_vector(
  outcome: dict[str, Any],
  name: str,
  recorded: np.ndarray,
  errors: list[str],
) -> None:
  try:
    value = np.asarray(outcome.get(name, ()), dtype=float)
  except (TypeError, ValueError):
    errors.append(f"outcome {name} is missing, non-finite or has the wrong shape")
    return
  if value.shape != recorded.shape or not np.all(np.isfinite(value)):
    errors.append(f"outcome {name} is missing, non-finite or has the wrong shape")
  elif not np.array_equal(value, recorded):
    errors.append(f"outcome {name} is not synchronized with the final state")


def _stream(group: Any, name: str, shape: tuple[int, ...], dtype: Any) -> Any:
  return group.create_dataset(
    name,
    shape=(0, *shape),
    maxshape=(None, *shape),
    chunks=(256, *shape),
    dtype=dtype,
  )


def _image_stream(group: Any, name: str, shape: tuple[int, ...], dtype: Any) -> Any:
  return group.create_dataset(
    name,
    shape=(0, *shape),
    maxshape=(None, *shape),
    chunks=(1, *shape),
    compression="gzip",
    compression_opts=1,
    shuffle=True,
    dtype=dtype,
  )


def _append(dataset: Any, value: Any) -> None:
  index = dataset.shape[0]
  dataset.resize(index + 1, axis=0)
  dataset[index] = value


def _append_many(dataset: Any, values: np.ndarray) -> None:
  count = int(values.shape[0])
  if not count:
    return
  start = dataset.shape[0]
  dataset.resize(start + count, axis=0)
  dataset[start:] = values


def _advance_sample_deadline(
  deadline: float,
  period: float,
  current_time: float,
  tolerance: float,
) -> float:
  """Skip missed deadlines rather than fabricating duplicate catch-up samples."""

  intervals = max(
    1,
    int(np.floor((current_time + tolerance - deadline) / period)) + 1,
  )
  return deadline + intervals * period


def _model_id(model: Any) -> str:
  """Return the MJCF ``<mujoco model=...>`` name stored by MuJoCo."""

  names = getattr(model, "names", b"")
  if isinstance(names, bytes):
    identifier = names.split(b"\0", maxsplit=1)[0].decode("utf-8", errors="strict")
    if identifier:
      return identifier
  raise RuntimeError("compiled MuJoCo model has no model identifier")


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def _import_h5py() -> Any:
  try:
    import h5py
  except ImportError as error:
    raise RuntimeError("HDF5 recording requires h5py; run `pixi install`") from error
  return h5py


def _arm_names() -> tuple[str, ...]:
  return tuple(
    f"{side}_arm_joint{joint}" for side in ("left", "right") for joint in range(1, 8)
  )


def _qpos_names(model: Any) -> list[str]:
  names = [""] * model.nq
  for joint_id in range(model.njnt):
    joint_name = model.joint(joint_id).name or f"joint_{joint_id}"
    address = int(model.jnt_qposadr[joint_id])
    width = (
      int(model.jnt_qposadr[joint_id + 1] - address)
      if joint_id + 1 < model.njnt
      else model.nq - address
    )
    labels = (
      ("x", "y", "z", "qw", "qx", "qy", "qz")
      if width == 7
      else tuple(str(i) for i in range(width))
    )
    for offset, label in enumerate(labels):
      names[address + offset] = joint_name if width == 1 else f"{joint_name}/{label}"
  return names


def _qvel_names(model: Any) -> list[str]:
  names = [""] * model.nv
  for joint_id in range(model.njnt):
    joint_name = model.joint(joint_id).name or f"joint_{joint_id}"
    address = int(model.jnt_dofadr[joint_id])
    width = (
      int(model.jnt_dofadr[joint_id + 1] - address)
      if joint_id + 1 < model.njnt
      else model.nv - address
    )
    labels = (
      ("vx", "vy", "vz", "wx", "wy", "wz")
      if width == 6
      else tuple(str(i) for i in range(width))
    )
    for offset, label in enumerate(labels):
      names[address + offset] = joint_name if width == 1 else f"{joint_name}/{label}"
  return names
