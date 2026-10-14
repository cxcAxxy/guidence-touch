"""USB-only adapters for the unchanged raw taskspace episode recorder."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from kaihand_tactile_env.shared.config import (
  SHARED_CAMERA_NAMES,
  TRAINING_CAMERA_NAMES,
  CameraConfig,
  WorkcellConfig,
  default_model_path,
  model_fingerprint,
)
from kaihand_tactile_env.shared.contact_tactile import SolverDistributedTactileProvider
from kaihand_tactile_env.shared.recording import (
  TERMINAL_ANGULAR_SPEED_THRESHOLD,
  TERMINAL_LINEAR_SPEED_THRESHOLD,
  TERMINAL_STABLE_DURATION,
  EpisodeRecorder,
  _append,
  _stream,
  validate_episode,
)
from kaihand_tactile_env.shared.simulation import ArmHandSimulation

from . import config as usb_config
from .buffered_h5 import BufferedH5File
from .execution import UsbInsertionExecutor
from .setup import initialize_for_insertion

RECORDING_CONTRACT = "usb_insert_taskspace_raw_v1"
OBSERVATION_CLOCK = "post_step_forward_v1"
INSERTION_METRIC_FIELDS = (
  "insertion_depth_m",
  "axial_resistance_n",
  "backstop_axial_resistance_n",
  "spring_axial_resistance_n",
  "spring_normal_load_n",
  "wall_normal_load_n",
  "backstop_normal_load_n",
  "axial_speed_m_s",
  "linear_speed_m_s",
  "angular_speed_rad_s",
  "maximum_socket_penetration_m",
  "orientation_error_rad",
)
INSERTION_FLAG_FIELDS = (
  "seated",
  "success",
  "shell_fits_aperture",
  "backstop_contact",
  "bottom_out_confirmed",
)
PROJECT_ROOT = Path(__file__).resolve().parents[4]
SOURCE_PATHS = (
  "scripts/workcell/record_usb_dataset.py",
  *(
    f"src/kaihand_tactile_env/tasks/usb_insert/{name}.py"
    for name in (
      "config",
      "setup",
      "grasp",
      "execution",
      "motion",
      "task",
      "precontact_noise",
      "recording",
      "buffered_h5",
      "batch",
    )
  ),
  *(
    f"src/kaihand_tactile_env/shared/{name}.py"
    for name in (
      "simulation",
      "recording",
      "taskspace_recording",
      "contact_tactile",
      "tactile",
      "config",
      "cameras",
      "rendering",
    )
  ),
)


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def controller_source_hashes() -> dict[str, str]:
  return {name: sha256_file(PROJECT_ROOT / name) for name in SOURCE_PATHS}


def episode_seeds(root_seed: int, episode_index: int) -> tuple[int, int]:
  if any(
    isinstance(value, bool) or not isinstance(value, int) or value < 0
    for value in (root_seed, episode_index)
  ):
    raise ValueError("root seed and episode index must be nonnegative integers")
  state = np.random.SeedSequence([root_seed, episode_index]).generate_state(
    2, dtype=np.uint64
  )
  return int(state[0]), int(state[1])


def recording_config(
  camera_hz: int = 30, include_overhead: bool = False,
  cameras: tuple[str, ...] = TRAINING_CAMERA_NAMES,
) -> WorkcellConfig:
  if (
    isinstance(camera_hz, bool)
    or not isinstance(camera_hz, (int, np.integer))
    or camera_hz <= 0
    or (camera_hz != 30 and 500 % camera_hz)
  ):
    raise ValueError("camera_hz must be 30 or a positive integer divisor of 500")
  if not isinstance(include_overhead, bool):
    raise ValueError("include_overhead must be a boolean")
  camera_names = tuple(cameras)
  if ("head" not in camera_names or len(set(camera_names)) != len(camera_names)
      or not set(camera_names).issubset(SHARED_CAMERA_NAMES)):
    raise ValueError("cameras must be unique shared names including head")
  if include_overhead and "overhead" not in camera_names:
    camera_names += ("overhead",)
  return WorkcellConfig(
    model_path=default_model_path("usb-insert"),
    physics_hz=500,
    control_hz=500,
    camera_hz=camera_hz,
    cameras=tuple(
      CameraConfig(name, 320, 240, depth=False, segmentation=False)
      for name in camera_names
    ),
  )


@dataclass(frozen=True)
class UsbRecordJob:
  output: Path
  episode_index: int
  root_seed: int
  xy_jitter_m: float = 0.010
  yaw_jitter_rad: float = math.radians(5)
  precontact_noise_std_m: float = 0.0005
  motion_profile: str = "fast"
  camera_hz: int = 30
  include_overhead: bool = False
  worker_count: int = 1
  cameras: tuple[str, ...] = TRAINING_CAMERA_NAMES

  def __post_init__(self):
    if isinstance(self.worker_count, bool) or self.worker_count not in (1, 2):
      raise ValueError("USB worker_count must be 1 or 2")
    episode_seeds(self.root_seed, self.episode_index)
    values = (self.xy_jitter_m, self.yaw_jitter_rad, self.precontact_noise_std_m)
    if not np.isfinite(values).all() or min(values) < 0:
      raise ValueError(
        "randomization ranges and noise sigma must be finite and nonnegative"
      )
    if self.motion_profile not in ("fast", "baseline"):
      raise ValueError("motion_profile must be fast or baseline")
    if self.output.suffix != ".h5":
      raise ValueError("USB raw output must end in .h5")
    recording_config(self.camera_hz, self.include_overhead, self.cameras)


class UsbRecordingSimulation(ArmHandSimulation):
  """Add cached-FK clock bookkeeping without changing the shared physics step."""

  drive_limit_n = None

  @property
  def observation_time(self) -> float:
    return min(
      float(self.data.time), getattr(self, "_observation_time", float(self.data.time))
    )

  def step(self, steps: int = 1) -> None:
    evaluated_at = float(self.data.time)
    super().step(steps)
    self._observation_time = max(evaluated_at, float(self.data.time) - self.timestep)

  def note_observation_refresh(self) -> None:
    """Record the existing monitor's mj_forward epoch; do not run it again."""
    self._observation_time = float(self.data.time)

  def drive_state(self) -> dict[str, float]:
    # This controller uses the original joint servo, with no poker drive mode.
    return {"drive_requested_fx_n": 0.0, "drive_actual_fx_n": 0.0}


class UsbEpisodeRecorder(EpisodeRecorder):
  """Reuse raw schemas and clocks, selecting only actual USB pad contact forces."""

  def __init__(self, path, simulation, config, **kwargs):
    if simulation.scene != "usb-insert":
      raise ValueError("UsbEpisodeRecorder requires the independent USB scene")
    ensure_new_episode(Path(path))
    self._insertion_state = None
    super().__init__(path, simulation, config, capture_taskspace=True, **kwargs)
    # Retain the common sampler/schema, changing only this task's disk writes.
    self._file = BufferedH5File(self._file)

  def _initialize(self, metadata):
    model = self.sim.model
    plug = model.body("usb_plug").id
    targets = tuple(
      model.geom(i).name
      for i in range(model.ngeom)
      if int(model.geom_bodyid[i]) == plug
      and (model.geom_contype[i] != 0 or model.geom_conaffinity[i] != 0)
    )
    self.contact_force_provider = SolverDistributedTactileProvider(
      model, self.sim.genesis_probe_layout, target_geom_names=targets
    )
    super()._initialize(metadata)
    self._file["tactile_contact_force"].attrs["clock_semantics"] = (
      "post-step monitor mj_forward observation at state/timestamp"
    )
    for camera in self.config.cameras:
      self._file[f"cameras/{camera.name}"].attrs["pose_clock_semantics"] = (
        "post-step monitor mj_forward pose; same epoch as RGB, tactile and state"
      )
    names = [
      model.joint(int(j)).name
      for side in ("left", "right")
      for j in self.sim._arm_joint_ids[side]
    ]
    ids = [int(j) for side in ("left", "right") for j in self.sim._arm_joint_ids[side]]
    group = self._file["model"]
    group.create_dataset(
      "arm_joint_names", data=names, dtype=self.h5py.string_dtype("utf-8")
    )
    group.create_dataset("arm_joint_limits_rad", data=model.jnt_range[ids])
    group.create_dataset(
      "arm_joint_qpos_indices",
      data=np.r_[self.sim._arm_qpos["left"], self.sim._arm_qpos["right"]],
    )
    group.create_dataset(
      "robot_joint_limits_rad",
      data=[model.jnt_range[self.sim._joint_id[name]] for name in self.sim.joint_names],
    )
    group.create_dataset("body_mass", data=model.body_mass)
    group.create_dataset("geom_friction", data=model.geom_friction)
    physics = self._file.create_group("physics")
    physics.attrs["state_index_reference"] = "/state/timestamp"
    physics.attrs["solver_clock"] = "post-step monitor mj_forward evaluation"
    physics.attrs["actuator_force_semantics"] = (
      "held control reevaluated at the resulting state by monitor mj_forward; "
      "not the integration step's pre-state actuator force"
    )
    _stream(physics, "solver_timestamp", (), np.float64)
    _stream(physics, "qfrc_applied", (model.nv,), np.float64)
    _stream(physics, "xfrc_applied", (model.nbody, 6), np.float64)
    _stream(physics, "qacc", (model.nv,), np.float64)
    _stream(physics, "actuator_force", (model.nu,), np.float64)
    _stream(physics, "noslip_iterations", (), np.int32)
    insertion = self._file.create_group("usb_insertion")
    insertion.attrs["schema_version"] = "usb_insertion_monitor_v1"
    insertion.attrs["contact_model_version"] = usb_config.CONTACT_MODEL_VERSION
    insertion.attrs["state_index_reference"] = "/state/timestamp"
    insertion.attrs["observation_clock"] = OBSERVATION_CLOCK
    insertion.attrs["source"] = (
      "initial monitor.measure snapshot; subsequent executor._state snapshots; "
      "recorder performs no dynamics or monitor updates"
    )
    insertion.attrs["axial_resistance_semantics"] = (
      "positive opposition to insertion from measured plug/socket contact forces; "
      "spring and backstop components are recorded separately, not synthesized"
    )
    _stream(insertion, "timestamp", (), np.float64)
    _stream(insertion, "state_index", (), np.int64)
    for name in INSERTION_METRIC_FIELDS:
      _stream(insertion, name, (), np.float64)
    for name in INSERTION_FLAG_FIELDS:
      _stream(insertion, name, (), np.bool_)

  def set_insertion_state(self, state):
    """Retain the controller's immutable measured snapshot; never evaluate it."""
    timestamp = float(state.timestamp)
    values = [float(getattr(state, name)) for name in INSERTION_METRIC_FIELDS]
    if not np.isfinite([timestamp, *values]).all():
      raise ValueError("USB insertion snapshot contains nonfinite metrics")
    if any(
      not isinstance(getattr(state, name), (bool, np.bool_))
      for name in INSERTION_FLAG_FIELDS
    ):
      raise ValueError("USB insertion snapshot flags must be booleans")
    if not math.isclose(timestamp, float(self.sim.data.time), rel_tol=0, abs_tol=1e-10):
      raise ValueError("USB insertion snapshot timestamp differs from current state")
    self._insertion_state = state

  def _record_state(self, phase):
    state = self._insertion_state
    if state is None:
      raise RuntimeError("USB recorder needs an insertion snapshot before each state")
    self.set_insertion_state(state)
    index = self._state_samples
    super()._record_state(phase)
    group = self._file["physics"]
    _append(group["solver_timestamp"], float(self.sim.observation_time))
    for name in ("qfrc_applied", "xfrc_applied", "qacc", "actuator_force"):
      _append(group[name], getattr(self.sim.data, name))
    _append(group["noslip_iterations"], self.sim.model.opt.noslip_iterations)
    insertion = self._file["usb_insertion"]
    _append(insertion["timestamp"], float(state.timestamp))
    _append(insertion["state_index"], index)
    for name in INSERTION_METRIC_FIELDS:
      _append(insertion[name], float(getattr(state, name)))
    for name in INSERTION_FLAG_FIELDS:
      _append(insertion[name], bool(getattr(state, name)))

  def insertion_stream_report(self):
    """Verify archived monitor rows before publishing; partial tails fail closed."""
    group = self._file["usb_insertion"]
    state_times = np.asarray(self._file["state/timestamp"][:], dtype=float)
    times = np.asarray(group["timestamp"][:], dtype=float)
    indices = np.asarray(group["state_index"][:], dtype=np.int64)
    counts = {
      name: group[name].shape[0]
      for name in (
        "timestamp",
        "state_index",
        *INSERTION_METRIC_FIELDS,
        *INSERTION_FLAG_FIELDS,
      )
    }
    valid = bool(
      len(state_times) > 0
      and all(count == len(state_times) for count in counts.values())
      and np.array_equal(times, state_times)
      and np.array_equal(indices, np.arange(len(state_times)))
    )
    terminal_recorded = bool(
      len(times)
      and math.isclose(
        float(times[-1]), float(self.sim.data.time), rel_tol=0, abs_tol=1e-10
      )
    )
    return {
      "valid": valid,
      "state_samples": len(state_times),
      "metric_samples": counts,
      "aligned_every_physics_step": valid,
      "terminal_monitor_row_recorded": terminal_recorded,
      "missing_or_partial_tail": not valid or not terminal_recorded,
    }


class TerminalEvidence:
  """Measure the already-executed post-release tail; never add a physics step."""

  def __init__(self):
    self.steps = self.stable_steps = 0
    self.linear_speed = self.angular_speed = 0.0

  def observe(self, simulation, phase):
    twist = simulation.object_twist("usb_plug")
    self.linear_speed = float(np.linalg.norm(twist[:3]))
    self.angular_speed = float(np.linalg.norm(twist[3:]))
    if phase not in ("retreat", "verify"):
      self.stable_steps = 0
      return
    self.steps += 1
    stable = (
      self.linear_speed < TERMINAL_LINEAR_SPEED_THRESHOLD
      and self.angular_speed < TERMINAL_ANGULAR_SPEED_THRESHOLD
    )
    self.stable_steps = self.stable_steps + 1 if stable else 0

  def report(self, timestep):
    return {
      "elapsed_seconds": self.steps * timestep,
      "stable_seconds": self.stable_steps * timestep,
      "linear_speed": self.linear_speed,
      "angular_speed": self.angular_speed,
      "steps": self.steps,
      "source": "recorded_post_release_velocity_window",
    }


def noise_trace_arrays(noise: dict, state_times: np.ndarray) -> tuple[dict, dict]:
  """Keep the nonuniform command trace alongside uniformly recorded real ctrl."""
  commands = noise.get("commands", [])
  phases = list(dict.fromkeys(command["phase"] for command in commands))
  shapes = {
    "time_s": (),
    "nominal_wrist_position_m": (3,),
    "applied_wrist_position_m": (3,),
    "random_offset_m": (3,),
    "recovery_offset_m": (3,),
    "arm_goal_rad": (7,),
    "gaussian_knot_draws": (),
  }
  arrays = {
    name: np.asarray([command[name] for command in commands], dtype=np.float64).reshape(
      (len(commands), *shape)
    )
    for name, shape in shapes.items()
  }
  arrays["phase_index"] = np.array(
    [phases.index(c["phase"]) for c in commands], dtype=np.int32
  )
  following = np.searchsorted(state_times, arrays["time_s"], side="right")
  arrays["state_index_at_or_before_issue"] = following.astype(np.int64) - 1
  arrays["first_future_state_index"] = np.where(
    following < len(state_times), following, -1
  ).astype(np.int64)
  contact = noise.get("first_contact")
  contact_index = None
  if contact is not None:
    matching = np.flatnonzero(
      np.isclose(state_times, contact["time_s"], rtol=0, atol=1e-10)
    )
    contact_index = int(matching[0]) if len(matching) else None
  return arrays, {
    **{key: value for key, value in noise.items() if key != "commands"},
    "phase_names": phases,
    "clock_semantics": "successful IK command issue time; nonuniform events, not a physics-rate trace",
    "physics_control_reference": "/commands/actuator_control at /state/timestamp",
    "state_index_semantics": "issue bracket in /state/timestamp; -1 means unavailable; new targets first affect a future physics step, not the previous step's ctrl",
    "first_contact_state_index": contact_index,
    "first_contact_clock": "post-step monitor mj_forward: force, FK and state timestamps coincide",
  }


def ensure_new_episode(output: Path) -> None:
  for path in (
    output,
    output.with_suffix(".h5.partial"),
    output.with_suffix(".json"),
    output.with_suffix(".result.json"),
  ):
    if path.exists() or path.is_symlink():
      raise FileExistsError(f"USB recording never overwrites existing evidence: {path}")


def write_json_new(path: Path, document: dict) -> None:
  with path.open("x", encoding="utf-8") as handle:
    json.dump(document, handle, indent=2, ensure_ascii=False, allow_nan=False)
    handle.write("\n")


def record_episode(job: UsbRecordJob, should_stop=lambda: False) -> dict:
  """Record one independent episode; only verified successful raw files finalize."""
  ensure_new_episode(job.output)
  job.output.parent.mkdir(parents=True, exist_ok=True)
  object_seed, noise_seed = episode_seeds(job.root_seed, job.episode_index)
  started = time.monotonic()
  source_hashes = controller_source_hashes()
  simulation = recorder = executor = None
  result = None
  outcome = {"success": False, "object_name": "usb_plug", "released": False}
  status, error = "failed", None
  errors = []
  evidence = TerminalEvidence()
  validation = None
  try:
    simulation = UsbRecordingSimulation(scene="usb-insert", add_genesis_probes=True)
    simulation.reset(seed=object_seed)
    initialization = initialize_for_insertion(
      simulation,
      seed=object_seed,
      xy_jitter_m=job.xy_jitter_m,
      yaw_jitter_rad=job.yaw_jitter_rad,
    )
    config = recording_config(job.camera_hz, job.include_overhead, job.cameras)
    if not math.isclose(
      simulation.timestep, 1 / config.physics_hz, rel_tol=0, abs_tol=1e-12
    ):
      raise ValueError("USB raw contract requires the actual 500 Hz physics timestep")
    recorder = UsbEpisodeRecorder(
      job.output,
      simulation,
      config,
      metadata={
        "recording_contract": RECORDING_CONTRACT,
        "observation_clock": OBSERVATION_CLOCK,
        "contact_model_version": usb_config.CONTACT_MODEL_VERSION,
        "insertion_resistance_model": (
          "passive spring-shoe compression and solver friction during insertion; "
          "physical backstop contact verifies active bottom-out before release; "
          "monitor metrics are direct post-step contact measurements"
        ),
        "episode_index": job.episode_index,
        "root_seed": job.root_seed,
        "seed": object_seed,
        "object_seed": object_seed,
        "noise_seed": noise_seed,
        "initial_pose_randomization": initialization,
        "precontact_noise_std_m": job.precontact_noise_std_m,
        "motion_profile": job.motion_profile,
        "object_name": "usb_plug",
        "side": "right",
        "controller_source_sha256": source_hashes,
        "base_model_sha256": sha256_file(config.model_path),
        "base_model_fingerprint": model_fingerprint(config.model_path),
        "state_and_tactile_hz": 500,
        "wrist_feedback_hz": 50,
        "camera_contract": (
          " and ".join(camera.name for camera in config.cameras)
          + " RGB 320x240; current rendered camera, cached full site SE3; no depth"
        ),
        "include_overhead": job.include_overhead,
        "force_contract": "true solver pad-on-usb forces; signed tangent taxels are conservative spatial estimates in N",
        "control_contract": "existing arm/hand actuator targets only; no object pose or force assistance",
        "worker_count": job.worker_count,
        "worker_pid": os.getpid(),
        "camera_sampling": "nearest_500hz_step_to_nominal_deadline; exact terminal frame retained",
        "thread_environment": {
          name: os.environ.get(name)
          for name in (
            "OPENBLAS_NUM_THREADS",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "LP_NUM_THREADS",
          )
        },
      },
    )

    def observe(current, phase):
      # UsbInsertionExecutor._step already called monitor.update()/mj_forward
      # after integration.  This only records that refresh; no extra physics,
      # force evaluation, command update or RNG draw is introduced here.
      current.note_observation_refresh()
      recorder.set_insertion_state(executor._state)
      evidence.observe(current, phase)
      recorder.observe(current, phase)

    executor = UsbInsertionExecutor(
      simulation,
      observer=observe,
      should_stop=should_stop,
      precontact_noise_std_m=job.precontact_noise_std_m,
      noise_seed=noise_seed,
      motion_profile=job.motion_profile,
    )
    # The executor constructor creates its monitor but does not yet measure.
    # Acquire the initial snapshot once at reset time, before recording RGB and
    # tactile. Every later sample comes directly from the executor's _state.
    initial_insertion = executor.monitor.measure()
    simulation.note_observation_refresh()
    recorder.set_insertion_state(initial_insertion)
    recorder.record_initial()
    result = asdict(executor.execute())
    outcome.update(result)
    outcome["executor_success"] = result["success"]
    outcome["terminal_stability"] = evidence.report(simulation.timestep)
    verified = (
      result["success"]
      and result["released"]
      and result["grasp_verified"]
      and result["active_bottom_out_confirmed"]
      and result["insertion"]["success"]
      and result["insertion"]["seated"]
      and evidence.stable_steps * simulation.timestep + 1e-9 >= TERMINAL_STABLE_DURATION
    )
    outcome["success"] = bool(verified)
    status = (
      "success"
      if verified
      else "cancelled"
      if result["failure_reason"] == "cancelled"
      else "failed"
    )
    if not verified:
      error = (
        result["failure_reason"]
        or "recorded terminal success/stability evidence is incomplete"
      )
    # Relabel/capture the exact existing terminal state, never advance physics.
    recorder.record_terminal("terminal_settle" if verified else status)
  except KeyboardInterrupt:
    status, error = "cancelled", "KeyboardInterrupt"
    outcome["success"] = False
  except Exception:
    status, error = "exception", traceback.format_exc()
    outcome["success"] = False
  finally:
    if simulation is not None:
      outcome["final_object_pose"] = simulation.object_pose("usb_plug").tolist()
      outcome["final_object_twist"] = simulation.object_twist("usb_plug").tolist()
    try:
      final_hashes = controller_source_hashes()
      outcome["controller_source_sha256_at_end"] = final_hashes
      outcome["source_files_unchanged"] = final_hashes == source_hashes
      if final_hashes != source_hashes:
        outcome["success"] = False
        status, error = "exception", "controller source changed during recording"
    except Exception as exception:
      outcome["success"] = False
      status, error = "exception", f"could not verify source identity: {exception}"
    if recorder is not None:
      # Even interrupted/failed runs keep all observations written so far, plus
      # the actual control trace and an explicitly unsuccessful terminal label.
      try:
        insertion_report = recorder.insertion_stream_report()
        outcome["usb_insertion_recording"] = insertion_report
        if (
          not insertion_report["valid"]
          or not insertion_report["terminal_monitor_row_recorded"]
        ):
          errors.append("USB insertion monitor stream is incomplete or misaligned")
          outcome["success"] = False
          if status == "success":
            status = "exception"
            error = "USB insertion monitor stream is incomplete or misaligned"
      except Exception as exception:
        errors.append(f"USB insertion monitor archive: {exception}")
        outcome["success"] = False
        if status == "success":
          status = "exception"
          error = f"USB insertion monitor archive: {exception}"
      try:
        noise = (
          result["precontact_noise"]
          if result
          else executor._noise.report()
          if executor is not None
          else None
        )
        if noise is not None:
          outcome["precontact_noise"] = noise
          trace, trace_metadata = noise_trace_arrays(
            noise, recorder._file["state/timestamp"][:]
          )
          recorder.write_precontact_noise_trace(trace, metadata=trace_metadata)
      except Exception as exception:
        errors.append(f"noise archive: {exception}")
        outcome["success"] = False
        status = "exception"
      outcome["recording_status"] = status
      outcome["recording_error"] = error
      outcome["recording_errors"] = errors
      recorder.set_outcome(outcome)
      recorder.close(finalize=False)
  if recorder is not None and outcome["success"]:
    try:
      validation = asdict(validate_episode(recorder.partial_path))
      if not validation["valid"]:
        raise ValueError(f"raw recording validation failed: {validation['errors']}")
      # Exclusive hard-link publication avoids EpisodeRecorder's historical
      # overwrite behavior; source and destination live in the same directory.
      os.link(recorder.partial_path, job.output)
      recorder.partial_path.unlink()
      write_json_new(
        job.output.with_suffix(".json"),
        {
          "schema_version": "kaihand_tactile_episode_v1",
          "episode": job.output.name,
          "sha256": sha256_file(job.output),
          "state_samples": validation["state_samples"],
          "camera_samples": validation["camera_samples"],
          "outcome": outcome,
        },
      )
    except Exception as exception:
      status, error = "exception", str(exception)
      outcome["success"] = False
      outcome["recording_status"] = status
      outcome["recording_error"] = error
      archived = job.output if job.output.exists() else recorder.partial_path
      with recorder.h5py.File(archived, "r+") as file:
        file.attrs["outcome_json"] = json.dumps(
          outcome, ensure_ascii=False, sort_keys=True
        )
  report = {
    "episode_index": job.episode_index,
    "root_seed": job.root_seed,
    "object_seed": object_seed,
    "noise_seed": noise_seed,
    "motion_profile": job.motion_profile,
    "status": status,
    "success": status == "success",
    "raw_path": str(
      job.output if job.output.exists() else job.output.with_suffix(".h5.partial")
    ),
    "failure_reason": error,
    "recording_errors": errors,
    "wall_duration_s": time.monotonic() - started,
    "recording_performance": (
      {
        "state_sample_seconds": recorder._state_write_seconds,
        "camera_capture_seconds": recorder._camera_write_seconds,
        "storage": recorder._file.statistics(),
      }
      if recorder is not None and isinstance(recorder._file, BufferedH5File)
      else None
    ),
    "validation": validation,
    "outcome": outcome,
  }
  write_json_new(job.output.with_suffix(".result.json"), report)
  return report
