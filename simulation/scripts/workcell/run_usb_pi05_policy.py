#!/usr/bin/env python3
"""Run a manifest-bound pi0.5 joint policy in the USB-insert simulation."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from kaihand_tactile_env.shared.config import CameraConfig
from kaihand_tactile_env.shared.evaluation_guards import (
  DEFAULT_MAX_SIM_SECONDS,
  GUARD_MODES,
  configure_evaluation_guards,
  ensure_evaluation_physics,
  evaluation_guard_report,
  recover_nonfinite_action,
  task_guard,
)
from kaihand_tactile_env.shared.evaluation_source_integrity import (
  verify_evaluation_sources,
)
from kaihand_tactile_env.shared.pi05_deployment_integrity import stable_file_bytes
from kaihand_tactile_env.shared.policy_cameras import (
  image_shape_hwc,
  pi05_image_payload,
  policy_camera_names,
)
from kaihand_tactile_env.shared.rendering import WorkcellRenderer
from kaihand_tactile_env.shared.simulation import ArmHandSimulation
from kaihand_tactile_env.tasks.usb_insert import config as usb_config
from kaihand_tactile_env.tasks.usb_insert.review_metrics import UsbPolicyOutcome
from kaihand_tactile_env.tasks.usb_insert.setup import initialize_for_insertion

try:
  from prepare_usb_pi05_deployment import (
    _canonical_hash,
    checkpoint_contract,
    evaluation_source_paths,
    openpi_source_paths,
    validate_deployment_identity,
  )
except ModuleNotFoundError as error:
  if error.name != "prepare_usb_pi05_deployment":
    raise
  from scripts.workcell.prepare_usb_pi05_deployment import (
    _canonical_hash,
    checkpoint_contract,
    evaluation_source_paths,
    openpi_source_paths,
    validate_deployment_identity,
  )

CONTROL_HZ = 30
CONTROLLER = "direct-30hz-pi05-absolute-joint-v1"
LINEAR_CONTROLLER = "pi05-absolute-joint-physics-interpolation-v2"
ROOT = Path(__file__).resolve().parents[2]
PENETRATION_GUARD_THRESHOLD_M = 0.0003


def current_controller_target(simulation, joint_names: list[str]) -> np.ndarray:
  """Read commanded targets; measured qpos is not an interpolation endpoint."""
  if len(joint_names) != 27 or len(set(joint_names)) != 27:
    raise ValueError("controller target requires 27 unique policy joints")
  if joint_names[:7] != [f"right_arm_joint{index}" for index in range(1, 8)]:
    raise ValueError("the first seven policy joints must be the ordered right arm")
  _, targets, names = simulation.command_state()
  if len(names) != len(set(names)):
    raise ValueError("simulator controller hand-target names are not unique")
  by_name = dict(zip(names, targets, strict=True))
  try:
    value = np.concatenate(
      (
        np.asarray(simulation.arm_goal["right"], dtype=float),
        np.asarray([by_name[name] for name in joint_names[7:]], dtype=float),
      )
    )
  except KeyError as error:
    raise ValueError(
      f"simulator has no controller target for {error.args[0]!r}"
    ) from error
  if value.shape != (27,) or not np.isfinite(value).all():
    raise ValueError("simulator controller target must be finite with width 27")
  return value


def linear_substep_action(
  start_action,
  end_action,
  *,
  interval_start_s: float,
  interval_end_s: float,
  current_time_s: float,
  timestep_s: float,
) -> np.ndarray:
  """Interpolate the next physics-step target between clipped endpoints."""
  start = np.asarray(start_action, dtype=np.float64)
  end = np.asarray(end_action, dtype=np.float64)
  duration = interval_end_s - interval_start_s
  if (
    start.shape != (27,)
    or end.shape != (27,)
    or not np.isfinite(start).all()
    or not np.isfinite(end).all()
    or not np.isfinite(
      (interval_start_s, interval_end_s, current_time_s, timestep_s)
    ).all()
    or duration <= 0
    or timestep_s <= 0
  ):
    raise ValueError("finite width-27 actions and positive finite timing are required")
  next_time = min(current_time_s + timestep_s, interval_end_s)
  alpha = float(np.clip((next_time - interval_start_s) / duration, 0, 1))
  return start + alpha * (end - start)


def action_execution(deployment: dict) -> str:
  if deployment.get("schema") != "usb_pi05_deployment_v2":
    return "zoh"
  required = deployment.get("required_action_interpolation")
  contract = deployment["training_provenance"]["record"]["action_contract"][
    "evaluation_execution"
  ]
  if (
    required != "linear"
    or contract.get("physical_substep_interpolation_required") != "linear"
    or contract.get("endpoint_rate_hz") != CONTROL_HZ
    or contract.get("zero_order_hold_equivalent") is not False
  ):
    raise RuntimeError("USB v2 requires physical-substep linear action interpolation")
  return "linear"


def evaluation_validity(deployment: dict, guard_mode: str) -> dict:
  reasons = []
  if deployment.get("schema") == "usb_pi05_deployment_v1":
    reasons.append("legacy_v1_unbound_deployment")
  elif (
    deployment.get("training_provenance", {})
    .get("record", {})
    .get("source_environment")
    is None
  ):
    reasons.append("source_environment_unbound")
  if guard_mode == "observe":
    reasons.append("guard_mode_observe")
  return {
    "diagnostic_reasons": reasons,
    "diagnostic_only": bool(reasons),
    "formal_metrics_valid": not reasons,
  }


def inspect_trial_integrity(
  path: Path, deployment: dict, expected_sha256: str | None = None
) -> dict:
  """Authenticate each child trial without importing the model environment."""
  result = {"status": "verified", "errors": []}
  try:
    payload, sha = stable_file_bytes(path)
    result["manifest_sha256"] = sha
    if json.loads(payload) != deployment or (
      expected_sha256 is not None and sha != expected_sha256
    ):
      raise RuntimeError("manifest content differs from batch snapshot")
    if deployment.get("schema") != "usb_pi05_deployment_v2":
      return {**result, "status": "unbound_historical_v1"}
    validate_deployment_identity(deployment)
    expected = (
      deployment["model_config_name"],
      deployment["normalizer_asset_id"],
      deployment["training_provenance"],
    )
    if (
      checkpoint_contract(
        Path(deployment["checkpoint_path"]), verify_numeric_normalization=False
      )
      != expected
    ):
      raise RuntimeError("checkpoint training provenance/config/normalizer changed")
    if (
      _canonical_hash(deployment["checkpoint_files"]) != deployment["checkpoint_sha256"]
    ):
      raise RuntimeError(
        "checkpoint SHA-256 does not authenticate checkpoint inventory"
      )
    for label, project, rows, aggregate, inventory in (
      (
        "evaluation_source",
        ROOT,
        deployment["evaluation_source_files"],
        deployment["evaluation_source_sha256"],
        evaluation_source_paths(),
      ),
      (
        "openpi_source",
        Path(deployment["model_project_path"]),
        deployment["openpi_source_files"],
        deployment["openpi_source_sha256"],
        openpi_source_paths(Path(deployment["model_project_path"])),
      ),
    ):
      proof = verify_evaluation_sources(
        {"evaluation_source_files": rows, "evaluation_source_sha256": aggregate},
        project,
      )
      result[label] = proof
      if proof["status"] != "verified" or {row["path"] for row in rows} != {
        entry.relative_to(project).as_posix() for entry in inventory
      }:
        raise RuntimeError(f"{label} content or inventory changed")
    result["training_provenance"] = {"status": "verified", "binding_status": "bound"}
  except (KeyError, OSError, RuntimeError, ValueError, TypeError) as error:
    result.update(status="mismatch", errors=[f"{type(error).__name__}: {error}"])
  return result


def _mark_record_error(report: dict, error: Exception) -> None:
  message = f"{type(error).__name__}: {error}"
  previous = report.get("record_error")
  report["record_error"] = message if previous is None else f"{previous}; {message}"
  report["stop_reason"] = "fatal_error"
  report["formal_metrics_valid"] = False


def parse_args(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--server", default="ws://127.0.0.1:18783")
  parser.add_argument("--deployment-manifest", type=Path, required=True)
  parser.add_argument("--expected-deployment-manifest-sha256")
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--xy-jitter-mm", type=float, default=10.0)
  parser.add_argument("--yaw-jitter-deg", type=float, default=5.0)
  parser.add_argument("--execute-steps", type=int, default=8)
  parser.add_argument("--staged-evaluation-version", choices=("staged_v1",), default=None)
  parser.add_argument("--max-requests", type=int, default=0)
  parser.add_argument("--max-sim-seconds", type=float, default=DEFAULT_MAX_SIM_SECONDS)
  parser.add_argument(
    "--guard-mode",
    choices=GUARD_MODES,
    default="observe",
    help=(
      "observe records task force/penetration/drop violations and continues; "
      "strict terminates on the first task violation"
    ),
  )
  parser.add_argument(
    "--disable-penetration-guard",
    action="store_true",
    help=(
      "Deprecated observe-mode alias: omit penetration-limit events while all "
      "other task violations remain observable"
    ),
  )
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--record", action=argparse.BooleanOptionalAction, default=True)
  parser.add_argument("--record-fps", type=int, choices=(5, 10), default=10)
  parser.add_argument("--reference-dataset", type=Path)
  parser.add_argument("--reference-episode-index", type=int, default=0)
  parser.add_argument("--review-width", type=int, default=1920)
  parser.add_argument("--review-height", type=int, default=1080)
  parser.add_argument("--review-render-width", type=int, default=640)
  parser.add_argument("--review-render-height", type=int, default=480)
  parser.add_argument(
    "--review-second-camera",
    choices=("global", "right_wrist", "overhead"),
    default="global",
  )
  parser.add_argument(
    "--save-first-request", action=argparse.BooleanOptionalAction, default=True
  )
  args = parser.parse_args(argv)
  if args.seed < 0 or args.max_requests < 0 or args.execute_steps <= 0:
    parser.error("seed/requests must be nonnegative; execute-steps must be positive")
  for key in ("xy_jitter_mm", "yaw_jitter_deg", "max_sim_seconds"):
    value = getattr(args, key)
    if not np.isfinite(value) or value < 0 or (key == "max_sim_seconds" and value == 0):
      parser.error(f"{key} must be finite and nonnegative")
  if (
    min(
      args.review_width,
      args.review_height,
      args.review_render_width,
      args.review_render_height,
    )
    <= 0
  ):
    parser.error("review dimensions must be positive")
  if args.review_width % 2 or args.review_height % 2:
    parser.error("review output dimensions must be even")
  if args.disable_penetration_guard and args.guard_mode == "strict":
    parser.error("--disable-penetration-guard is incompatible with --guard-mode strict")
  return args


def load_deployment_manifest(path: Path) -> tuple[Path, dict]:
  resolved = path.expanduser().resolve()
  deployment = json.loads(resolved.read_text(encoding="utf-8"))
  required = (
    "schema",
    "task",
    "deployment_id",
    "model_family",
    "checkpoint_path",
    "checkpoint_sha256",
    "checkpoint_step",
    "prediction_horizon",
    "model_action_dim",
    "action_dim",
    "control_hz",
    "joint_names",
    "observation_contract",
    "action_representation",
    "instruction",
  )
  missing = [key for key in required if key not in deployment]
  if missing:
    raise RuntimeError(f"deployment manifest missing fields: {missing}")
  if deployment["schema"] not in {"usb_pi05_deployment_v1", "usb_pi05_deployment_v2"}:
    raise RuntimeError("unsupported deployment manifest schema")
  if deployment["task"] != "usb-insert" or deployment["model_family"] != "pi0.5":
    raise RuntimeError("runner requires a USB pi0.5 deployment")
  if deployment["schema"] == "usb_pi05_deployment_v2":
    validate_deployment_identity(deployment)
  return resolved, deployment


def validate_server(metadata: dict, deployment: dict, execute_steps: int) -> int:
  expected = {
    "evaluation_task": "usb-insert",
    "deployment_id": deployment["deployment_id"],
    "model_family": "pi0.5",
    "checkpoint_path": deployment["checkpoint_path"],
    "checkpoint_sha256": deployment["checkpoint_sha256"],
    "prediction_horizon": deployment["prediction_horizon"],
    "action_horizon": deployment["prediction_horizon"],
    "model_action_dim": deployment["model_action_dim"],
    "action_dim": deployment["action_dim"],
    "control_hz": deployment["control_hz"],
    "joint_names": deployment["joint_names"],
    "observation_contract": deployment["observation_contract"],
    "action_representation": deployment["action_representation"],
  }
  if deployment.get("schema") == "usb_pi05_deployment_v2":
    expected.update(
      {
        key: deployment[key]
        for key in (
          "model_config_name",
          "normalizer_asset_id",
          "training_provenance",
          "required_action_interpolation",
        )
      }
    )
  mismatches = {
    key: {"expected": value, "actual": metadata.get(key)}
    for key, value in expected.items()
    if metadata.get(key) != value
  }
  if mismatches:
    raise RuntimeError(f"server deployment identity mismatch: {mismatches}")
  horizon = int(metadata["prediction_horizon"])
  if not 1 <= execute_steps <= horizon:
    raise RuntimeError(f"execute_steps must be in [1, {horizon}], got {execute_steps}")
  if int(metadata["control_hz"]) != CONTROL_HZ:
    raise RuntimeError(f"runner requires {CONTROL_HZ} Hz policy control")
  observation = metadata["observation_contract"]
  policy_camera_names(observation)
  image_shape_hwc(observation)
  if observation.get("tactile_sent_to_model") is not False:
    raise RuntimeError("this pi0.5 checkpoint must not receive tactile input")
  return horizon


def right_joint_state(
  simulation: ArmHandSimulation, joint_names: list[str]
) -> np.ndarray:
  if len(joint_names) != 27 or len(set(joint_names)) != 27:
    raise RuntimeError("deployment must declare 27 distinct joint names")
  try:
    state = np.asarray(
      [simulation.data.qpos[simulation._qpos_address[name]] for name in joint_names],
      dtype=np.float32,
    )
  except KeyError as error:
    raise RuntimeError(f"simulator is missing model joint {error.args[0]!r}") from error
  if state.shape != (27,) or not np.isfinite(state).all():
    raise RuntimeError("right-side policy state is nonfinite or malformed")
  return state


def validated_actions(
  response: dict,
  *,
  horizon: int,
  action_dim: int,
) -> np.ndarray:
  if not isinstance(response, dict) or "actions" not in response:
    raise RuntimeError("pi0.5 server response has no actions")
  actions = np.asarray(response["actions"], dtype=np.float64)
  if actions.shape != (horizon, action_dim):
    raise RuntimeError(
      f"expected policy actions shape {(horizon, action_dim)}, got {actions.shape}"
    )
  if not np.isfinite(actions).all():
    raise RuntimeError("policy actions contain nonfinite values")
  return actions


def joint_limits(
  simulation: ArmHandSimulation, joint_names: list[str]
) -> tuple[np.ndarray, np.ndarray]:
  ids = np.asarray([simulation._joint_id[name] for name in joint_names], dtype=int)
  ranges = np.asarray(simulation.model.jnt_range[ids], dtype=np.float64)
  if ranges.shape != (27, 2) or np.any(ranges[:, 0] >= ranges[:, 1]):
    raise RuntimeError("simulator has invalid right-side joint limits")
  return ranges[:, 0], ranges[:, 1]


def guard_usb_state(
  outcome: UsbPolicyOutcome,
  simulation: ArmHandSimulation,
  *,
  penetration_guard_enabled: bool = True,
) -> None:
  # Numerical faults are terminal and take precedence over observable task limits.
  ensure_evaluation_physics(simulation)
  state = outcome.state
  task_guard(
    simulation,
    state.socket_normal_load_n > usb_config.MAX_SOCKET_NORMAL_LOAD_N,
    "usb_socket_normal_load",
    "USB socket total normal load exceeded task limit",
    measured=state.socket_normal_load_n,
    threshold=usb_config.MAX_SOCKET_NORMAL_LOAD_N,
  )
  task_guard(
    simulation,
    state.axial_resistance_n > usb_config.MAX_SOCKET_AXIAL_FORCE_N,
    "usb_socket_axial_resistance",
    "USB socket axial resistance exceeded task limit",
    measured=state.axial_resistance_n,
    threshold=usb_config.MAX_SOCKET_AXIAL_FORCE_N,
  )
  task_guard(
    simulation,
    state.wall_normal_load_n > usb_config.MAX_SOCKET_WALL_LOAD_N,
    "usb_socket_wall_load",
    "USB socket side-wall load exceeded task limit",
    measured=state.wall_normal_load_n,
    threshold=usb_config.MAX_SOCKET_WALL_LOAD_N,
  )
  if penetration_guard_enabled:
    task_guard(
      simulation,
      state.maximum_socket_penetration_m > PENETRATION_GUARD_THRESHOLD_M,
      "usb_socket_penetration",
      "USB socket penetration exceeded 0.3 mm",
      measured=state.maximum_socket_penetration_m,
      threshold=PENETRATION_GUARD_THRESHOLD_M,
    )
  plug_height = float(simulation.object_pose("usb_plug")[2])
  task_guard(
    simulation,
    plug_height < 0.5,
    "usb_plug_dropped",
    "USB plug fell below work surface",
    measured=plug_height,
    threshold=0.5,
  )


def apply_action(
  simulation: ArmHandSimulation,
  joint_names: list[str],
  action: np.ndarray,
  lower: np.ndarray,
  upper: np.ndarray,
) -> int:
  # The simulator's position servos provide their normal velocity/force safety.
  # We additionally bound extrapolated policy values to physical joint limits.
  clipped = np.clip(action, lower, upper)
  clipped_count = int(np.count_nonzero(np.abs(clipped - action) > 1.0e-9))
  simulation.set_arm_joint_goal("right", clipped[:7])
  accepted = simulation.set_hand_joint_targets(joint_names[7:], clipped[7:])
  if accepted != 20:
    raise RuntimeError(f"right hand accepted only {accepted}/20 joint targets")
  return clipped_count


class ActionStepper:
  """Persistent controller endpoints, including across requests and recovery."""

  def __init__(self, simulation, joint_names, lower, upper, interpolation):
    self.simulation = simulation
    self.joint_names = joint_names
    self.lower, self.upper = lower, upper
    self.interpolation = interpolation
    self.control_tick = 0
    self.epoch_s = float(simulation.data.time)
    self.previous_endpoint = (
      np.clip(current_controller_target(simulation, joint_names), lower, upper)
      if interpolation == "linear"
      else None
    )

  def advance(self, action, *, max_sim_seconds, after_step, should_stop) -> int:
    """Clip each control endpoint once and apply every linear physics substep."""
    self.control_tick += 1
    target_time = min(max_sim_seconds, self.epoch_s + self.control_tick / CONTROL_HZ)
    interval_start = float(self.simulation.data.time)
    start = None if self.previous_endpoint is None else self.previous_endpoint.copy()
    # Rejected predictions hold the last valid controller command, not qpos.
    endpoint = start if action is None else np.clip(action, self.lower, self.upper)
    clipped = (
      0 if action is None else int(np.count_nonzero(np.abs(endpoint - action) > 1e-9))
    )
    if action is not None and self.interpolation == "zoh":
      apply_action(self.simulation, self.joint_names, endpoint, self.lower, self.upper)
    while self.simulation.data.time < target_time - 1e-12:
      if self.interpolation == "linear":
        command = linear_substep_action(
          start,
          endpoint,
          interval_start_s=interval_start,
          interval_end_s=target_time,
          current_time_s=float(self.simulation.data.time),
          timestep_s=self.simulation.timestep,
        )
        if apply_action(
          self.simulation, self.joint_names, command, self.lower, self.upper
        ):
          raise RuntimeError("interpolation between clipped endpoints was clipped")
        self.previous_endpoint = command.copy()
      before_time = float(self.simulation.data.time)
      self.simulation.step()
      after_step(before_time)
      if should_stop():
        break
    return clipped


def run(args) -> dict:
  from openpi_client import websocket_client_policy

  output = args.output_dir.expanduser().resolve()
  deployment_path, deployment = load_deployment_manifest(args.deployment_manifest)
  output.mkdir(parents=True, exist_ok=False)
  interpolation = action_execution(deployment)
  validity = evaluation_validity(deployment, args.guard_mode)
  report = {
    "task": "usb-insert",
    "controller": LINEAR_CONTROLLER if interpolation == "linear" else CONTROLLER,
    "action_interpolation": interpolation,
    "controller_contract": {
      "endpoint_rate_hz": CONTROL_HZ,
      "initial_interpolation_start": "current simulator arm/hand controller targets",
      "subsequent_interpolation_start": "previous clipped action endpoint, persistent across chunks and replans",
      "endpoint_clipping": "once per 30 Hz action before physics substeps",
      "physics_substeps": "linear interpolation at every simulator physics step"
      if interpolation == "linear"
      else "zero-order hold of each clipped endpoint",
    },
    "server": args.server,
    "seed": args.seed,
    "instruction": deployment["instruction"],
    "deployment_manifest": str(deployment_path),
    "deployment_id": deployment["deployment_id"],
    "checkpoint_path": deployment["checkpoint_path"],
    "checkpoint_sha256": deployment["checkpoint_sha256"],
    "model_family": deployment["model_family"],
    "model_action_dim": deployment["model_action_dim"],
    "prediction_horizon": deployment["prediction_horizon"],
    "action_dim": deployment["action_dim"],
    "observation_contract": deployment["observation_contract"],
    "action_representation": deployment["action_representation"],
    "execute_steps": args.execute_steps,
    "max_sim_seconds": args.max_sim_seconds,
    "control_hz": CONTROL_HZ,
    "replan_period_s": args.execute_steps / CONTROL_HZ,
    "guard_mode": args.guard_mode,
    **validity,
    "task_success": False,
    "penetration_guard_enabled": not args.disable_penetration_guard,
    "penetration_guard_threshold_m": PENETRATION_GUARD_THRESHOLD_M,
    "expert_used": False,
    "tactile_sent_to_model": False,
    "render_shadows": False,
  }
  started = time.monotonic()
  client = simulation = recorder = outcome = None
  staged_monitor = staged_log = None
  staged_version = getattr(args, "staged_evaluation_version", None)
  report["full_duration_evaluation"] = bool(staged_version)
  requests = action_steps = clipped_values = control_tick = 0
  error_text = None
  first_penetration_limit_exceeded_s = None
  try:
    report["deployment_integrity"] = inspect_trial_integrity(
      deployment_path,
      deployment,
      getattr(args, "expected_deployment_manifest_sha256", None),
    )
    if report["deployment_integrity"]["status"] == "mismatch":
      raise RuntimeError(
        f"trial deployment integrity mismatch: {report['deployment_integrity']['errors']}"
      )
    client = websocket_client_policy.WebsocketClientPolicy(args.server)
    metadata = client.get_server_metadata()
    horizon = validate_server(metadata, deployment, args.execute_steps)
    report["server_metadata"] = metadata
    report["prediction_horizon"] = horizon
    print(
      f"[policy] connected step={deployment['checkpoint_step']} H={horizon} "
      f"execute_steps={args.execute_steps}",
      flush=True,
    )

    simulation = ArmHandSimulation(scene="usb-insert")
    simulation.reset(seed=args.seed)
    report["initial_randomization"] = initialize_for_insertion(
      simulation,
      seed=args.seed,
      xy_jitter_m=args.xy_jitter_mm / 1000.0,
      yaw_jitter_rad=float(np.deg2rad(args.yaw_jitter_deg)),
    )
    configure_evaluation_guards(simulation, args.guard_mode)
    ensure_evaluation_physics(simulation)
    report["initial_plug_pose"] = simulation.object_pose("usb_plug").tolist()
    outcome = UsbPolicyOutcome(simulation)
    if staged_version:
      from kaihand_tactile_env.shared.staged_evaluation import StagedEvaluationMonitor
      from kaihand_tactile_env.shared.staged_recording import StagedMetricView, StagedStateLog

      staged_monitor = StagedEvaluationMonitor("usb-insert", simulation, version=staged_version)
      staged_monitor.update(simulation, task_state=outcome.state)
      staged_log = StagedStateLog(output, staged_monitor, identity=report)
      staged_log.record(0)
      video_metrics = StagedMetricView(outcome, staged_monitor)
    else:
      video_metrics = outcome
    if outcome.state.maximum_socket_penetration_m > PENETRATION_GUARD_THRESHOLD_M:
      first_penetration_limit_exceeded_s = float(simulation.data.time)
    guard_usb_state(
      outcome,
      simulation,
      penetration_guard_enabled=not args.disable_penetration_guard,
    )
    joint_names = list(deployment["joint_names"])
    lower, upper = joint_limits(simulation, joint_names)
    stepper = ActionStepper(simulation, joint_names, lower, upper, interpolation)
    observation = deployment["observation_contract"]
    camera_names = policy_camera_names(observation)
    image_height, image_width, _ = image_shape_hwc(observation)
    cameras = {
      name: CameraConfig(
        name,
        width=image_width,
        height=image_height,
        rgb=True,
        depth=False,
        segmentation=False,
      )
      for name in camera_names
    }
    with (
      WorkcellRenderer(
        simulation.model, tuple(cameras.values()), shadows=False
      ) as renderer,
      (output / "requests.jsonl").open("x", encoding="utf-8") as log,
    ):
      report["render_backend"] = renderer.backend_info
      if args.record:
        from kaihand_tactile_env.shared.evaluation_video import EvaluationVideo

        recorder = EvaluationVideo(
          simulation,
          output,
          fps=args.record_fps,
          width=args.review_width,
          height=args.review_height,
          render_width=args.review_render_width,
          render_height=args.review_render_height,
          second_camera=args.review_second_camera,
          include_model_wrist=(
            "right_wrist" in camera_names and args.review_second_camera != "right_wrist"
          ),
          include_review_wrist=(
            "right_wrist" not in camera_names
            and args.review_second_camera != "right_wrist"
          ),
          comparison=True,
          reference_dataset=args.reference_dataset,
          reference_episode_index=args.reference_episode_index,
          metrics=video_metrics,
          heading="USB INSERT pi0.5 MODEL EVALUATION",
          metadata={
            "server": args.server,
            "seed": args.seed,
            "model_family": deployment["model_family"],
            "checkpoint_path": deployment["checkpoint_path"],
            "checkpoint_sha256": deployment["checkpoint_sha256"],
            "prediction_horizon": horizon,
            "model_action_dim": deployment["model_action_dim"],
            "action_dim": deployment["action_dim"],
            "execute_steps": args.execute_steps,
            "control_hz": CONTROL_HZ,
            "replan_period_s": args.execute_steps / CONTROL_HZ,
            "max_sim_seconds": args.max_sim_seconds,
            "guard_mode": args.guard_mode,
            **validity,
            "action_interpolation": interpolation,
            "controller": report["controller"],
            "controller_contract": report["controller_contract"],
            "penetration_guard_enabled": not args.disable_penetration_guard,
            "penetration_guard_threshold_m": PENETRATION_GUARD_THRESHOLD_M,
            "observation_contract": deployment["observation_contract"],
            "action_representation": deployment["action_representation"],
            "initial_randomization": report["initial_randomization"],
          },
        )
        recorder.capture(0, outcome.stage())

      def advance_held_or_applied_tick(action=None) -> None:
        nonlocal \
          action_steps, \
          control_tick, \
          clipped_values, \
          first_penetration_limit_exceeded_s

        def after_step(before_time):
          nonlocal first_penetration_limit_exceeded_s
          # Numerical integrity is checked before task metrics touch a bad state.
          ensure_evaluation_physics(simulation, before_time=before_time)
          outcome.update(simulation)
          if (
            first_penetration_limit_exceeded_s is None
            and outcome.state.maximum_socket_penetration_m
            > PENETRATION_GUARD_THRESHOLD_M
          ):
            first_penetration_limit_exceeded_s = float(simulation.data.time)
          guard_usb_state(
            outcome,
            simulation,
            penetration_guard_enabled=not args.disable_penetration_guard,
          )
          if staged_monitor is not None:
            staged_monitor.update(simulation, task_state=outcome.state, return_snapshot=False)

        clipped_values += stepper.advance(
          action,
          max_sim_seconds=args.max_sim_seconds,
          after_step=after_step,
          should_stop=lambda: outcome.success and not staged_version,
        )
        control_tick = stepper.control_tick
        action_steps += 1
        if staged_log is not None:
          staged_log.record(control_tick)
        if recorder is not None:
          recorder.capture(control_tick, outcome.stage())

      while simulation.data.time < args.max_sim_seconds and (staged_version or not outcome.success):
        if args.max_requests and requests >= args.max_requests:
          break
        images = {
          name: renderer.capture(simulation.data, camera)["rgb"]
          for name, camera in cameras.items()
        }
        state = right_joint_state(simulation, joint_names)
        request_started = time.monotonic()
        response = client.infer(
          {
            **pi05_image_payload(images, observation),
            "state": state,
            "prompt": deployment["instruction"],
          }
        )
        requests += 1
        try:
          actions = validated_actions(
            response, horizon=horizon, action_dim=deployment["action_dim"]
          )
        except RuntimeError as error:
          if not recover_nonfinite_action(simulation, error):
            raise
          row = {
            "request": requests,
            "sim_time": float(simulation.data.time),
            "stage": outcome.stage(),
            "round_trip_s": time.monotonic() - request_started,
            "server_timing": response.get("server_timing", {}),
            "state": state.tolist(),
            "action_rejected": "nonfinite_action",
            "recovery": "hold_last_valid_command_and_replan",
          }
          log.write(json.dumps(row, ensure_ascii=False) + "\n")
          log.flush()
          advance_held_or_applied_tick()
          simulation._evaluation_action_rejected = False
          continue
        row = {
          "request": requests,
          "sim_time": float(simulation.data.time),
          "stage": outcome.stage(),
          "round_trip_s": time.monotonic() - request_started,
          "server_timing": response.get("server_timing", {}),
          "state": state.tolist(),
          "predicted_action_min": np.min(actions, axis=0).tolist(),
          "predicted_action_max": np.max(actions, axis=0).tolist(),
        }
        log.write(json.dumps(row, ensure_ascii=False) + "\n")
        log.flush()
        if requests == 1 and args.save_first_request:
          np.savez_compressed(
            output / "first_request.npz",
            **{f"{name}_image": image for name, image in images.items()},
            state=state,
            predicted_actions=actions,
          )
        print(
          f"[policy] request={requests} sim_t={simulation.data.time:.3f}s "
          f"round_trip={row['round_trip_s']:.2f}s stage={outcome.stage()}",
          flush=True,
        )

        for index in range(args.execute_steps):
          if simulation.data.time >= args.max_sim_seconds or (outcome.success and not staged_version):
            break
          advance_held_or_applied_tick(actions[index])

      report["status"] = "success" if outcome.success else "task_not_completed"
  except Exception as error:
    error_text = f"{type(error).__name__}: {error}"
    report.update(status="error", error=error_text)
    raise
  finally:
    if client is not None:
      try:
        client._ws.close()
      except Exception:
        pass
    guard_report = {}
    if simulation is not None:
      guard_report = evaluation_guard_report(
        simulation, status=report.get("status"), error=error_text
      )
      report.update(guard_report)
      report["formal_metrics_valid"] = bool(
        report["formal_metrics_valid"] and validity["formal_metrics_valid"]
      )
      sim_seconds = float(simulation.data.time)
      report["sim_seconds"] = sim_seconds if math.isfinite(sim_seconds) else None
      report["final_plug_pose"] = (
        simulation.object_pose("usb_plug").tolist()
        if guard_report["physics_valid"]
        else None
      )
    else:
      report.update(
        guard_events=[],
        stop_reason=("fatal_error" if report.get("status") == "error" else None),
        physics_valid=None,
        formal_metrics_valid=False,
      )
    if outcome is not None:
      report["evaluation"] = outcome.report()
      report["task_success"] = bool(outcome.success)
    report["penetration_diagnostics"] = {
      "guard_enabled": not args.disable_penetration_guard,
      "threshold_m": PENETRATION_GUARD_THRESHOLD_M,
      "maximum_socket_penetration_m": (
        float(outcome.maximum_socket_penetration_m) if outcome is not None else 0.0
      ),
      "first_limit_exceeded_sim_time_s": first_penetration_limit_exceeded_s,
    }
    report["stats"] = {
      "requests": requests,
      "action_steps": action_steps,
      "clipped_action_values": clipped_values,
    }
    report["wall_seconds"] = time.monotonic() - started
    report["deployment_integrity_after"] = inspect_trial_integrity(
      deployment_path,
      deployment,
      getattr(args, "expected_deployment_manifest_sha256", None),
    )
    if report["deployment_integrity_after"]["status"] == "mismatch":
      report.update(
        status="error",
        stop_reason="fatal_error",
        formal_metrics_valid=False,
        error="trial deployment integrity changed during rollout",
      )
    if recorder is not None:
      try:
        if report.get("physics_valid", True):
          recorder.capture(control_tick, outcome.stage(), force=True)
      except Exception as record_error:
        _mark_record_error(report, record_error)
      try:
        video = recorder.finish(
          status=report.get("status", "error"),
          evaluation=report.get("evaluation"),
          error=error_text or report.get("record_error"),
        )
        video.update(
          seed=args.seed,
          success=bool(report.get("evaluation", {}).get("success", False)),
          task_success=report["task_success"],
          inference_requests=requests,
          simulation_time_s=report.get("sim_seconds"),
          wall_time_s=report["wall_seconds"],
          model_family=deployment["model_family"],
          checkpoint_path=deployment["checkpoint_path"],
          checkpoint_sha256=deployment["checkpoint_sha256"],
          prediction_horizon=report.get("prediction_horizon"),
          model_action_dim=deployment["model_action_dim"],
          action_dim=deployment["action_dim"],
          execute_steps=args.execute_steps,
          control_hz=CONTROL_HZ,
          replan_period_s=args.execute_steps / CONTROL_HZ,
          guard_events=report["guard_events"],
          stop_reason=report["stop_reason"],
          physics_valid=report["physics_valid"],
          formal_metrics_valid=report["formal_metrics_valid"],
          diagnostic_only=report["diagnostic_only"],
          diagnostic_reasons=report["diagnostic_reasons"],
          action_interpolation=interpolation,
          controller=report["controller"],
          controller_contract=report["controller_contract"],
          guard_mode=report["guard_mode"],
          record_error=report.get("record_error"),
          penetration_guard_enabled=not args.disable_penetration_guard,
          penetration_guard_threshold_m=PENETRATION_GUARD_THRESHOLD_M,
          maximum_socket_penetration_m=(
            report["penetration_diagnostics"]["maximum_socket_penetration_m"]
          ),
          first_penetration_limit_exceeded_sim_time_s=(
            first_penetration_limit_exceeded_s
          ),
        )
        (output / "review.json").write_text(
          json.dumps(video, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        report["video"] = video
      except Exception as record_error:
        _mark_record_error(report, record_error)
    if staged_log is not None:
      staged_log.finish(report)
    (output / "summary.json").write_text(
      json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
      f"[report] {output / 'summary.json'} status={report.get('status')} "
      f"requests={requests} action_steps={action_steps}",
      flush=True,
    )
  return report


def main() -> None:
  run(parse_args())


if __name__ == "__main__":
  main()
