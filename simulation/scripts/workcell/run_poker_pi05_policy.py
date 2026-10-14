#!/usr/bin/env python3
"""Run a manifest-bound pi0.5 joint policy in the poker-draw simulation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from kaihand_tactile_env.shared.config import CameraConfig, default_model_path
from kaihand_tactile_env.shared.evaluation_guards import (
  configure_evaluation_guards,
  ensure_evaluation_physics,
  evaluation_guard_report,
  mark_physics_invalid,
  record_guard_event,
  recover_nonfinite_action,
  task_guard,
)
from kaihand_tactile_env.shared.physics_integrity import PhysicsIntegrityError
from kaihand_tactile_env.shared.pi05_deployment_integrity import (
  POKER_DEPLOYMENT_SCHEMA_V1,
  POKER_DEPLOYMENT_SCHEMA_V2,
  deployment_diagnostic_reasons,
  inspect_shared_pi05_runtime_integrity,
  validate_poker_deployment_semantics,
)
from kaihand_tactile_env.shared.policy_cameras import (
  image_shape_hwc,
  pi05_image_payload,
  policy_camera_names,
)
from kaihand_tactile_env.shared.rendering import WorkcellRenderer
from kaihand_tactile_env.tasks.poker_draw.mid_full import middle_force_simulation
from kaihand_tactile_env.tasks.poker_draw.policy_control import PokerPolicyController
from kaihand_tactile_env.tasks.poker_draw.randomization import reset_randomized_card
from kaihand_tactile_env.tasks.poker_draw.review_metrics import PokerReviewMetrics
from kaihand_tactile_env.tasks.poker_draw.rollout_eval import (
  PokerOutcomeMonitor,
  observe_simulation,
)
from run_usb_pi05_policy import (
  apply_action,
  current_controller_target,
  joint_limits,
  linear_substep_action,
  right_joint_state,
  validated_actions,
)

CONTROL_HZ = 30
CONTROLLER = "direct-30hz-pi05-absolute-joint-v1"
LINEAR_CONTROLLER = "pi05-absolute-joint-physics-interpolation-v2"
ROOT = Path(__file__).resolve().parents[2]
PENETRATION_GUARD_THRESHOLD_M = 0.0006


def parse_args(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--server", default="ws://127.0.0.1:18784")
  parser.add_argument("--deployment-manifest", type=Path, required=True)
  parser.add_argument("--expected-deployment-manifest-sha256")
  parser.add_argument(
    "--action-interpolation", choices=("linear", "zoh"),
    help="Defaults to linear for checkpoint-bound v2 and historical hold for v1",
  )
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--xy-jitter-mm", type=float, default=4.0)
  parser.add_argument("--yaw-jitter-deg", type=float, default=0.5)
  parser.add_argument("--execute-steps", type=int, default=8)
  parser.add_argument("--staged-evaluation-version", choices=("staged_v1",), default=None)
  parser.add_argument("--max-requests", type=int, default=0)
  parser.add_argument("--max-sim-seconds", type=float, default=90.0)
  parser.add_argument(
    "--guard-mode", choices=("observe", "strict"), default="observe"
  )
  parser.add_argument("--success-hold-seconds", type=float, default=0.10)
  parser.add_argument(
    "--full-duration-evaluation", action="store_true",
    help="Diagnostic rollout: continue after card/table penetration or a dropped card; only nonfinite states abort",
  )
  parser.add_argument(
    "--disable-penetration-guard",
    action="store_true",
    help=(
      "Diagnostic-only: continue after supported card/table penetration exceeds "
      "0.6 mm; nonfinite-state and fallen-card guards remain enabled"
    ),
  )
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--record", action=argparse.BooleanOptionalAction, default=True)
  parser.add_argument(
    "--save-raw", action=argparse.BooleanOptionalAction, default=False,
    help="Save a synchronized Raw HDF5 episode and five-finger force plot",
  )
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
    "--show-model-wrist-in-review",
    action=argparse.BooleanOptionalAction,
    default=True,
  )
  parser.add_argument(
    "--save-first-request", action=argparse.BooleanOptionalAction, default=True
  )
  args = parser.parse_args(argv)
  if args.full_duration_evaluation:
    args.disable_penetration_guard = True
  if not np.isfinite(args.success_hold_seconds) or args.success_hold_seconds <= 0:
    parser.error("success-hold-seconds must be positive and finite")
  if args.seed < 0 or args.max_requests < 0 or args.execute_steps <= 0:
    parser.error("seed/requests must be nonnegative; execute-steps must be positive")
  for key in ("xy_jitter_mm", "yaw_jitter_deg", "max_sim_seconds"):
    value = getattr(args, key)
    if not np.isfinite(value) or value < 0 or (
      key == "max_sim_seconds" and value == 0
    ):
      parser.error(f"{key} must be finite and nonnegative")
  if args.xy_jitter_mm > 5.0 or args.yaw_jitter_deg > 1.0:
    parser.error("poker randomization is limited to 5 mm XY and 1 degree yaw")
  if args.show_model_wrist_in_review and args.review_second_camera == "right_wrist":
    parser.error("review second camera duplicates the model wrist view")
  if min(
    args.review_width,
    args.review_height,
    args.review_render_width,
    args.review_render_height,
  ) <= 0:
    parser.error("review dimensions must be positive")
  if args.review_width % 2 or args.review_height % 2:
    parser.error("review output dimensions must be even")
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
  if deployment["schema"] not in {
    POKER_DEPLOYMENT_SCHEMA_V1, POKER_DEPLOYMENT_SCHEMA_V2,
  }:
    raise RuntimeError("unsupported deployment manifest schema")
  if deployment["task"] != "poker-draw" or deployment["model_family"] != "pi0.5":
    raise RuntimeError("runner requires a poker-draw pi0.5 deployment")
  validate_poker_deployment_semantics(deployment)
  return resolved, deployment


def validate_server(metadata: dict, deployment: dict, execute_steps: int) -> int:
  expected = {
    "evaluation_task": "poker-draw",
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
  if deployment.get("schema") == POKER_DEPLOYMENT_SCHEMA_V2:
    expected.update({
      key: deployment[key] for key in (
        "model_config_name", "normalizer_asset_id", "required_action_interpolation",
        "diagnostic_only", "formal_metrics_valid",
      )
    })
  mismatches = {
    key: {"expected": value, "actual": metadata.get(key)}
    for key, value in expected.items()
    if metadata.get(key) != value
  }
  if mismatches:
    raise RuntimeError(f"server deployment identity mismatch: {mismatches}")
  horizon = int(metadata["prediction_horizon"])
  if not 1 <= execute_steps <= horizon:
    raise RuntimeError(
      f"execute_steps must be in [1, {horizon}], got {execute_steps}"
    )
  if int(metadata["control_hz"]) != CONTROL_HZ:
    raise RuntimeError(f"runner requires {CONTROL_HZ} Hz policy control")
  observation = metadata["observation_contract"]
  policy_camera_names(observation)
  image_shape_hwc(observation)
  if observation.get("tactile_sent_to_model") is not False:
    raise RuntimeError("this pi0.5 checkpoint must not receive tactile input")
  return horizon


def outcome_stage(monitor: PokerOutcomeMonitor) -> str:
  if monitor.success:
    return "success"
  if monitor.lifted:
    return "lifted"
  if monitor.edge_reached:
    return "edge_reached"
  if monitor.contacted:
    return "contacted"
  return "awaiting_contact"


def guard_poker_state(
  observer: PokerPolicyController, *, penetration_guard_enabled: bool = True,
  fallen_card_guard_enabled: bool = True,
) -> float:
  simulation = observer.sim
  ensure_evaluation_physics(simulation)
  card_height = float(simulation.object_pose("card")[2])
  task_guard(
    simulation,
    fallen_card_guard_enabled and card_height < 0.5,
    "card_drop",
    "card fell below work surface",
    measured=card_height,
    threshold=0.5,
    recovery="continue_recording_without_state_repair",
  )
  supported_penetration_m = (
    max(0.0, -float(observer._card_table_clearance()))
    if observer._supported()
    else 0.0
  )
  task_guard(
    simulation,
    penetration_guard_enabled
    and supported_penetration_m > PENETRATION_GUARD_THRESHOLD_M,
    "supported_table_penetration",
    "card penetrated supported tabletop beyond 0.6 mm",
    measured=supported_penetration_m,
    threshold=PENETRATION_GUARD_THRESHOLD_M,
    recovery="continue_recording_without_state_repair",
  )
  return supported_penetration_m


def run(args) -> dict:
  from openpi_client import websocket_client_policy

  output = args.output_dir.expanduser().resolve()
  deployment_path, deployment = load_deployment_manifest(args.deployment_manifest)
  interpolation = args.action_interpolation or (
    "linear" if deployment["schema"] == POKER_DEPLOYMENT_SCHEMA_V2 else "zoh"
  )
  if deployment["schema"] == POKER_DEPLOYMENT_SCHEMA_V2 and interpolation != deployment["required_action_interpolation"]:
    raise RuntimeError("checkpoint-bound Poker evaluation requires linear interpolation")
  diagnostic_reasons = deployment_diagnostic_reasons(
    deployment, action_interpolation=interpolation,
  )
  if args.disable_penetration_guard:
    diagnostic_reasons.append("penetration_guard_disabled")
  if args.guard_mode == "observe":
    diagnostic_reasons.append("guard_mode_observe")
  validity = {"diagnostic_only": bool(diagnostic_reasons), "formal_metrics_valid": not diagnostic_reasons}
  output.mkdir(parents=True, exist_ok=False)
  report = {
    "task": "poker-draw",
    "controller": LINEAR_CONTROLLER if interpolation == "linear" else CONTROLLER,
    "action_interpolation": interpolation,
    "required_action_interpolation": deployment.get("required_action_interpolation"),
    "diagnostic_reasons": diagnostic_reasons,
    "server": args.server,
    "seed": args.seed,
    "instruction": deployment["instruction"],
    "deployment_manifest": str(deployment_path),
    "deployment_id": deployment["deployment_id"],
    "checkpoint_path": deployment["checkpoint_path"],
    "checkpoint_sha256": deployment["checkpoint_sha256"],
    "checkpoint_step": deployment["checkpoint_step"],
    "model_family": deployment["model_family"],
    "model_action_dim": deployment["model_action_dim"],
    "action_dim": deployment["action_dim"],
    "observation_contract": deployment["observation_contract"],
    "action_representation": deployment["action_representation"],
    "execute_steps": args.execute_steps,
    "max_sim_seconds": args.max_sim_seconds,
    "success_hold_seconds": args.success_hold_seconds,
    "full_duration_evaluation": args.full_duration_evaluation,
    "control_hz": CONTROL_HZ,
    "replan_period_s": args.execute_steps / CONTROL_HZ,
    "guard_mode": args.guard_mode,
    "guard_events": [],
    "stop_reason": None,
    "physics_valid": False,
    **validity,
    "task_success": False,
    "penetration_guard_enabled": not args.disable_penetration_guard,
    "penetration_guard_threshold_m": PENETRATION_GUARD_THRESHOLD_M,
    "expert_used": False,
    "low_level_tactile_feedback": False,
    "tactile_sent_to_model": False,
    "render_shadows": False,
  }
  started = time.monotonic()
  client = simulation = recorder = observer = monitor = review_metrics = raw_capture = None
  staged_monitor = staged_log = None
  staged_version = getattr(args, "staged_evaluation_version", None)
  report["full_duration_evaluation"] = bool(staged_version or args.full_duration_evaluation)
  requests = action_steps = clipped_values = control_tick = 0
  maximum_supported_table_penetration_m = 0.0
  first_penetration_limit_exceeded_s = None
  error_text = None
  artifact_failure = None
  try:
    if deployment["schema"] == POKER_DEPLOYMENT_SCHEMA_V2:
      integrity = inspect_shared_pi05_runtime_integrity(
        deployment_path, deployment, evaluation_root=ROOT,
        expected_manifest_sha256=args.expected_deployment_manifest_sha256,
      )
      report["deployment_integrity"] = integrity
      if integrity["status"] != "verified":
        raise RuntimeError(f"child deployment integrity verification failed: {json.dumps(integrity, ensure_ascii=False, sort_keys=True)}")
    client = websocket_client_policy.WebsocketClientPolicy(args.server)
    metadata = client.get_server_metadata()
    horizon = validate_server(metadata, deployment, args.execute_steps)
    report["server_metadata"] = metadata
    report["prediction_horizon"] = horizon
    print(
      f"[policy] connected step={deployment['checkpoint_step']} H={horizon} "
      f"execute_steps={args.execute_steps} "
      f"diagnostic_only={args.disable_penetration_guard}",
      flush=True,
    )

    with middle_force_simulation(default_model_path("poker-draw")) as (
      simulation,
      contact,
    ):
      configure_evaluation_guards(simulation, args.guard_mode)
      report["contact_model"] = contact
      report["initial_card_randomization"] = reset_randomized_card(
        simulation,
        seed=args.seed,
        xy_jitter_m=args.xy_jitter_mm / 1000.0,
        yaw_jitter_rad=float(np.deg2rad(args.yaw_jitter_deg)),
      )
      ensure_evaluation_physics(simulation)
      report["initial_card_pose"] = simulation.object_pose("card").tolist()
      monitor = PokerOutcomeMonitor(
        required_hold_seconds=args.success_hold_seconds,
        require_clearance_during_hold=args.success_hold_seconds > 0.10,
      )
      # This object is used only for read-only contact/geometry extraction. Its
      # contact-conditioned command hooks are intentionally never called.
      observer = PokerPolicyController(
        simulation,
        penetration_guard_enabled=not args.disable_penetration_guard,
      )
      review_metrics = PokerReviewMetrics(
        report["initial_card_pose"][0], outcome_monitor=monitor
      )
      review_metrics.update(simulation)
      if staged_version:
        from kaihand_tactile_env.shared.staged_evaluation import StagedEvaluationMonitor
        from kaihand_tactile_env.shared.staged_recording import StagedMetricView, StagedStateLog

        staged_monitor = StagedEvaluationMonitor("poker-draw", simulation, version=staged_version)
        staged_monitor.update(simulation)
        staged_log = StagedStateLog(output, staged_monitor, identity=report)
        staged_log.record(0)
        video_metrics = StagedMetricView(review_metrics, staged_monitor)
      else:
        video_metrics = review_metrics
      joint_names = list(deployment["joint_names"])
      lower, upper = joint_limits(simulation, joint_names)
      last_controller_target = (
        current_controller_target(simulation, joint_names)
        if interpolation == "linear" else None
      )
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
      if args.save_raw:
        from poker_pi05_rollout_artifacts import PokerPi05RawCapture

        raw_capture = PokerPi05RawCapture(
          simulation, output, cameras,
          metadata={
            "model_family": "pi0.5",
            "deployment_id": deployment["deployment_id"],
            "checkpoint_path": deployment["checkpoint_path"],
            "checkpoint_sha256": deployment["checkpoint_sha256"],
            "seed": args.seed,
            "execute_steps": args.execute_steps,
            "policy_control_hz": CONTROL_HZ,
            "initial_card_randomization": report["initial_card_randomization"],
          },
        )
      with WorkcellRenderer(
        simulation.model, tuple(cameras.values()), shadows=False
      ) as renderer, (output / "requests.jsonl").open("x", encoding="utf-8") as log:
        report["render_backend"] = renderer.backend_info
        if args.record:
          from kaihand_tactile_env.shared.policy_video import PokerPolicyVideo

          recorder = PokerPolicyVideo(
            simulation,
            output,
            fps=args.record_fps,
            width=args.review_width,
            height=args.review_height,
            render_width=args.review_render_width,
            render_height=args.review_render_height,
            second_camera=args.review_second_camera,
            include_model_wrist=(
              args.show_model_wrist_in_review and "right_wrist" in camera_names
            ),
            include_review_wrist=(
              "right_wrist" not in camera_names
              and args.review_second_camera != "right_wrist"
            ),
            comparison=True,
            reference_dataset=args.reference_dataset,
            reference_episode_index=args.reference_episode_index,
            metrics=video_metrics,
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
              "diagnostic_reasons": diagnostic_reasons,
              "action_interpolation": interpolation,
              "penetration_guard_enabled": not args.disable_penetration_guard,
              "penetration_guard_threshold_m": PENETRATION_GUARD_THRESHOLD_M,
              "observation_contract": deployment["observation_contract"],
              "action_representation": deployment["action_representation"],
              "initial_randomization": report["initial_card_randomization"],
              "contact_model": contact,
            },
          )
          recorder.capture(0, outcome_stage(monitor))

        while simulation.data.time < args.max_sim_seconds and (staged_version or not monitor.success):
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
          except Exception as error:
            if not recover_nonfinite_action(simulation, error):
              raise
            actions = None
          row = {
            "request": requests,
            "sim_time": float(simulation.data.time),
            "stage": outcome_stage(monitor),
            "round_trip_s": time.monotonic() - request_started,
            "server_timing": response.get("server_timing", {}),
            "state": state.tolist(),
            "predicted_action_min": None if actions is None else np.min(actions, axis=0).tolist(),
            "predicted_action_max": None if actions is None else np.max(actions, axis=0).tolist(),
            "predicted_actions": actions.tolist() if args.save_raw and actions is not None else None,
          }
          log.write(json.dumps(row, ensure_ascii=False) + "\n")
          log.flush()
          if requests == 1 and args.save_first_request and actions is not None:
            np.savez_compressed(
              output / "first_request.npz",
              **{f"{name}_image": image for name, image in images.items()},
              state=state,
              predicted_actions=actions,
            )
          print(
            f"[policy] request={requests} sim_t={simulation.data.time:.3f}s "
            f"round_trip={row['round_trip_s']:.2f}s stage={outcome_stage(monitor)}",
            flush=True,
          )

          action_count = 1 if actions is None else args.execute_steps
          for index in range(action_count):
            if simulation.data.time >= args.max_sim_seconds or (monitor.success and not staged_version):
              break
            action_rejected = actions is None
            if actions is not None:
              if interpolation == "linear":
                endpoint = np.clip(actions[index], lower, upper)
                clipped_values += int(np.count_nonzero(actions[index] != endpoint))
              else:
                clipped_values += apply_action(
                  simulation, joint_names, actions[index], lower, upper
                )
            control_tick += 1
            target_time = control_tick / CONTROL_HZ
            interval_start_time = float(simulation.data.time)
            interval_start_target = (
              last_controller_target.copy() if interpolation == "linear" else None
            )
            while (
              simulation.data.time < target_time - 1.0e-12
              and simulation.data.time < args.max_sim_seconds - 1.0e-12
            ):
              before_time = float(simulation.data.time)
              if actions is not None and interpolation == "linear":
                last_controller_target = linear_substep_action(
                  interval_start_target, endpoint,
                  interval_start_s=interval_start_time,
                  interval_end_s=min(target_time, args.max_sim_seconds),
                  current_time_s=before_time, timestep_s=simulation.timestep,
                )
                if apply_action(simulation, joint_names, last_controller_target, lower, upper):
                  raise RuntimeError("interpolation between clipped endpoints was clipped")
              simulation.step()
              ensure_evaluation_physics(simulation, before_time=before_time)
              review_metrics.update(simulation)
              observe_simulation(
                monitor, observer, report["initial_card_pose"][0]
              )
              if raw_capture is not None:
                raw_capture.observe(simulation, outcome_stage(monitor))
              supported_penetration_m = guard_poker_state(
                observer,
                penetration_guard_enabled=not args.disable_penetration_guard,
                fallen_card_guard_enabled=not args.full_duration_evaluation,
              )
              maximum_supported_table_penetration_m = max(
                maximum_supported_table_penetration_m, supported_penetration_m
              )
              if staged_monitor is not None:
                staged_monitor.update(simulation, return_snapshot=False)
              if (
                supported_penetration_m > PENETRATION_GUARD_THRESHOLD_M
                and first_penetration_limit_exceeded_s is None
              ):
                first_penetration_limit_exceeded_s = float(simulation.data.time)
              if monitor.success and not staged_version:
                break
            action_steps += 1
            if staged_log is not None:
              staged_log.record(control_tick)
            if recorder is not None:
              recorder.capture(control_tick, outcome_stage(monitor))
            if action_rejected:
              break

      report["status"] = "success" if monitor.success else "task_not_completed"
  except Exception as error:
    if simulation is not None and isinstance(error, PhysicsIntegrityError):
      mark_physics_invalid(simulation)
      record_guard_event(
        simulation,
        "physics_invalid",
        "poker physics integrity check failed",
        recovery="stop",
        details={"error": str(error), **error.details},
      )
    error_text = f"{type(error).__name__}: {error}"
    report.update(status="error", error=error_text)
    raise
  finally:
    if client is not None:
      try:
        client._ws.close()
      except Exception:
        pass
    if simulation is not None:
      guard_report = evaluation_guard_report(
        simulation, status=report.get("status"), error=error_text
      )
      report.update(guard_report)
      # Task guards report runtime validity; they cannot upgrade deployment
      # provenance or legacy/interpolation diagnostic restrictions.
      if diagnostic_reasons:
        report.update(diagnostic_only=True, formal_metrics_valid=False)
      sim_time = float(simulation.data.time)
      report["sim_seconds"] = sim_time if np.isfinite(sim_time) else None
      if guard_report["physics_valid"]:
        report["final_card_pose"] = simulation.object_pose("card").tolist()
        report["last_drive_state"] = simulation.drive_state()
      else:
        report["final_card_pose"] = None
        report["last_drive_state"] = None
      if (
        report.get("status") == "task_not_completed"
        and args.max_requests
        and requests >= args.max_requests
        and report["sim_seconds"] is not None
        and report["sim_seconds"] < args.max_sim_seconds
      ):
        report["stop_reason"] = "request_limit"
      if args.disable_penetration_guard:
        report.update(diagnostic_only=True, formal_metrics_valid=False)
    if observer is not None:
      report["terminal_metrics"] = (
        observer.terminal_metrics() if report.get("physics_valid") else None
      )
    if monitor is not None:
      report["evaluation"] = monitor.report()
    report["stats"] = {
      "requests": requests,
      "action_steps": action_steps,
      "clipped_action_values": clipped_values,
    }
    report["penetration_diagnostics"] = {
      "guard_enabled": not args.disable_penetration_guard,
      "threshold_m": PENETRATION_GUARD_THRESHOLD_M,
      "maximum_supported_table_penetration_m": (
        maximum_supported_table_penetration_m
      ),
      "first_limit_exceeded_sim_time_s": first_penetration_limit_exceeded_s,
    }
    report["wall_seconds"] = time.monotonic() - started
    report["task_success"] = bool(
      report.get("evaluation", {}).get("success", False)
    )
    if raw_capture is not None:
      try:
        if report.get("physics_valid") is False:
          report["raw"] = {
            "episode": str(raw_capture.path.relative_to(output)),
            "validated": False,
            "terminal_capture_skipped": "physics_invalid",
          }
        else:
          report["raw"] = raw_capture.finish(report, outcome_stage(monitor))
      except Exception as raw_error:
        report["raw_record_error"] = f"{type(raw_error).__name__}: {raw_error}"
        report.update(stop_reason="fatal_error", formal_metrics_valid=False)
        if error_text is None:
          report.update(status="error", error=report["raw_record_error"])
        artifact_failure = raw_error
      try:
        raw_capture.close_incomplete()
      except Exception as raw_close_error:
        report["raw_record_error"] = (
          f"{type(raw_close_error).__name__}: {raw_close_error}"
        )
        report.update(stop_reason="fatal_error", formal_metrics_valid=False)
        if error_text is None:
          report.update(status="error", error=report["raw_record_error"])
        artifact_failure = artifact_failure or raw_close_error
    if recorder is not None:
      try:
        if not getattr(simulation, "_evaluation_physics_invalid", False):
          recorder.capture(control_tick, outcome_stage(monitor), force=True)
        video = recorder.finish(
          status=report.get("status", "error"),
          evaluation=report.get("evaluation"),
          error=error_text or report.get("raw_record_error"),
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
          action_interpolation=interpolation,
          diagnostic_reasons=diagnostic_reasons,
          guard_mode=report.get("guard_mode"),
          guard_events=report.get("guard_events", []),
          stop_reason=report.get("stop_reason"),
          physics_valid=report.get("physics_valid"),
          diagnostic_only=report.get("diagnostic_only"),
          formal_metrics_valid=report.get("formal_metrics_valid"),
          penetration_guard_enabled=not args.disable_penetration_guard,
          penetration_guard_threshold_m=PENETRATION_GUARD_THRESHOLD_M,
          maximum_supported_table_penetration_m=(
            maximum_supported_table_penetration_m
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
        report["record_error"] = f"{type(record_error).__name__}: {record_error}"
        report.update(stop_reason="fatal_error", formal_metrics_valid=False)
        if error_text is None:
          report.update(status="error", error=report["record_error"])
        artifact_failure = artifact_failure or record_error
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
    if artifact_failure is not None and error_text is None:
      raise RuntimeError(report["error"]) from artifact_failure
  return report


def main() -> None:
  run(parse_args())


if __name__ == "__main__":
  main()
