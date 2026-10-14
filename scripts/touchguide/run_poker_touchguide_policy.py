#!/usr/bin/env python3
"""Run one 30-Hz poker rollout using the shared simulator and TouchGuide server."""

# Imports of simulator modules follow explicit source/EGL setup.
# ruff: noqa: PLC0415

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-root", type=Path, required=True)
    parser.add_argument("--server", default="ws://127.0.0.1:18786")
    parser.add_argument("--variant", choices=("pi05", "pi05+touchguidance"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--execute-steps", type=int, default=5)
    parser.add_argument("--max-sim-seconds", type=float, default=50.0)
    parser.add_argument("--success-hold-seconds", type=float, default=5.0)
    parser.add_argument("--xy-jitter-mm", type=float, default=4.0)
    parser.add_argument("--yaw-jitter-deg", type=float, default=0.5)
    parser.add_argument("--record-fps", type=int, choices=(5, 10), default=10)
    parser.add_argument("--video-mode", choices=("all", "head-only", "overhead-only"), default="all")
    parser.add_argument("--save-raw", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--record", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-policy-traces", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compare-base", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--full-duration-evaluation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reference-dataset", type=Path)
    parser.add_argument("--tactile-display-scale", type=float, default=0.05)
    args = parser.parse_args(argv)
    if not 1 <= args.execute_steps <= 30:
        parser.error("execute-steps must be in 1..30")
    if args.seed < 0:
        parser.error("seed must be nonnegative")
    for name in ("max_sim_seconds", "success_hold_seconds", "tactile_display_scale"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"{name} must be finite and positive")
    if not 0 <= args.xy_jitter_mm <= 5 or not 0 <= args.yaw_jitter_deg <= 1:
        parser.error("Use the original randomization bounds: at most 5 mm XY and 1 degree yaw")
    return args


def configure_simulator(root: Path) -> None:
    root = root.expanduser().resolve(strict=True)
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root / "scripts/workcell"))
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("KAIHAND_RENDER_BACKEND", "hardware")
    os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", str(root / "scripts/collect/nvidia_egl_vendor.json"))


def validate_metadata(metadata: dict, execute_steps: int = 5) -> None:
    expected = {
        "schema": "openpi_poker_touchguide_policy_v1",
        "task": "poker-draw",
        "control_hz": 30,
        "prediction_horizon": 30,
        "execute_steps": execute_steps,
        "model_action_dim": 32,
        "action_dim": 27,
        "model_input_cameras": ["head", "right_wrist"],
        "image_shape_hwc": [240, 320, 3],
        "tactile_shape": [5, 3, 7, 5],
        "tactile_channels": ["normal", "tangent0", "tangent1"],
        "finger_order": ["thumb", "index", "middle", "ring", "pinky"],
        "tactile_unit": "N",
        "tactile_target_geometry": "card_core_geom",
        "tactile_clipping": False,
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Server observation/action contract does not match this poker runner")
    if len(metadata.get("joint_names", [])) != 27 or Path(metadata["base_source"]).name != "20000":
        raise RuntimeError("Use the fixed 20000-step right-side pi0.5 base")


def run(args) -> dict:
    configure_simulator(args.sim_root)
    from kaihand_tactile_env.shared.config import FINGERTIP_LINK_NAMES
    from kaihand_tactile_env.shared.config import CameraConfig
    from kaihand_tactile_env.shared.config import default_model_path
    from kaihand_tactile_env.shared.contact_tactile import SolverDistributedTactileProvider
    from kaihand_tactile_env.shared.policy_video import PokerPolicyVideo
    from kaihand_tactile_env.shared.rendering import WorkcellRenderer
    from kaihand_tactile_env.tasks.poker_draw.mid_full import middle_force_simulation
    from kaihand_tactile_env.tasks.poker_draw.policy_control import PokerPolicyController
    from kaihand_tactile_env.tasks.poker_draw.randomization import reset_randomized_card
    from kaihand_tactile_env.tasks.poker_draw.review_metrics import PokerReviewMetrics
    from kaihand_tactile_env.tasks.poker_draw.rollout_eval import PokerOutcomeMonitor
    from kaihand_tactile_env.tasks.poker_draw.rollout_eval import observe_simulation
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    from poker_pi05_rollout_artifacts import PokerPi05RawCapture
    from run_poker_pi05_policy import guard_poker_state
    from run_poker_pi05_policy import outcome_stage
    from run_usb_pi05_policy import apply_action
    from run_usb_pi05_policy import joint_limits
    from run_usb_pi05_policy import right_joint_state
    from run_usb_pi05_policy import validated_actions
    from sim_recording import SynchronizedVideos

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "task": "poker-draw",
        "model_family": args.variant,
        "seed": args.seed,
        "server": args.server,
        "control_hz": 30,
        "prediction_horizon": 30,
        "execute_steps": args.execute_steps,
        "replan_period_s": args.execute_steps / 30,
        "max_sim_seconds": args.max_sim_seconds,
        "success_hold_seconds": args.success_hold_seconds,
        "video_mode": args.video_mode,
        "full_duration_evaluation": args.full_duration_evaluation,
        "diagnostic_only": args.full_duration_evaluation,
        "formal_metrics_valid": not args.full_duration_evaluation,
        "controller": "direct-30hz-pi05-absolute-joint-v1",
        "low_level_tactile_feedback": False,
        "tactile_sent_to_cpm": args.variant == "pi05+touchguidance",
        "rtc_enabled": False,
        "observation_contract": {"cameras": ["head", "right_wrist"], "image_shape_hwc": [240, 320, 3]},
        "status": "error",
    }
    started = time.monotonic()
    client = None
    requests = ticks = clipped = 0
    try:
        client = WebsocketClientPolicy(args.server)
        metadata = client.get_server_metadata()
        validate_metadata(metadata, args.execute_steps)
        report["server_metadata"] = metadata
        report["base_checkpoint_path"] = metadata["base_source"]
        report["cpm_checkpoint"] = metadata["cpm_checkpoint"]
        with middle_force_simulation(default_model_path("poker-draw")) as (simulation, contact), ExitStack() as stack:
            report["contact_model"] = contact
            report["initial_card_randomization"] = reset_randomized_card(
                simulation,
                seed=args.seed,
                xy_jitter_m=args.xy_jitter_mm / 1000,
                yaw_jitter_rad=float(np.deg2rad(args.yaw_jitter_deg)),
            )
            report["initial_card_pose"] = simulation.object_pose("card").tolist()
            monitor = PokerOutcomeMonitor(
                required_hold_seconds=args.success_hold_seconds,
                require_clearance_during_hold=args.success_hold_seconds > 0.10,
            )
            observer = PokerPolicyController(simulation, penetration_guard_enabled=not args.full_duration_evaluation)
            metrics = PokerReviewMetrics(report["initial_card_pose"][0], outcome_monitor=monitor)
            metrics.update(simulation)
            joints = metadata["joint_names"]
            lower, upper = joint_limits(simulation, joints)
            provider = SolverDistributedTactileProvider(
                simulation.model, target_geom_names=("card_core_geom",), link_names=tuple(FINGERTIP_LINK_NAMES[5:])
            )
            cameras = {
                name: CameraConfig(name, width=320, height=240, rgb=True, depth=False, segmentation=False)
                for name in ("head", "right_wrist", "overhead")
            }
            raw = review = videos = None
            maximum_penetration = 0.0
            try:
                renderer = stack.enter_context(
                    WorkcellRenderer(simulation.model, tuple(cameras.values()), shadows=False)
                )
                report["render_backend"] = renderer.backend_info
                log = stack.enter_context((output / "requests.jsonl").open("x"))
                if args.save_policy_traces:
                    (output / "policy_requests").mkdir()
                if args.save_raw:
                    raw = PokerPi05RawCapture(
                        simulation,
                        output,
                        cameras,
                        metadata={
                            "model_family": args.variant,
                            "seed": args.seed,
                            "base_checkpoint_path": metadata["base_source"],
                            "cpm_checkpoint_path": metadata["cpm_checkpoint"],
                            "execute_steps": args.execute_steps,
                            "policy_control_hz": 30,
                            "initial_card_randomization": report["initial_card_randomization"],
                        },
                    )
                if args.record and args.video_mode == "all":
                    review = PokerPolicyVideo(
                        simulation,
                        output / "review",
                        fps=args.record_fps,
                        second_camera="overhead",
                        include_model_wrist=True,
                        comparison=args.reference_dataset is not None,
                        reference_dataset=args.reference_dataset,
                        metrics=metrics,
                        metadata={**report, "model_family": args.variant},
                    )
                    review.capture(0, outcome_stage(monitor))
                if args.record:
                    camera = args.video_mode.removesuffix("-only")
                    video_cameras = cameras if args.video_mode == "all" else {camera: cameras[camera]}
                    videos = SynchronizedVideos(
                        output / "videos",
                        renderer,
                        video_cameras,
                        provider,
                        fps=args.record_fps,
                        scale=args.tactile_display_scale,
                        include_tactile=args.video_mode == "all",
                    )
                    videos.capture(simulation, 0)
                while simulation.data.time < args.max_sim_seconds and not monitor.success:
                    timestamp = float(simulation.data.time)
                    images = {
                        name: renderer.capture(simulation.data, cameras[name])["rgb"]
                        for name in ("head", "right_wrist")
                    }
                    state = right_joint_state(simulation, joints)
                    tactile = provider.read(simulation.data)
                    maps = np.stack(
                        [
                            tactile.normal_taxel_force_n,
                            tactile.tangent_taxel_force_n[..., 0],
                            tactile.tangent_taxel_force_n[..., 1],
                        ],
                        axis=1,
                    ).astype(np.float32)
                    if float(simulation.data.time) != timestamp:
                        raise RuntimeError("Simulation advanced while capturing synchronized observations")
                    request_started = time.monotonic()
                    response = client.infer(
                        {
                            "head_image": images["head"],
                            "right_wrist_image": images["right_wrist"],
                            "state": state,
                            "tactile_map_raw": maps,
                            "prompt": metadata["instruction"],
                            "variant": args.variant,
                            "episode_seed": args.seed,
                            "request_index": requests,
                            "execute_steps": args.execute_steps,
                            "return_trace": args.save_policy_traces,
                            "compare_base": args.compare_base,
                        }
                    )
                    actions = validated_actions(response, horizon=30, action_dim=27)
                    diagnostics = response["diagnostics"]
                    if diagnostics["variant"] != args.variant:
                        raise RuntimeError("Server used another policy variant")
                    requests += 1
                    trace_path = None
                    if args.save_policy_traces:
                        trace_path = f"policy_requests/request_{requests:06d}.npz"
                        arrays = {
                            "head_image": images["head"],
                            "right_wrist_image": images["right_wrist"],
                            "state_raw": state,
                            "tactile_map_raw": maps,
                            "actions": actions,
                            "sim_time_s": np.asarray(timestamp),
                            **response["audit"],
                        }
                        if "base_actions" in response:
                            arrays["base_actions"] = response["base_actions"]
                        np.savez_compressed(output / trace_path, **arrays)
                    row = {
                        "request": requests,
                        "predicted_action_steps": 30,
                        "planned_execute_steps": args.execute_steps,
                        "sim_time": timestamp,
                        "stage": outcome_stage(monitor),
                        "round_trip_s": time.monotonic() - request_started,
                        "state": state.tolist(),
                        "server_timing": response.get("server_timing", {}),
                        "diagnostics": diagnostics,
                        "right_normal_force_n": maps[:, 0].sum(axis=(1, 2)).tolist(),
                        "right_tangent0_force_n": maps[:, 1].sum(axis=(1, 2)).tolist(),
                        "right_tangent1_force_n": maps[:, 2].sum(axis=(1, 2)).tolist(),
                        "audit": trace_path,
                    }
                    if requests == 1:
                        np.savez_compressed(
                            output / "first_request.npz",
                            **images,
                            state=state,
                            tactile_map_raw=maps,
                            predicted_actions=actions,
                        )
                    print(
                        f"[policy] {args.variant} seed={args.seed} request={requests} sim_t={timestamp:.3f}s "
                        f"arm_delta_rms={diagnostics.get('arm_target_delta_rms_rad')} "
                        f"hand_delta_rms={diagnostics.get('hand_target_delta_rms_rad')}",
                        flush=True,
                    )
                    ticks_before = ticks
                    for index in range(args.execute_steps):
                        if simulation.data.time >= args.max_sim_seconds or monitor.success:
                            break
                        clipped += apply_action(simulation, joints, actions[index], lower, upper)
                        ticks += 1
                        target_time = ticks / 30
                        while simulation.data.time < min(target_time, args.max_sim_seconds) - 1e-12:
                            simulation.step()
                            metrics.update(simulation)
                            observe_simulation(monitor, observer, report["initial_card_pose"][0])
                            if raw is not None:
                                raw.observe(simulation, outcome_stage(monitor))
                            penetration = guard_poker_state(
                                observer,
                                penetration_guard_enabled=not args.full_duration_evaluation,
                                fallen_card_guard_enabled=not args.full_duration_evaluation,
                            )
                            maximum_penetration = max(maximum_penetration, penetration)
                            if monitor.success:
                                break
                        if review is not None:
                            review.capture(ticks, outcome_stage(monitor))
                        if videos is not None:
                            videos.capture(simulation, ticks)
                    row["executed_action_steps"] = ticks - ticks_before
                    log.write(json.dumps(row) + "\n")
                    log.flush()
                report["status"] = "success" if monitor.success else "task_not_completed"
            except Exception as error:
                report.update(status="error", error=f"{type(error).__name__}: {error}")
            finally:
                report.update(
                    sim_seconds=float(simulation.data.time),
                    final_card_pose=simulation.object_pose("card").tolist(),
                    evaluation=monitor.report(),
                    terminal_metrics=observer.terminal_metrics(),
                    maximum_supported_table_penetration_m=maximum_penetration,
                )
                report["stats"] = {"requests": requests, "action_steps": ticks, "clipped_action_values": clipped}
                recording_errors = []
                if raw is not None:
                    try:
                        report["raw"] = raw.finish(report, outcome_stage(monitor))
                    except Exception as error:
                        recording_errors.append(f"raw: {error}")
                    finally:
                        raw.close_incomplete()
                if review is not None:
                    try:
                        review.capture(ticks, outcome_stage(monitor), force=True)
                        # The recorder copied the report at startup; refresh its initial error status.
                        review.metadata["status"] = report["status"]
                        report["review"] = review.finish(
                            status=report["status"], evaluation=report["evaluation"], error=report.get("error")
                        )
                    except Exception as error:
                        recording_errors.append(f"review: {error}")
                if videos is not None:
                    try:
                        videos.capture(simulation, ticks, force=True)
                        report["videos"] = videos.finish()
                    except Exception as error:
                        videos.abort()
                        recording_errors.append(f"videos: {error}")
                if recording_errors:
                    report.update(status="error", recording_errors=recording_errors)
    except Exception as error:
        report.update(status="error", error=f"{type(error).__name__}: {error}")
    finally:
        if client is not None:
            client._ws.close()  # noqa: SLF001 -- same existing client lifecycle as simulator runners
        report["wall_seconds"] = time.monotonic() - started
        (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"[report] {output / 'summary.json'} status={report['status']}", flush=True)
    if report["status"] == "error":
        raise RuntimeError(f"Rollout failed; inspect {output / 'summary.json'}")
    return report


if __name__ == "__main__":
    run(parse_args())
