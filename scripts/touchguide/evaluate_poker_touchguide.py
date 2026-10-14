#!/usr/bin/env python3
"""Launch one model server and paired, recorded poker evaluation subprocesses."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import time
import urllib.request

REPO = Path(__file__).resolve().parents[2]
VARIANTS = ("pi05", "pi05+touchguidance")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-root", type=Path, required=True)
    parser.add_argument("--model-python", type=Path, required=True)
    parser.add_argument("--sim-python", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--cpm-checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--tokenizer-model", type=Path, required=True)
    parser.add_argument("--reference-dataset", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-name", default=f"pi05_vs_touchguidance_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--num-trials", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--port", type=int, default=18786)
    parser.add_argument("--server-wait-seconds", type=float, default=600)
    parser.add_argument("--trial-wall-limit", type=float, default=3600)
    parser.add_argument("--execute-steps", type=int, default=5)
    parser.add_argument("--max-sim-seconds", type=float, default=50)
    parser.add_argument("--success-hold-seconds", type=float, default=5)
    parser.add_argument("--xy-jitter-mm", type=float, default=4)
    parser.add_argument("--yaw-jitter-deg", type=float, default=0.5)
    parser.add_argument("--record-fps", type=int, choices=(5, 10), default=10)
    parser.add_argument("--video-mode", choices=("all", "head-only", "overhead-only"), default="all")
    parser.add_argument("--guidance-scale", type=float, default=1)
    parser.add_argument("--guidance-steps", type=int, default=3)
    parser.add_argument("--tactile-display-scale", type=float, default=0.05)
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--save-raw", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--record", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-policy-traces", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compare-base", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--full-duration-evaluation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--resume", action="store_true", help="Keep validated trials and extend an existing run to num-trials"
    )
    args = parser.parse_args(argv)
    if not 1 <= args.execute_steps <= 30:
        parser.error("execute-steps must be in 1..30")
    if args.num_trials <= 0 or args.seed_start < 0 or not 1 <= args.port <= 65535:
        parser.error("Use positive trial count, nonnegative seed and port in 1..65535")
    if not args.run_name or Path(args.run_name).name != args.run_name or args.run_name in (".", ".."):
        parser.error("run-name must be one directory name")
    if len(args.variants) != len(set(args.variants)):
        parser.error("variants must be distinct")
    for name in (
        "server_wait_seconds",
        "trial_wall_limit",
        "max_sim_seconds",
        "success_hold_seconds",
        "tactile_display_scale",
    ):
        if not 0 < getattr(args, name) < float("inf"):
            parser.error(f"{name} must be finite and positive")
    if not 0 <= args.xy_jitter_mm <= 5 or not 0 <= args.yaw_jitter_deg <= 1:
        parser.error("Use at most 5 mm XY and 1 degree yaw randomization")
    if not 0 <= args.guidance_steps <= 9 or not 0 <= args.guidance_scale < float("inf"):
        parser.error("Use guidance-steps in 0..9 and finite nonnegative scale")
    for name in ("model_python", "sim_python", "cpm_checkpoint", "tokenizer_model"):
        if not getattr(args, name).expanduser().is_file():
            parser.error(f"Missing {name}: {getattr(args, name)}")
    for name in ("model_python", "sim_python"):
        if not os.access(getattr(args, name).expanduser(), os.X_OK):
            parser.error(f"Python is not executable: {getattr(args, name)}")
    for path in (
        args.base_checkpoint / "model.safetensors",
        args.base_checkpoint / "config.json",
        args.data_root / "config/cpm.json",
        args.sim_root / "scripts/workcell/run_poker_pi05_policy.py",
        args.sim_root / "scripts/collect/nvidia_egl_vendor.json",
    ):
        if not path.expanduser().is_file():
            parser.error(f"Missing inference dependency: {path}")
    if args.reference_dataset is not None and not (args.reference_dataset / "meta/info.json").is_file():
        parser.error("reference-dataset must be a valid LeRobot dataset")
    return args


def environment(args, *, model: bool) -> dict:
    env = dict(os.environ)
    paths = (
        [REPO / "src", REPO / "packages/openpi-client/src"]
        if model
        else [args.sim_root / "src", args.sim_root / "scripts/workcell", REPO / "packages/openpi-client/src"]
    )
    env.update(
        PYTHONPATH=os.pathsep.join(map(str, paths)),
        PYTHONDONTWRITEBYTECODE="1",
        CUDA_VISIBLE_DEVICES=args.gpu_id,
        JAX_PLATFORMS="cpu",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        OMP_NUM_THREADS="8" if model else "1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NO_PROXY="127.0.0.1,localhost",
        no_proxy="127.0.0.1,localhost",
    )
    if not model:
        env.update(
            MUJOCO_GL="egl",
            KAIHAND_RENDER_BACKEND="hardware",
            __EGL_VENDOR_LIBRARY_FILENAMES=str(args.sim_root / "scripts/collect/nvidia_egl_vendor.json"),
        )
        env.pop("LIBGL_ALWAYS_SOFTWARE", None)
    return env


def server_command(args, output: Path) -> list[str]:
    return [
        str(args.model_python),
        "-u",
        str(REPO / "scripts/touchguide/serve_poker_touchguide.py"),
        "--base-checkpoint",
        str(args.base_checkpoint),
        "--cpm-checkpoint",
        str(args.cpm_checkpoint),
        "--data-root",
        str(args.data_root),
        "--tokenizer-model",
        str(args.tokenizer_model),
        "--execute-steps",
        str(args.execute_steps),
        "--guidance-scale",
        str(args.guidance_scale),
        "--guidance-steps",
        str(args.guidance_steps),
        "--port",
        str(args.port),
        "--metadata-output",
        str(output / "server_metadata.json"),
    ]


def trial_command(args, output: Path, variant: str, seed: int) -> list[str]:
    command = [
        str(args.sim_python),
        "-u",
        str(REPO / "scripts/touchguide/run_poker_touchguide_policy.py"),
        "--sim-root",
        str(args.sim_root),
        "--server",
        f"ws://127.0.0.1:{args.port}",
        "--variant",
        variant,
        "--seed",
        str(seed),
        "--output-dir",
        str(output / variant / f"seed_{seed:03d}"),
        "--execute-steps",
        str(args.execute_steps),
        "--max-sim-seconds",
        str(args.max_sim_seconds),
        "--success-hold-seconds",
        str(args.success_hold_seconds),
        "--record-fps",
        str(args.record_fps),
        "--xy-jitter-mm",
        str(args.xy_jitter_mm),
        "--yaw-jitter-deg",
        str(args.yaw_jitter_deg),
        "--tactile-display-scale",
        str(args.tactile_display_scale),
        "--video-mode",
        args.video_mode,
    ]
    for name in ("save_raw", "record", "save_policy_traces", "compare_base", "full_duration_evaluation"):
        flag = name.replace("_", "-")
        command.append(f"--{flag}" if getattr(args, name) else f"--no-{flag}")
    if args.reference_dataset is not None:
        command += ["--reference-dataset", str(args.reference_dataset)]
    return command


def stop_process(process) -> None:
    """Stop only this launcher's own child process group, including video encoders."""
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def wait_for_server(process, args, log: Path) -> None:
    deadline = time.monotonic() + args.server_wait_seconds
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Model server exited; inspect {log}")
        try:
            with opener.open(f"http://127.0.0.1:{args.port}/healthz", timeout=1) as response:
                if response.status == 200 and response.read() == b"OK\n":
                    return
        except OSError:
            pass
        time.sleep(1)
    raise RuntimeError(f"Model server did not become ready; inspect {log}")


def validate_artifacts(args, trial: Path, summary: dict, variant: str, seed: int) -> None:
    expected = {
        "task": "poker-draw",
        "model_family": variant,
        "seed": seed,
        "prediction_horizon": 30,
        "execute_steps": args.execute_steps,
        "control_hz": 30,
        "success_hold_seconds": args.success_hold_seconds,
        "max_sim_seconds": args.max_sim_seconds,
    }
    if any(summary.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"Trial identity mismatch: {trial}")
    if summary.get("status") not in ("success", "task_not_completed"):
        raise RuntimeError(f"Incomplete trial: {trial}; error={summary.get('error')}")
    if summary["status"] == "task_not_completed" and summary["sim_seconds"] < args.max_sim_seconds - 0.003:
        raise RuntimeError(f"Trial stopped before the configured time limit: {trial}")
    required = ["summary.json", "requests.jsonl", "first_request.npz"]
    if args.save_raw:
        required += [
            "raw/episode.h5",
            "raw/episode.json",
            "raw/episode.result.json",
            "curves/right_hand_force_curves.png",
        ]
        if summary.get("raw", {}).get("validated") is not True:
            raise RuntimeError(f"Raw recording validation failed: {trial}")
    if args.record and args.video_mode == "all":
        required += [
            "review/review.mp4",
            "review/review.json",
            "review/frames.jsonl",
            "videos/head.mp4",
            "videos/right_wrist.mp4",
            "videos/overhead.mp4",
            "videos/tactile.mp4",
            "videos/metadata.json",
        ]
        if summary.get("videos", {}).get("completed") is not True:
            raise RuntimeError(f"Video recording did not finish: {trial}")
        review = summary.get("review", {})
        if review.get("completed") is not True or review.get("task_status") != summary["status"]:
            raise RuntimeError(f"Review recording did not finish with the reported task status: {trial}")
    if args.record and args.video_mode != "all":
        camera = args.video_mode.removesuffix("-only")
        required += [f"videos/{camera}.mp4", "videos/metadata.json", "videos/frames.jsonl"]
        videos = summary.get("videos", {})
        if videos.get("completed") is not True or videos.get("camera_names") != [camera]:
            raise RuntimeError(f"Single-camera video recording did not finish: {trial}")
        if videos.get("tactile_video_saved") is not False:
            raise RuntimeError(f"Unexpected tactile video in single-camera mode: {trial}")
        if sorted(path.name for path in trial.rglob("*.mp4")) != [f"{camera}.mp4"]:
            raise RuntimeError(f"Unexpected video files in single-camera mode: {trial}")
    if args.save_policy_traces:
        count = len(list((trial / "policy_requests").glob("request_*.npz")))
        if count != summary["stats"]["requests"]:
            raise RuntimeError(f"Missing policy input/trajectory records: {trial}")
    missing = [name for name in required if not (trial / name).is_file()]
    if missing:
        raise RuntimeError(f"Missing inference artifacts under {trial}: {missing}")


def result_row(output: Path, variant: str, seed: int, summary: dict) -> dict:
    return {
        "variant": variant,
        "seed": seed,
        "status": summary["status"],
        "success": summary["evaluation"]["success"],
        "sim_seconds": summary["sim_seconds"],
        "initial_card_pose": summary["initial_card_pose"],
        "stats": summary["stats"],
        "summary": str((output / variant / f"seed_{seed:03d}" / "summary.json").relative_to(output)),
    }


def write_results(args, output: Path, seeds: list[int], rows: list[dict]) -> None:
    ordered = sorted(rows, key=lambda row: (args.variants.index(row["variant"]), row["seed"]))
    aggregate = {
        "complete": len(rows) == len(seeds) * len(args.variants),
        "target_trials_per_variant": len(seeds),
        "seeds": seeds,
        "results": ordered,
        "variants": {
            family: {
                "completed": sum(row["variant"] == family for row in rows),
                "successes": sum(row["variant"] == family and row["success"] for row in rows),
            }
            for family in args.variants
        },
        "diagnostic_only": args.full_duration_evaluation,
        "success_definition": "Legacy simulator outcome monitor; evaluate the agreed three stages from raw recordings separately",
    }
    temporary = output / "results.json.tmp"
    temporary.write_text(json.dumps(aggregate, indent=2) + "\n")
    temporary.replace(output / "results.json")


def resume_trials(args, output: Path, seeds: list[int]) -> tuple[list[dict], list[tuple[str, int]]]:
    """Read-only planning; accept only completed artifacts under the same policy/physics."""
    protocol = json.loads((output / "protocol.json").read_text())
    for key in (
        "guidance_scale",
        "guidance_steps",
        "execute_steps",
        "max_sim_seconds",
        "success_hold_seconds",
        "xy_jitter_mm",
        "yaw_jitter_deg",
        "record_fps",
        "save_raw",
        "record",
        "save_policy_traces",
        "compare_base",
        "full_duration_evaluation",
        "variants",
    ):
        if protocol.get(key, 30 if key == "execute_steps" else None) != getattr(args, key):
            raise RuntimeError(f"Cannot resume with changed {key}: {protocol.get(key)!r} -> {getattr(args, key)!r}")
    for key in ("base_checkpoint", "cpm_checkpoint", "data_root", "sim_root", "tokenizer_model"):
        if Path(protocol[key]).expanduser().resolve() != getattr(args, key).expanduser().resolve():
            raise RuntimeError(f"Cannot resume with changed {key}")
    retained_mode = protocol.get("retained_video_mode", protocol.get("video_mode", "all"))
    if retained_mode != args.video_mode:
        raise RuntimeError(f"Use the existing retained video mode: {retained_mode}")
    if not set(protocol["seeds"]).issubset(seeds):
        raise RuntimeError("The new seed range must include all previous seeds")
    # Recording and launcher changes do not affect policy or simulation dynamics.
    allowed_recording_changes = {
        REPO / "scripts/touchguide/evaluate_poker_touchguide.py",
        REPO / "scripts/touchguide/run_poker_touchguide_policy.py",
        REPO / "scripts/touchguide/sim_recording.py",
        REPO / "scripts/run_poker_touchguide_comparison.sh",
    }
    for name, previous_hash in protocol["source_sha256"].items():
        path = Path(name)
        if path in allowed_recording_changes:
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest() != previous_hash:
            raise RuntimeError(f"Policy/physics source changed since the existing trials: {path}")
    rows, pending = [], []
    for variant in args.variants:
        for path in (output / variant).glob("seed_*"):
            suffix = path.name.removeprefix("seed_")
            if path.is_dir() and suffix.isdigit() and int(suffix) not in seeds:
                raise RuntimeError(f"Existing trial outside the requested seed range: {path}")
        for seed in seeds:
            trial = output / variant / f"seed_{seed:03d}"
            summary_path = trial / "summary.json"
            summary = None
            if summary_path.is_file():
                with contextlib.suppress(json.JSONDecodeError):
                    summary = json.loads(summary_path.read_text())
            if summary is not None and summary.get("status") in ("success", "task_not_completed"):
                metadata = summary.get("server_metadata", {})
                if metadata.get("guidance", {}).get("scale") != args.guidance_scale:
                    raise RuntimeError(f"Trial belongs to a different guidance scale: {trial}")
                if metadata.get("guidance", {}).get("guidance_steps") != args.guidance_steps:
                    raise RuntimeError(f"Trial belongs to different guidance steps: {trial}")
                validate_artifacts(args, trial, summary, variant, seed)
                rows.append(result_row(output, variant, seed, summary))
            else:
                pending.append((variant, seed))
    return rows, pending


def archive_incomplete(output: Path, variant: str, seed: int, stamp: str) -> None:
    """Retain interrupted attempts before retrying their seed; never replace completed trials."""
    trial = output / variant / f"seed_{seed:03d}"
    log = trial.with_suffix(".log")
    if not trial.exists() and not log.exists():
        return
    archive = output / "incomplete_attempts" / variant / f"seed_{seed:03d}_{stamp}"
    archive.mkdir(parents=True, exist_ok=False)
    if trial.exists():
        trial.rename(archive / "trial")
    if log.exists():
        log.rename(archive / "trial.log")
    print(f"ARCHIVED incomplete seed={seed}: {archive}", flush=True)


def main():
    args = parse_args()
    output = args.output_root.expanduser().resolve() / args.run_name
    serve = server_command(args, output)
    seeds = list(range(args.seed_start, args.seed_start + args.num_trials))
    results = []
    pending = [(variant, seed) for variant in args.variants for seed in seeds]
    resuming = args.resume and output.exists()
    if resuming:
        results, pending = resume_trials(args, output, seeds)
        for row in results:
            print(f"SKIP completed {row['variant']} seed={row['seed']}", flush=True)
    elif output.exists():
        raise RuntimeError(f"Output already exists; use --resume: {output}")
    if args.dry_run:
        print("OUTPUT:", output)
        print(f"TARGET={len(seeds)} per variant; COMPLETED={len(results)}; REMAINING={len(pending)}")
        if pending:
            print("SERVER:", shlex.join(serve))
        for variant, seed in pending:
            print("TRIAL:", shlex.join(trial_command(args, output, variant, seed)))
        return
    if not pending:
        write_results(args, output, seeds, results)
        print("COMPLETE (no new trials needed) RUN_DIR=" + str(output), flush=True)
        return
    with socket.socket() as connection:
        connection.settimeout(0.5)
        if connection.connect_ex(("127.0.0.1", args.port)) == 0:
            raise RuntimeError(f"Port {args.port} is occupied; choose another port")
    output.mkdir(parents=True, exist_ok=resuming)
    lock_file = (output / ".evaluation.lock").open("a")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lock_file.close()
        raise RuntimeError(f"Another evaluation is already writing to {output}") from error
    stamp = f"{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    server_log_path = output / (f"server_resume_{stamp}.log" if resuming else "server.log")
    if resuming:
        serve[-1] = str(output / f"server_metadata_resume_{stamp}.json")
    paths = [
        REPO / "src/openpi/touchguide" / name
        for name in (
            "policy.py",
            "sampler.py",
            "inference.py",
            "model.py",
            "tactile_dino.py",
            "recovery_model.py",
            "normalization.py",
            "negative_pairing.py",
        )
    ]
    paths += [
        Path(__file__),
        REPO / "scripts/touchguide/run_poker_touchguide_policy.py",
        REPO / "scripts/touchguide/sim_recording.py",
        REPO / "scripts/touchguide/serve_poker_touchguide.py",
        REPO / "scripts/run_poker_touchguide_comparison.sh",
        REPO / "src/openpi/models/tokenizer.py",
    ]
    paths += [
        args.sim_root / "src/kaihand_tactile_env" / name
        for name in (
            "tasks/poker_draw/mid_full.py",
            "tasks/poker_draw/rollout_eval.py",
            "tasks/poker_draw/randomization.py",
            "shared/contact_tactile.py",
            "shared/recording.py",
        )
    ]
    protocol = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    protocol.update(
        control_hz=30,
        execute_steps=args.execute_steps,
        prediction_horizon=30,
        seeds=seeds,
        source_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        retained_video_mode=args.video_mode,
    )
    if resuming:
        backup = output / f"protocol_before_resume_{stamp}.json"
        backup.write_bytes((output / "protocol.json").read_bytes())
        protocol["resumed_from_protocol"] = backup.name
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    launch_name = f"launch_script_resume_{stamp}.sh" if resuming else "launch_script.sh"
    (output / launch_name).write_bytes((REPO / "scripts/run_poker_touchguide_comparison.sh").read_bytes())
    write_results(args, output, seeds, results)
    server = trial_process = None
    print("RUN_DIR=" + str(output), flush=True)
    try:
        with server_log_path.open("x") as server_log:
            server = subprocess.Popen(
                serve,
                cwd=REPO,
                env=environment(args, model=True),
                stdout=server_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            wait_for_server(server, args, server_log_path)
            for variant in args.variants:
                (output / variant).mkdir(exist_ok=resuming)
                for seed in seeds:
                    if (variant, seed) not in pending:
                        continue
                    if resuming:
                        archive_incomplete(output, variant, seed, stamp)
                    command = trial_command(args, output, variant, seed)
                    print(f"START {variant} seed={seed}; log={output / variant / f'seed_{seed:03d}.log'}", flush=True)
                    with (output / variant / f"seed_{seed:03d}.log").open("x") as trial_log:
                        trial_process = subprocess.Popen(
                            command,
                            cwd=args.sim_root,
                            env=environment(args, model=False),
                            stdout=trial_log,
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                        )
                        try:
                            returncode = trial_process.wait(timeout=args.trial_wall_limit)
                        except subprocess.TimeoutExpired as error:
                            raise RuntimeError(f"Trial wall-time limit exceeded: {variant} seed={seed}") from error
                        finally:
                            stop_process(trial_process)
                            trial_process = None
                    trial = output / variant / f"seed_{seed:03d}"
                    if returncode != 0:
                        raise RuntimeError(f"Trial failed; inspect {output / variant / f'seed_{seed:03d}.log'}")
                    summary = json.loads((trial / "summary.json").read_text())
                    validate_artifacts(args, trial, summary, variant, seed)
                    if variant != args.variants[0]:
                        first = next(
                            row for row in results if row["variant"] == args.variants[0] and row["seed"] == seed
                        )
                        if summary["initial_card_pose"] != first["initial_card_pose"]:
                            raise RuntimeError("Paired trials started from different randomized card poses")
                    row = result_row(output, variant, seed, summary)
                    results.append(row)
                    write_results(args, output, seeds, results)
                    print(
                        f"DONE {variant} seed={seed} status={row['status']} sim_t={row['sim_seconds']:.3f}s", flush=True
                    )
        print("COMPLETE RUN_DIR=" + str(output), flush=True)
    finally:
        stop_process(trial_process)
        stop_process(server)
        lock_file.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
