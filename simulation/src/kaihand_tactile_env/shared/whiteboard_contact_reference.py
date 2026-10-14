"""Read recorded Whiteboard contact forces and cleaning-effect signals.

Camera pixels, model actions, and replayed physics are never used to
manufacture a missing contact or cleaning measurement.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .policy_tasks import WHITEBOARD_TASKS
from .whiteboard_contact_intensity import ContactTrace, contact_phases

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
DEFAULT_DEMO_ROOT = Path("/path/to/datasets/sim/task_demo")


def _text(value: Any) -> str:
  return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _json_attribute(attrs: Any, name: str) -> dict[str, Any]:
  value = json.loads(_text(attrs.get(name, "{}")))
  if not isinstance(value, dict):
    raise ValueError(f"{name} must encode a JSON object")
  return value


def _times(values: Any, name: str) -> np.ndarray:
  times = np.asarray(values, dtype=np.float64)
  if times.ndim != 1 or not len(times) or not np.isfinite(times).all():
    raise ValueError(f"{name} must be a nonempty finite timestamp vector")
  if np.any(np.diff(times) <= 0):
    raise ValueError(f"{name} must be strictly increasing")
  return times


def _durations(times: np.ndarray) -> np.ndarray:
  # Left-hold over observed intervals; never extend past the last sample.
  return np.r_[np.diff(times), 0.0]


def _force(values: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
  force = np.asarray(values, dtype=np.float64)
  if force.shape != shape or not np.isfinite(force).all() or np.any(force < 0):
    raise ValueError(f"{name} must contain finite nonnegative forces with shape {shape}")
  return force


def _signal(values: Any, shape: tuple[int, ...], name: str, *, maximum: float | None = None) -> np.ndarray:
  """Keep missing/invalid effect samples missing, including bounded fractions."""
  signal = np.asarray(values, dtype=np.float64)
  if signal.shape != shape:
    raise ValueError(f"{name} must have shape {shape}")
  valid = np.isfinite(signal) & (signal >= 0.0)
  if maximum is not None:
    valid &= signal <= maximum
  return np.where(valid, signal, np.nan)


def _patch_signal(values: Any, rows: int, name: str, *, remaining: bool) -> np.ndarray:
  patches = np.asarray(values, dtype=np.float64)
  if patches.ndim != 2 or patches.shape[0] != rows or patches.shape[1] == 0:
    raise ValueError(f"{name} must have shape ({rows}, nonzero patch count)")
  valid = np.isfinite(patches).all(axis=1) & (patches >= 0.0).all(axis=1)
  if remaining:
    valid &= (patches <= 1.0).all(axis=1)
    result = 1.0 - patches.mean(axis=1)
  else:
    result = patches.sum(axis=1)
  return np.where(valid, result, np.nan)


def _final_cleaned_fraction(values: dict[str, Any]) -> float | None:
  if values.get("cleaned_fraction") is not None:
    fraction = _signal(values["cleaned_fraction"], (), "final cleaned_fraction", maximum=1.0)
    return float(fraction) if np.isfinite(fraction) else None
  if values.get("ink_remaining") is not None:
    patches = np.asarray(values["ink_remaining"], dtype=np.float64)
    if patches.ndim == 1 and patches.size:
      fraction = _patch_signal(patches[None], 1, "final ink_remaining", remaining=True)[0]
      return float(fraction) if np.isfinite(fraction) else None
  return None


def _observed_hz(times: np.ndarray) -> float | None:
  return float(1.0 / np.median(np.diff(times))) if len(times) > 1 else None


def _right_indices(names: list[str]) -> list[int]:
  expected = [f"hand_r_{finger}_link{6 if finger == 'thumb' else 4}" for finger in FINGERS]
  if len(names) != len(set(names)) or any(name not in names for name in expected):
    raise ValueError("force link_names must contain each canonical right fingertip exactly once")
  return [names.index(name) for name in expected]


def _finger_channels(normal: np.ndarray, tangent: np.ndarray | None = None) -> dict[str, np.ndarray]:
  result = {f"finger_{name}_normal_n": normal[:, i] for i, name in enumerate(FINGERS)}
  result["grip_normal_n"] = normal.sum(axis=1)
  if tangent is not None:
    result.update({f"finger_{name}_tangent_load_n": tangent[:, i] for i, name in enumerate(FINGERS)})
    result["grip_tangent_load_n"] = tangent.sum(axis=1)
  return result


def load_whiteboard_contact_reference(
  *,
  task: str = "whiteboard-wipe",
  reference_hdf5: Path | None = None,
  demo_root: Path = DEFAULT_DEMO_ROOT,
) -> ContactTrace:
  """Load one successful expert, resolving the exact task-demo replay source.

  The replay's hash is retained as declared provenance, not independently
  verified: hashing a multi-GB video-containing HDF5 is unnecessary for this
  force-only report. Different task variants cannot silently share references.
  """
  if task not in WHITEBOARD_TASKS:
    raise ValueError(f"not a Whiteboard task: {task}")
  replay_path = None
  replay: dict[str, Any] = {}
  if reference_hdf5 is None:
    root = Path(demo_root).expanduser()
    candidates = (
      root / "_metadata" / task / "replay.json",
      root / "_metadata" / task / "review" / "replay.json",
      root / task / "review" / "replay.json",
      root / task / "replay.json",
    )
    replay_path = next((path for path in candidates if path.is_file()), None)
    if replay_path is None:
      raise FileNotFoundError(f"no task-demo replay.json for {task} under {root}")
    replay = json.loads(replay_path.read_text())
    if replay.get("task") != task or replay.get("completed") is not True:
      raise ValueError(f"{replay_path}: replay must be completed and match task {task}")
    source = replay.get("source_hdf5")
    if not isinstance(source, str) or not source:
      raise ValueError(f"{replay_path}: missing source_hdf5")
    reference_hdf5 = Path(source).expanduser()
    if not reference_hdf5.is_absolute():
      reference_hdf5 = replay_path.parent / reference_hdf5
  source_path = Path(reference_hdf5).expanduser().resolve(strict=True)
  with h5py.File(source_path, "r") as file:
    metadata = _json_attribute(file.attrs, "metadata_json")
    outcome = _json_attribute(file.attrs, "outcome_json")
    recorded_task = metadata.get("scene", metadata.get("task", file.attrs.get("task")))
    if recorded_task != task:
      raise ValueError(f"expert task {recorded_task!r} does not match {task!r}")
    if outcome.get("success") is not True:
      raise ValueError("contact expert must have outcome_json.success=true")
    if task == "whiteboard-wipe-horizontal-line":
      from .config import task_config
      from .task_variants import current_variant_environment, validate_variant_outcome

      validate_variant_outcome(task, outcome)
      environment = current_variant_environment(task)
      for key, expected in environment.items():
        recorded = metadata.get(key, file.attrs.get(key))
        if recorded != expected:
          raise ValueError(f"single-line expert {key} does not match current task")
      if (
        metadata.get("ink_pattern") != task_config(task).INK_PATTERN
        or metadata.get("ink_patch_count") != task_config(task).INK_COUNT
      ):
        raise ValueError("single-line expert ink layout does not match current task")
    force_group = file["tactile_contact_force"]
    if _text(force_group.attrs.get("force_unit", "N")) != "N":
      raise ValueError("expert contact force unit must be N")
    clock_path = "tactile_contact_force/timestamp" if "timestamp" in force_group else "state/timestamp"
    times = _times(file[clock_path][:], clock_path)
    state_times = _times(file["state/timestamp"][:], "state/timestamp")
    if len(state_times) != len(times):
      raise ValueError("expert force and state rows must have matching sample counts")
    # These channels share native recorder rows. Preserve their actual clock
    # skew rather than resampling and implying a higher measurement rate.
    clock_skew = state_times - times
    max_skew = float(np.max(np.abs(clock_skew)))
    step_limit = max(float(np.median(np.diff(state_times))) if len(times) > 1 else 0.0, 1e-6)
    if max_skew > step_limit + 1e-6:
      raise ValueError("expert force/state clocks differ by more than one recorded interval")
    names = [_text(name) for name in force_group["link_names"][:]]
    indices = _right_indices(names)
    shape = (len(times), len(names))
    normal = _force(force_group["normal_force_n"][:], shape, "normal_force_n")[:, indices]
    tangent = None
    if "tangent_load_n" in force_group:
      tangent = _force(force_group["tangent_load_n"][:], shape, "tangent_load_n")[:, indices]
    channels = _finger_channels(normal, tangent)
    group = task.replace("-", "_")
    if task == "whiteboard-wipe" or (
      task == "whiteboard-wipe-horizontal" and group not in file
    ):
      source_fields = {
        "board_normal_n": "whiteboard_wipe/board_force_n",
        "board_tangent_load_n": "whiteboard_wipe/board_tangent_force_n",
        "hand_board_normal_n": "whiteboard_wipe/direct_hand_board_force_n",
      }
    else:
      source_fields = {
        "board_normal_n": f"{group}/board_normal_force_n",
        "board_tangent_load_n": f"{group}/board_tangent_force_n",
        "hand_board_normal_n": f"{group}/direct_hand_board_force_n",
      }
    for channel, field in source_fields.items():
      if field in file:
        channels[channel] = _force(file[field][:], (len(times),), field)
    if "board_normal_n" not in channels:
      raise ValueError(f"expert is missing {source_fields['board_normal_n']}")
    board_group = source_fields["board_normal_n"].rsplit("/", 1)[0]
    # Task fields are appended once per native state row. If a task supplies
    # its own clock, verify it rather than silently pairing unrelated rows.
    signal_clock = f"{board_group}/timestamp"
    if signal_clock in file:
      board_times = _times(file[signal_clock][:], signal_clock)
      if len(board_times) != len(state_times) or not np.allclose(board_times, state_times, rtol=0, atol=1e-7):
        raise ValueError("expert Whiteboard signal clock does not match paired state rows")
    else:
      signal_clock = "state/timestamp; paired native recorder rows"
    signals = {"cleaned_fraction": np.full(len(times), np.nan)}
    signal_sources: dict[str, str] = {}
    cleaned_field = f"{board_group}/cleaned_fraction"
    remaining_field = f"{board_group}/ink_remaining"
    work_field = f"{board_group}/patch_work_j"
    if cleaned_field in file:
      signals["cleaned_fraction"] = _signal(file[cleaned_field][:], (len(times),), cleaned_field, maximum=1.0)
      signal_sources["cleaned_fraction"] = cleaned_field
    elif remaining_field in file:
      signals["cleaned_fraction"] = _patch_signal(file[remaining_field][:], len(times), remaining_field, remaining=True)
      signal_sources["cleaned_fraction"] = f"1 - mean({remaining_field}, patch axis)"
    if work_field in file:
      signals["cleaning_work_j"] = _patch_signal(file[work_field][:], len(times), work_field, remaining=False)
      signal_sources["cleaning_work_j"] = f"sum({work_field}, patch axis)"
    peaks = {}
    for channel, key in (
      ("board_normal_n", "peak_board_force_n"),
      ("board_tangent_load_n", "peak_board_tangent_force_n"),
      ("hand_board_normal_n", "peak_direct_hand_board_force_n"),
    ):
      if key in outcome:
        peaks[channel] = float(_force(outcome[key], (), key))
    native_phases = [_text(phase) for phase in file["commands/phase"][:]] if "commands/phase" in file else []
    protocol = {
      "task": task,
      "source_kind": "expert_hdf5",
      "source_hdf5": str(source_path),
      "source_replay_json": str(replay_path) if replay_path else None,
      "declared_source_sha256": replay.get("source_sha256"),
      "source_sha256_verified": False,
      "reference_episode_count": 1,
      "reference_population": "one successful task-demo example; not an expert population confidence interval",
      "success": True,
      "task_success": True,
      "trial_status": "success",
      "stop_reason": outcome.get("reason"),
      "termination_reason": outcome.get("reason"),
      "trial_sim_seconds": outcome.get("elapsed_s"),
      "final_cleaned_fraction": _final_cleaned_fraction(outcome),
      "final_cleaned_fraction_source": "outcome_json; terminal value only, not used to fill the curve",
      "signal_sources": signal_sources,
      "signal_clock": signal_clock,
      "cleaned_fraction_signal_status": "recorded" if signal_sources.get("cleaned_fraction") else "missing_time_series",
      "missing_cleaned_fraction_sample_count": int(np.count_nonzero(~np.isfinite(signals["cleaned_fraction"]))),
      "cleaned_fraction_semantics": "1 - mean remaining ink fraction across patches, in [0, 1]",
      "cleaning_work_semantics": "sum of cumulative friction work assigned to local ink patches; not total robot or eraser mechanical work",
      "sampling": "native recorded samples; left-hold over observed intervals; terminal weight zero",
      "clock": clock_path,
      "clock_semantics": _text(force_group.attrs.get("clock_semantics", "native recorded timestamp")),
      "board_clock": "state/timestamp; paired native recorder rows",
      "maximum_force_state_clock_skew_s": max_skew,
      "observed_hz": _observed_hz(times),
      "source_start_time_s": float(times[0]),
      "reported_peaks_n": peaks,
      "reported_peaks_source": "outcome_json task-maintained peaks; separate from sampled curves",
      "ink_randomization": metadata.get("ink_randomization", outcome.get("ink_randomization")),
      "model_sha256": _text(file.attrs.get("model_sha256", "")),
      "model_fingerprint": _text(file.attrs.get("model_fingerprint", "")),
      "physics_hz": float(file.attrs["physics_hz"]) if "physics_hz" in file.attrs else None,
      "control_hz": float(file.attrs["control_hz"]) if "control_hz" in file.attrs else None,
      "mechanics": {key: outcome[key] for key in ("eraser_board_friction", "eraser_board_solimp", "friction_impedance_ratio", "eraser_contact_margins_m") if key in outcome},
      "native_phase_counts": {phase: native_phases.count(phase) for phase in dict.fromkeys(native_phases)},
      "phase_definition": "common contact-state phases; native expert command phases are metadata only",
      "tangent_load_semantics": "sum of contact tangent magnitudes, not norm of net tangent vector",
      "force_source": _text(file.attrs.get("contact_force_source", "unknown")),
      "limitations": ["Single expert; initial conditions and mechanics must be checked before model ranking.", "Recorded-sample peaks can miss between-sample impacts; task-reported peaks are separate."],
    }
    if task == "whiteboard-wipe-horizontal-line":
      protocol.update(
        mechanics_version=metadata.get("mechanics_version", file.attrs.get("mechanics_version")),
        ink_pattern=metadata["ink_pattern"],
        ink_patch_count=metadata["ink_patch_count"],
      )
  return ContactTrace(time_s=times - times[0], duration_s=_durations(times), channels=channels, phases=contact_phases(channels), metadata=protocol, signals=signals)


def load_historical_whiteboard_contact_trial(trial_dir: Path) -> ContactTrace:
  """Read old flat/review layouts, joining only truly coincident timestamps."""
  trial = Path(trial_dir).expanduser().resolve(strict=True)
  summary = json.loads((trial / "summary.json").read_text())
  evaluation = summary.get("evaluation", {})
  task = evaluation.get("task", summary.get("task"))
  if task not in WHITEBOARD_TASKS:
    raise ValueError(f"not a Whiteboard evaluation: {task}")
  trace_path = next((path for path in (trial / "evaluation_rollout_trace.npz", trial / "review" / "evaluation_rollout_trace.npz") if path.is_file()), None)
  if trace_path is None:
    raise FileNotFoundError(f"no evaluation_rollout_trace.npz under {trial}")
  with np.load(trace_path, allow_pickle=False) as data:
    if "simulation_time_s" not in data:
      raise ValueError("historical trace needs absolute simulation_time_s for exact control-log alignment")
    times = _times(data["simulation_time_s"], "simulation_time_s")
    normal = _force(data["fingertip_normal_force_n"], (len(times), 5), "fingertip_normal_force_n")
    if "fingertip_link_names" in data:
      indices = _right_indices([_text(value) for value in data["fingertip_link_names"]])
      normal = normal[:, indices]
    channels = _finger_channels(normal)
  original_count = len(times)
  clock_source = "evaluation_rollout_trace.npz simulation_time_s"
  channel_sources = {channel: str(trace_path) for channel in channels}
  signals = {"cleaned_fraction": np.full(len(times), np.nan)}
  signal_sources: dict[str, str] = {}
  control_path = trial / "control_steps.jsonl"
  matched_indices: list[int] = []
  board: list[float] = []
  native_stages: list[str] = []
  last_control_index = -1
  if control_path.is_file():
    for line in control_path.read_text().splitlines():
      if not line.strip():
        continue
      row = json.loads(line)
      task_row = row.get("task", {})
      value = task_row.get("board_normal_force_n")
      progress = task_row.get("cleaned_fraction")
      time = row.get("sim_time")
      if (value is None and progress is None) or time is None:
        continue
      if not np.isfinite(float(time)):
        continue
      candidate = int(np.searchsorted(times, float(time)))
      choices = [index for index in (candidate - 1, candidate) if 0 <= index < len(times)]
      index = min(choices, key=lambda index: abs(times[index] - float(time)))
      if abs(times[index] - float(time)) > 1e-7:
        continue
      if index <= last_control_index:
        raise ValueError("control_steps matching timestamps must be strictly increasing")
      last_control_index = index
      if value is not None:
        matched_indices.append(index)
        board.append(float(value))
      if progress is not None:
        signals["cleaned_fraction"][index] = _signal(progress, (), "control task.cleaned_fraction", maximum=1.0)
        signal_sources["cleaned_fraction"] = str(control_path)
      native_stages.append(str(row.get("stage_after_action", row.get("task", {}).get("stage", "unknown"))))
  # Some recorded layouts keep task snapshots in frames.jsonl. Only explicit
  # progress values at the same simulation timestamp may fill missing rows.
  frame_path = next((path for path in (trial / "frames.jsonl", trial / "review" / "frames.jsonl") if path.is_file()), None)
  if frame_path is not None and not np.isfinite(signals["cleaned_fraction"]).all():
    for line in frame_path.read_text().splitlines():
      if not line.strip():
        continue
      row = json.loads(line)
      snapshot = row.get("task", row.get("task_snapshot", {}))
      progress = snapshot.get("cleaned_fraction", snapshot.get("state", {}).get("cleaned_fraction"))
      time = row.get("simulation_time_s", row.get("sim_time"))
      if progress is None or time is None:
        continue
      if not np.isfinite(float(time)):
        continue
      candidate = int(np.searchsorted(times, float(time)))
      choices = [index for index in (candidate - 1, candidate) if 0 <= index < len(times)]
      index = min(choices, key=lambda index: abs(times[index] - float(time)))
      if abs(times[index] - float(time)) > 1e-7 or np.isfinite(signals["cleaned_fraction"][index]):
        continue
      signals["cleaned_fraction"][index] = _signal(progress, (), "frame task.cleaned_fraction", maximum=1.0)
      signal_sources["cleaned_fraction"] = "; ".join(dict.fromkeys([signal_sources.get("cleaned_fraction", ""), str(frame_path)])).strip("; ")
  if matched_indices:
    board_values = np.full(len(times), np.nan)
    board_values[matched_indices] = _force(board, (len(matched_indices),), "control task.board_normal_force_n")
    channels["board_normal_n"] = board_values
    channel_sources["board_normal_n"] = str(control_path)
    clock_source += "; exact matched control_steps.jsonl sim_time, tolerance 1e-7 s"
  guards = evaluation.get("contact_guard_diagnostics", {})
  reported_peaks = {}
  for channel, key in (("board_normal_n", "peak_board_normal_force_n"), ("hand_board_normal_n", "peak_direct_hand_board_force_n")):
    if key in guards:
      reported_peaks[channel] = float(_force(guards[key], (), key))
  metadata = {
    "task": task,
    "source_kind": "historical_control_step_preview",
    "trial_dir": str(trial),
    "source_trace": str(trace_path),
    "success": evaluation.get("success"),
    "task_success": evaluation.get("success"),
    "trial_status": summary.get("status"),
    "stop_reason": summary.get("stop_reason", summary.get("termination_reason", summary.get("error"))),
    "termination_reason": summary.get("termination_reason", summary.get("stop_reason", summary.get("error"))),
    "trial_sim_seconds": summary.get("trial_sim_seconds", summary.get("sim_seconds")),
    "final_cleaned_fraction": _final_cleaned_fraction(evaluation.get("state", evaluation)),
    "final_cleaned_fraction_source": "original summary.json evaluation.state; terminal value only, not used to fill the curve",
    "signal_sources": signal_sources,
    "signal_clock": "exact matched control/frame simulation timestamps, tolerance 1e-7 s",
    "cleaned_fraction_signal_status": "recorded" if signal_sources.get("cleaned_fraction") else "missing_time_series",
    "missing_cleaned_fraction_sample_count": int(np.count_nonzero(~np.isfinite(signals["cleaned_fraction"]))),
    "cleaned_fraction_semantics": "recorded task.cleaned_fraction, in [0, 1]",
    "clock": clock_source,
    "source_start_time_s": float(times[0]),
    "observed_hz": _observed_hz(times),
    "sampling": "saved control-cadence samples; left-hold; last sample has zero duration; no resimulation",
    "source_sample_count": original_count,
    "matched_board_sample_count": len(matched_indices),
    "missing_board_sample_count": original_count - len(matched_indices),
    "channel_sources": channel_sources,
    "reported_peaks_n": reported_peaks,
    "reported_peaks_source": "original summary.json evaluation.contact_guard_diagnostics",
    "ink_randomization": evaluation.get("ink_randomization"),
    "native_stage_counts": {stage: native_stages.count(stage) for stage in dict.fromkeys(native_stages)},
    "phase_definition": "common contact-state phases; native model stages are metadata only",
    "limitations": [
      "Historical preview is control-cadence, not physics-step data; short impacts may be missed.",
      "Old net tangent vectors cannot reconstruct tangent_load_n; no shear-strength curves are fabricated.",
      "Hand-board collision has a reported peak only; its missing time series is not filled with zeros.",
      "Only exact tactile/control timestamp matches supply board forces; missing board samples stay NaN with explicit coverage.",
      "Missing or invalid cleaning samples stay NaN; a summary terminal value does not manufacture a cleaning curve.",
      "Initial conditions may differ from the expert, so this is descriptive rather than matched-seed ranking.",
    ],
  }
  return ContactTrace(time_s=times - times[0], duration_s=_durations(times), channels=channels, phases=contact_phases(channels), metadata=metadata, signals=signals)
