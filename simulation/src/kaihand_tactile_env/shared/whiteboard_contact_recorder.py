"""Read-only physics-rate contact diagnostics for whiteboard policy trials.

Forces are staged between mj_step and the task's next mj_forward, then committed
by TaskPolicyAdapter only after its normal physics validation succeeds. Export
uses cached valid samples, including when the last simulation state is invalid.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from .config import FINGERTIP_LINK_NAMES
from .policy_tasks import WHITEBOARD_TASKS

DEFAULT_DEMO_ROOT = Path("/path/to/datasets/sim/task_demo")
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
CHANNEL_NAMES = (
  *(f"finger_{finger}_normal_n" for finger in FINGERS),
  *(f"finger_{finger}_tangent_load_n" for finger in FINGERS),
  "grip_normal_n",
  "grip_tangent_load_n",
  "board_normal_n",
  "board_tangent_load_n",
  "hand_board_normal_n",
)
SIGNAL_NAMES = ("cleaned_fraction", "cleaning_work_j")


def _cleaning_signals(simulation: Any) -> np.ndarray:
  remaining = np.asarray(getattr(simulation, "remaining", []), dtype=np.float64)
  cleaned = (
    float(np.mean(1.0 - remaining))
    if remaining.size
    and np.isfinite(remaining).all()
    and np.all((remaining >= 0) & (remaining <= 1))
    else np.nan
  )
  cleaning = getattr(simulation, "cleaning", None)
  work = np.asarray(getattr(cleaning, "work_j", []), dtype=np.float64)
  return np.array(
    [
      cleaned,
      float(work.sum())
      if work.size and np.isfinite(work).all() and np.all(work >= 0)
      else np.nan,
    ]
  )


class WhiteboardContactRecorder:
  """Keep one scalar per contact channel and valid integrated physics step."""

  def __init__(self, adapter: Any) -> None:
    self.task = adapter.task
    self.seed = getattr(adapter, "seed", None)
    self.simulation = adapter.simulation
    self.error: str | None = None
    self.attempted_steps = 0
    self.rejected_samples = 0
    self._times: list[float] = []
    self._durations: list[float] = []
    self._rows: list[np.ndarray] = []
    self._signal_rows: list[np.ndarray] = []
    self._pending: tuple[float, float, np.ndarray, np.ndarray] | None = None
    self._finished: dict | None = None
    self._wrench = np.zeros(6, dtype=np.float64)
    self._callback = self.stage_step
    self._initial_metadata: dict[str, Any] = {}
    try:
      sim = self.simulation
      if getattr(sim, "evaluation_contact_observer", None) is not None:
        raise RuntimeError("whiteboard contact observer is already attached")
      self._pad_indices = {
        int(sim.model.geom(f"{name}_tactile_pad_col").id): index
        for index, name in enumerate(FINGERTIP_LINK_NAMES[5:])
      }
      self._target_ids = {
        int(sim.model.geom(name).id) for name in ("eraser_handle", "eraser_pad")
      }
      self._initial_metadata = {
        "physics_timestep_s": float(sim.timestep),
        "ink_randomization": getattr(sim, "ink_randomization", None),
        "initial_object_pose_wxyz": np.asarray(
          adapter.initial_object_pose, dtype=np.float64
        ).tolist(),
        "initial_cleaned_fraction": float(_cleaning_signals(sim)[0]),
        "cleaned_fraction_clock": "interval_end",
        "cleaning_work_semantics": "各墨迹块累计摩擦功之和；不能当作整体机械做功",
      }
      if self.task == "whiteboard-wipe-horizontal-line":
        self._initial_metadata.update(
          mechanics_version=sim.task_config.MECHANICS_VERSION,
          ink_pattern=sim.task_config.INK_PATTERN,
          ink_patch_count=sim.task_config.INK_COUNT,
        )
      sim.evaluation_contact_observer = self._callback
      adapter._contact_intensity_recorder = self
    except Exception as exc:
      self.error = (
        f"contact recorder initialization failed: {type(exc).__name__}: {exc}"
      )

  def stage_step(self, simulation: Any, before_time: float) -> None:
    """Cache the current solver result; never step or forward the simulation."""
    if self._finished is not None:
      return
    self.attempted_steps += 1
    if self.error is not None:
      self.rejected_samples += 1
      return
    if self._pending is not None:
      self.rejected_samples += 1
      self._pending = None
    try:
      duration = float(simulation.timestep)
      after_time = float(simulation.data.time)
      if (
        not math.isfinite(before_time)
        or not math.isfinite(after_time)
        or not math.isfinite(duration)
        or duration <= 0
        or not math.isclose(
          after_time - before_time, duration, rel_tol=1e-7, abs_tol=1e-12
        )
      ):
        self.rejected_samples += 1
        return
      row = np.zeros(len(CHANNEL_NAMES), dtype=np.float64)
      for contact_id in range(int(simulation.data.ncon)):
        contact = simulation.data.contact[contact_id]
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        index = self._pad_indices.get(geom1) if geom2 in self._target_ids else None
        if index is None and geom1 in self._target_ids:
          index = self._pad_indices.get(geom2)
        if index is None:
          continue
        self._wrench.fill(0)
        mujoco.mj_contactForce(
          simulation.model, simulation.data, contact_id, self._wrench
        )
        row[index] += abs(float(self._wrench[0]))
        # Sum magnitudes per solver contact, preserving opposing shear loads.
        row[5 + index] += float(np.linalg.norm(self._wrench[1:3]))
      row[10] = row[:5].sum()
      row[11] = row[5:10].sum()
      row[12:] = (
        float(simulation.board_force),
        float(simulation.board_tangent_force),
        float(simulation.direct_hand_board_force),
      )
      if not np.isfinite(row).all() or np.any(row < 0):
        self.rejected_samples += 1
        return
      self._pending = (before_time, duration, row, _cleaning_signals(simulation))
    except Exception as exc:
      # This optional diagnostic never changes the policy's stopping rules.
      self.error = f"contact capture failed: {type(exc).__name__}: {exc}"
      self.rejected_samples += 1
      self._pending = None

  def commit_step(self) -> None:
    """Accept a cached step after the adapter has checked physics integrity."""
    if self._pending is None:
      return
    timestamp, duration, row, signal_row = self._pending
    self._pending = None
    if self._times and timestamp <= self._times[-1]:
      self.error = "contact capture timestamps are not strictly increasing"
      self.rejected_samples += 1
      return
    self._times.append(timestamp)
    self._durations.append(duration)
    self._rows.append(row)
    self._signal_rows.append(signal_row)

  def trace(self, report: dict):
    """Build an exportable snapshot without reading current simulation data."""
    from .whiteboard_contact_intensity import ContactTrace, contact_phases

    if self._pending is not None:
      self.rejected_samples += 1
      self._pending = None
    rows = (
      np.stack(self._rows)
      if self._rows
      else np.empty((0, len(CHANNEL_NAMES)), dtype=np.float64)
    )
    channels = {name: rows[:, index] for index, name in enumerate(CHANNEL_NAMES)}
    signal_rows = (
      np.stack(self._signal_rows)
      if self._signal_rows
      else np.empty((0, len(SIGNAL_NAMES)), dtype=np.float64)
    )
    return ContactTrace(
      time_s=np.asarray(self._times, dtype=np.float64),
      duration_s=np.asarray(self._durations, dtype=np.float64),
      channels=channels,
      phases=contact_phases(channels),
      signals={name: signal_rows[:, index] for index, name in enumerate(SIGNAL_NAMES)},
      metadata={
        "task": self.task,
        "seed": self.seed,
        "sampling": "physics_step",
        "source": "MuJoCo integrated solver contacts before post-step mj_forward",
        "clock": "time_s is interval start; duration_s is the integrated timestep",
        "force_unit": "N",
        "tangent_semantics": "sum of individual contact tangent magnitudes; no cancellation",
        "target_geom_names": ["eraser_handle", "eraser_pad"],
        "attempted_steps": self.attempted_steps,
        "valid_steps": len(self._times),
        "rejected_samples": self.rejected_samples,
        "capture_error": self.error,
        "task_success": report.get("task_success", False),
        "trial_status": report.get("status"),
        "trial_sim_seconds": report.get("sim_seconds"),
        "stop_reason": report.get("stop_reason"),
        "termination_reason": report.get("error"),
        "final_cleaned_fraction": float(signal_rows[-1, 0])
        if len(signal_rows)
        else None,
        "physics_valid": report.get("physics_valid"),
        "model_family": report.get("model_family"),
        "sampling_is_independent_of_video": True,
        **self._initial_metadata,
      },
    )

  def detach(self) -> None:
    if getattr(self.simulation, "evaluation_contact_observer", None) is self._callback:
      self.simulation.evaluation_contact_observer = None


def start_whiteboard_contact_recording(
  adapter: Any,
) -> WhiteboardContactRecorder | None:
  if getattr(adapter, "task", None) not in WHITEBOARD_TASKS:
    return None
  return WhiteboardContactRecorder(adapter)


def finish_whiteboard_contact_recording(
  recorder: WhiteboardContactRecorder,
  output: Path,
  report: dict,
  *,
  reference_hdf5: Path | None = None,
  demo_root: Path = DEFAULT_DEMO_ROOT,
) -> dict:
  """Export independent diagnostics; unavailable experts do not fail a trial."""
  if recorder._finished is not None:
    return recorder._finished
  recorder.detach()
  try:
    from .whiteboard_contact_intensity import export_contact_intensity
    from .whiteboard_contact_reference import load_whiteboard_contact_reference

    rollout = recorder.trace(report)
    reference = None
    reference_error = None
    try:
      reference = load_whiteboard_contact_reference(
        task=recorder.task,
        reference_hdf5=reference_hdf5,
        demo_root=demo_root,
      )
    except Exception as exc:
      reference_error = f"{type(exc).__name__}: {exc}"
    recorder._finished = export_contact_intensity(
      output, rollout, reference, reference_error=reference_error
    )
  except Exception as exc:
    recorder._finished = {
      "status": "error",
      "reason": f"contact intensity export failed: {type(exc).__name__}: {exc}",
      "capture_error": recorder.error,
      "valid_steps": len(recorder._times),
    }
  return recorder._finished
