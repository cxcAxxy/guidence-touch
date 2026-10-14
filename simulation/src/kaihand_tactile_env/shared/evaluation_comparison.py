"""Model-independent control-step rollout and reference-comparison artifacts.

The rollout is always saved, even when the training dataset does not contain
the physical fields needed for a truthful episode-00 comparison.  Reference
curves are never synthesized from actions or sparse policy observations.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .config import FINGERTIP_LINK_NAMES
from .openwam_evaluation_plots import (
  FINGER_NAMES,
  NATIVE_RIGHT_WRIST_SITE,
  RIGHT_FINGERTIP_LINK_NAMES,
  RIGHT_HAND_ACTUATED_JOINT_NAMES,
  OpenWAMEvaluationPlots,
  _canonical_right_sample,
  aggregate_fingertip_forces,
  load_openwam_reference,
)

SCHEMA_VERSION = "kaihand-evaluation-comparison-v1"


def _right_wrist_and_hand_state(simulation: Any) -> np.ndarray:
  """Read physical MuJoCo state, independent of any policy input contract."""
  model = simulation.model
  data = simulation.data
  site_id = int(model.site(NATIVE_RIGHT_WRIST_SITE).id)
  position = np.asarray(data.site_xpos[site_id], dtype=np.float64)
  rotation = np.asarray(data.site_xmat[site_id], dtype=np.float64).reshape(3, 3)
  if position.shape != (3,):
    raise ValueError(f"right wrist site position must have shape (3,), got {position.shape}")
  if not np.isfinite(position).all() or not np.isfinite(rotation).all():
    raise ValueError("right wrist site pose must be finite")
  # The dataset contract stores rotation columns, not Euler angles or a
  # policy-specific wrist target.  Match its nine-dimensional pose exactly.
  wrist = np.concatenate((position, rotation[:, 0], rotation[:, 1]))
  addresses = getattr(simulation, "_qpos_address", None)
  if addresses is None:
    indices = [int(model.joint(name).qposadr) for name in RIGHT_HAND_ACTUATED_JOINT_NAMES]
  else:
    indices = [int(addresses[name]) for name in RIGHT_HAND_ACTUATED_JOINT_NAMES]
  hand = np.asarray(data.qpos[indices], dtype=np.float64)
  state = np.concatenate((wrist, hand))
  if state.shape != (29,) or not np.isfinite(state).all():
    raise ValueError(f"right wrist and hand state must be finite (29,), got {state.shape}")
  return state


def _expected_reference_gap(error: Exception) -> bool:
  """Identify absent comparison data, not malformed records or plotting bugs."""
  if isinstance(error, FileNotFoundError):
    return True
  if not isinstance(error, ValueError):
    return False
  message = str(error)
  return (
    "no record for output_episode_index" in message
    or "missing required dataset" in message
    or "missing required group cameras/head" in message
    or "is missing required attribute" in message
    or "does not record native site" in message
    or "is missing names:" in message
  )


class EvaluationComparisonTrace:
  """Capture physical right-hand state and tactile grids at control cadence.

  ``capture`` does not advance the simulation.  Repeated calls for the same
  simulation timestamp are ignored, so both a controller and video recorder
  may safely ask for the current state without duplicating samples.
  """

  def __init__(self, tactile_provider: Any | None = None) -> None:
    self.tactile_provider = tactile_provider
    self._times: list[float] = []
    self._states: list[np.ndarray] = []
    self._normal_grids: list[np.ndarray] = []
    self._tangent_grids: list[np.ndarray] = []
    self._tactile_source = str(getattr(tactile_provider, "source", "unspecified"))
    self._finished = False

  @property
  def sample_count(self) -> int:
    return len(self._times)

  def capture(
    self,
    simulation: Any,
    timestamp: float | None = None,
    *,
    normal_taxel_force_n: np.ndarray | None = None,
    tangent_taxel_force_n: np.ndarray | None = None,
    link_names: Sequence[str] | None = None,
  ) -> bool:
    """Append one physical sample; return ``False`` for a duplicate time.

    The optional direct arrays reuse a video frame's already-read tactile
    sample.  They must contain both force components and be ordered as
    ``FINGERTIP_LINK_NAMES`` (10 fingertips) or
    ``RIGHT_FINGERTIP_LINK_NAMES`` (right five), unless ``link_names`` gives
    their actual order.  Skipped video frames can omit them and read the
    provider normally.
    """
    if self._finished:
      raise RuntimeError("evaluation comparison trace has already been finished")
    data = simulation.data
    raw_time = data.time if timestamp is None else timestamp
    if isinstance(raw_time, bool):
      raise ValueError("capture timestamp must be a finite scalar")
    try:
      current_time = float(raw_time)
    except (TypeError, ValueError) as error:
      raise ValueError("capture timestamp must be a finite scalar") from error
    if not np.isfinite(current_time):
      raise ValueError("capture timestamp must be a finite scalar")
    if self._times:
      difference = current_time - self._times[-1]
      if abs(difference) <= 1.0e-9:
        return False
      if difference < 0.0:
        raise ValueError("rollout simulation times must be increasing")

    state = _right_wrist_and_hand_state(simulation)
    direct_normal = normal_taxel_force_n is not None
    direct_tangent = tangent_taxel_force_n is not None
    if direct_normal != direct_tangent:
      raise ValueError("normal and tangent tactile grids must be provided together")
    if direct_normal:
      normal_input = np.asarray(normal_taxel_force_n, dtype=np.float64)
      if normal_input.ndim != 3 or normal_input.shape[1:] != (7, 5):
        raise ValueError(
          "direct normal tactile grid must have shape (5 or 10, 7, 5), "
          f"got {normal_input.shape}"
        )
      if link_names is None:
        if normal_input.shape[0] == len(FINGERTIP_LINK_NAMES):
          link_names = FINGERTIP_LINK_NAMES
        elif normal_input.shape[0] == len(RIGHT_FINGERTIP_LINK_NAMES):
          link_names = RIGHT_FINGERTIP_LINK_NAMES
        else:
          raise ValueError("direct tactile grids must contain five or ten fingertips")
      sample = {
        "link_names": tuple(link_names),
        "normal_taxel_force_n": normal_input,
        "tangent_taxel_force_n": tangent_taxel_force_n,
      }
    else:
      if link_names is not None:
        raise ValueError("link_names requires direct tactile grids")
      provider = self.tactile_provider
      if provider is None:
        provider = getattr(simulation, "evaluation_tactile_provider", None)
      if provider is None:
        raise ValueError("a tactile provider is required to capture fingertip force grids")
      self._tactile_source = str(getattr(provider, "source", "unspecified"))
      sample = provider.read(data)
    normal, tangent = _canonical_right_sample(sample)
    # This also validates finite, nonnegative normal force and canonical grid
    # dimensions before an invalid sample can enter the saved rollout.
    aggregate_fingertip_forces(normal, tangent)
    self._times.append(current_time)
    self._states.append(state.copy())
    self._normal_grids.append(normal.copy())
    self._tangent_grids.append(tangent.copy())
    return True

  def finish(
    self,
    output_dir: str | Path,
    reference_dataset: str | Path | None = None,
    reference_episode_index: int = 0,
  ) -> dict[str, Any]:
    """Save the full rollout and, if available, three episode comparisons."""
    if self._finished:
      raise RuntimeError("evaluation comparison trace has already been finished")
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    trace_path = output / "evaluation_rollout_trace.npz"
    metadata_path = output / "evaluation_comparison.json"

    times = np.asarray(self._times, dtype=np.float64)
    elapsed = times - times[0] if len(times) else times.copy()
    states = (
      np.stack(self._states)
      if self._states else np.empty((0, 29), dtype=np.float64)
    )
    normal_grids = (
      np.stack(self._normal_grids)
      if self._normal_grids else np.empty((0, 5, 7, 5), dtype=np.float64)
    )
    tangent_grids = (
      np.stack(self._tangent_grids)
      if self._tangent_grids else np.empty((0, 5, 7, 5, 2), dtype=np.float64)
    )
    normal_force, tangent_force = aggregate_fingertip_forces(
      normal_grids, tangent_grids
    )
    np.savez_compressed(
      trace_path,
      time_s=elapsed,
      simulation_time_s=times,
      state_29=states,
      wrist_state=states[:, :9],
      hand_joint_position=states[:, 9:],
      normal_taxel_force_n=normal_grids,
      tangent_taxel_force_n=tangent_grids,
      fingertip_normal_force_n=normal_force,
      fingertip_tangent_force_n=tangent_force,
      joint_names=np.asarray(RIGHT_HAND_ACTUATED_JOINT_NAMES),
      fingertip_link_names=np.asarray(RIGHT_FINGERTIP_LINK_NAMES),
    )

    metadata: dict[str, Any] = {
      "schema": SCHEMA_VERSION,
      "status": "reference_unavailable",
      "rollout": {
        "samples": len(times),
        "source_start_time_s": float(times[0]) if len(times) else None,
        "duration_s": float(elapsed[-1]) if len(elapsed) else 0.0,
        "sampling": "one sample per distinct control-step simulation timestamp",
        "observed_hz": (
          float(1.0 / np.median(np.diff(times))) if len(times) > 1 else None
        ),
      },
      "reference": {
        "dataset_root": str(Path(reference_dataset).expanduser().resolve())
        if reference_dataset is not None else None,
        "output_episode_index": reference_episode_index,
      },
      "state": {
        "layout": "native_right_wrist_xyz_rot6d_then_20_actuated_hand_dof",
        "dimension": 29,
        "wrist_site": NATIVE_RIGHT_WRIST_SITE,
        "rot6d": "first rotation column followed by second rotation column",
        "joint_names": list(RIGHT_HAND_ACTUATED_JOINT_NAMES),
      },
      "tactile": {
        "finger_order": list(FINGER_NAMES),
        "link_names": list(RIGHT_FINGERTIP_LINK_NAMES),
        "taxel_grid": [7, 5],
        "Fn": "sum of all 7x5 normal taxel forces",
        "Ft": "sqrt(sum(Ft_x over 7x5)^2 + sum(Ft_y over 7x5)^2)",
        "unit": "N",
        "source": self._tactile_source,
        "comparison_note": (
          "Reference and rollout tactile providers may differ; compare force "
          "magnitudes only when their calibration is known to match."
        ),
      },
      "styles": {
        "reference": "solid",
        "evaluation_rollout": "dashed",
        "time_axis": "elapsed seconds; each sequence starts at zero",
      },
      "artifacts": {
        "rollout_trace": str(trace_path),
        "metadata": str(metadata_path),
      },
    }

    def save_metadata() -> None:
      metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
      )

    if not len(times):
      metadata["status"] = "no_samples"
      metadata["reason"] = "the rollout contains no control-step samples"
      save_metadata()
      self._finished = True
      return metadata
    if reference_dataset is None:
      metadata["reason"] = "reference_dataset was not provided"
      save_metadata()
      self._finished = True
      return metadata

    try:
      reference = load_openwam_reference(
        reference_dataset, output_episode_index=reference_episode_index
      )
    except Exception as error:
      if _expected_reference_gap(error):
        metadata["reason"] = str(error)
        save_metadata()
        self._finished = True
        return metadata
      metadata["status"] = "error"
      metadata["reason"] = f"{type(error).__name__}: {error}"
      save_metadata()
      raise

    metadata["reference"].update({
      "source_hdf5": str(reference.source_hdf5),
      "samples": len(reference.time_s),
      "duration_s": float(reference.time_s[-1]),
      "alignment": "head camera observations select latest nonfuture state samples",
    })
    try:
      plots = OpenWAMEvaluationPlots(reference, artifact_prefix="evaluation")
      for index, time in enumerate(times):
        plots.capture(
          float(time),
          states[index],
          normal_taxel_force_n=normal_grids[index],
          tangent_taxel_force_n=tangent_grids[index],
        )
      plot_metadata = plots.finish(output)
    except Exception as error:
      metadata["status"] = "error"
      metadata["reason"] = f"{type(error).__name__}: {error}"
      save_metadata()
      raise
    metadata["status"] = "ok"
    metadata["artifacts"].update({
      key: plot_metadata["artifacts"][key]
      for key in (
        "right_wrist_state",
        "right_hand_actuated_dof",
        "right_fingertip_tactile",
      )
    })
    save_metadata()
    self._finished = True
    return metadata


__all__ = ["EvaluationComparisonTrace", "SCHEMA_VERSION"]
