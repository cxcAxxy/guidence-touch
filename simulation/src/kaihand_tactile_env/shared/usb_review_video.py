"""Offline USB/Card replay video with bilateral spatial tactile evidence."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np

from .config import FINGERTIP_LINK_NAMES
from .evaluation_video import (
  CHANNEL_NAMES,
  NORMAL_TAXEL_MAX_N,
  TANGENT_TAXEL_MAX_N,
  compose_evaluation_frame,
)
from .poker_review import (
  _FORCE_GROUP,
  _NORMAL,
  _TANGENT,
  _json_attr,
  _probe_video,
  _sha256,
  _text,
  _validate_probe,
  _video_validation_backend,
  plan_review_frames,
)
from .recording import validate_episode
from .task_video import _FfmpegPipeWriter, _find_ffmpeg_executable

REVIEW_SCHEMA = "usb-offline-tactile-review-v2"


def _force_layout(file: h5py.File) -> np.ndarray:
  group = file[_FORCE_GROUP]
  names = [_text(value) for value in group["link_names"][:]]
  if len(names) != len(set(names)) or not set(FINGERTIP_LINK_NAMES).issubset(names):
    raise ValueError("contact forces must identify all ten canonical fingertips")
  count = len(group["timestamp"])
  if group[_NORMAL].shape != (count, len(names), 7, 5):
    raise ValueError("normal contact-force grids must have shape [T,finger,7,5]")
  if group[_TANGENT].shape != (count, len(names), 7, 5, 2):
    raise ValueError(
      "signed tangential contact-force grids must have shape [T,finger,7,5,2]"
    )
  return np.asarray(
    [names.index(name) for name in FINGERTIP_LINK_NAMES], dtype=np.intp
  )


def _force_means(normal: np.ndarray, tangent: np.ndarray) -> np.ndarray:
  """Return canonical Ft_col/Ft_row/Fn means for all ten fingertips."""
  return np.column_stack(
    (
      tangent[..., 0].mean(axis=(1, 2)),
      tangent[..., 1].mean(axis=(1, 2)),
      normal.mean(axis=(1, 2)),
    )
  )


def export_usb_review_video(
  source: str | Path,
  output_dir: str | Path,
  *,
  fps: float = 10,
  width: int = 1920,
  height: int = 1080,
  tolerance_s: float = 0.020,
  source_scene: str = "usb-insert",
) -> dict:
  """Export synchronized head/wrist RGB and enlarged tactile heatmaps."""
  if source_scene not in ("usb-insert", "poker-draw"):
    raise ValueError("bilateral tactile review supports USB or Card episodes")
  source = Path(source).expanduser().resolve()
  output = Path(output_dir).expanduser().absolute()
  partial = output.with_name(output.name + ".partial")
  if output.exists() or output.is_symlink() or partial.exists() or partial.is_symlink():
    raise FileExistsError(
      "review output or partial directory already exists; choose a new output directory"
    )
  if source.suffix != ".h5" or not source.is_file():
    raise ValueError("review source must be a completed .h5 episode")
  if any(
    path.exists()
    for path in (
      source.with_suffix(".h5.partial"),
      source.with_suffix(".h5.lock"),
      source.with_suffix(".failure.json"),
    )
  ):
    raise ValueError("source has partial, active, or failure artifacts")
  if width < 960 or height < 540 or width % 2 or height % 2:
    raise ValueError("review dimensions must be even and at least 960 by 540")

  ffmpeg_executable = _find_ffmpeg_executable()
  video_backend = _video_validation_backend(ffmpeg_executable)
  validation = validate_episode(source)
  if not validation.valid:
    raise ValueError(f"source HDF5 failed validation: {validation.errors}")
  source_digest = _sha256(source)

  with h5py.File(source, "r") as file:
    metadata = _json_attr(file, "metadata_json")
    outcome = _json_attr(file, "outcome_json")
    if metadata.get("scene") != source_scene:
      raise ValueError(f"review export requires a {source_scene} episode")
    if source_scene == "poker-draw" and outcome.get("success") is not True:
      raise ValueError("Card review export requires a successful episode")
    frames = plan_review_frames(
      file,
      fps=fps,
      tolerance_s=tolerance_s,
      second_camera="right_wrist",
    )
    force_indices = _force_layout(file)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial.mkdir()
    writer = None
    last_image = None
    history: list[np.ndarray] = []
    try:
      writer = _FfmpegPipeWriter(
        partial / "review.mp4",
        fps=fps,
        width=width,
        height=height,
        executable=ffmpeg_executable,
      )
      with (partial / "frames.jsonl").open("x", encoding="utf-8") as log:
        for frame in frames:
          head = np.asarray(file["cameras/head/rgb"][frame.camera_index])
          wrist = np.asarray(file["cameras/right_wrist/rgb"][frame.camera_index])
          group = file[_FORCE_GROUP]
          normal = np.asarray(group[_NORMAL][frame.tactile_index])[force_indices]
          tangent = np.asarray(group[_TANGENT][frame.tactile_index])[force_indices]
          means = _force_means(normal, tangent)
          history.append(means)
          phase = _text(file["commands/phase"][frame.tactile_index])
          composite = compose_evaluation_frame(
            head,
            wrist,
            normal,
            tangent,
            np.asarray(history),
            width=width,
            height=height,
            simulation_time_s=frame.camera_pose_timestamp_s,
            phase=phase,
            second_camera_label="RIGHT WRIST / MODEL VIEW",
            heading=(
              "CARD DATASET REPLAY"
              if source_scene == "poker-draw"
              else "USB DATASET REPLAY"
            ),
          )
          writer.write(np.asarray(composite))
          if frame.output_index == 0:
            composite.save(partial / "first_frame.png")
          last_image = composite
          log.write(
            json.dumps(
              {
                **asdict(frame),
                "phase": phase,
                "fingertip_order": list(FINGERTIP_LINK_NAMES),
                "force_mean_n_per_taxel": means.tolist(),
              },
              ensure_ascii=False,
            )
            + "\n"
          )
      writer.finish()
      if last_image is None:
        raise ValueError("review frame plan is empty")
      last_image.save(partial / "last_frame.png")
      probe = _probe_video(partial / "review.mp4", backend=video_backend)
      _validate_probe(probe, count=len(frames), fps=fps, width=width, height=height)
      summary = {
        "schema_version": (
          "card-offline-tactile-review-v2"
          if source_scene == "poker-draw"
          else REVIEW_SCHEMA
        ),
        "completed": True,
        "task": source_scene,
        "source_hdf5": str(source),
        "source_sha256": source_digest,
        "exporter_source_sha256": _sha256(Path(__file__)),
        "source_episode_index": metadata.get("episode_index"),
        "source_seed": metadata.get("seed"),
        "source_outcome_success": outcome.get("success"),
        "source_validation": asdict(validation),
        "camera_names": ["head", "right_wrist"],
        "fingertip_order": list(FINGERTIP_LINK_NAMES),
        "time_series_displayed": False,
        "force_mean_channel_order": list(CHANNEL_NAMES),
        "force_mean_value": "arithmetic mean across each fingertip's 7x5 taxels",
        "force_mean_storage": "frames.jsonl.force_mean_n_per_taxel",
        "normal_color_scale_n_per_taxel": [0, NORMAL_TAXEL_MAX_N],
        "tangent_color_scale_n_per_taxel": [0, TANGENT_TAXEL_MAX_N],
        "layout": "enlarged head/right-wrist RGB; enlarged bilateral Fn/|Ft| maps",
        "force_sources": {
          "normal": f"/{_FORCE_GROUP}/{_NORMAL}",
          "signed_tangent": f"/{_FORCE_GROUP}/{_TANGENT}",
        },
        "force_visualization": (
          "heatmaps show Fn and per-taxel |Ft|; signed Ft_col/Ft_row and "
          "nonnegative Fn means remain in frames.jsonl"
        ),
        "source_camera_frame_count": len(file["cameras/head/rgb"]),
        "output_frame_count": len(frames),
        "fps": fps,
        "output_size": [width, height],
        "first_camera_pose_timestamp_s": frames[0].camera_pose_timestamp_s,
        "last_camera_pose_timestamp_s": frames[-1].camera_pose_timestamp_s,
        "last_tactile_timestamp_s": frames[-1].tactile_timestamp_s,
        "maximum_tactile_age_s": max(frame.tactile_age_s for frame in frames),
        "constant_fps_duration_s": len(frames) / fps,
        "video_validation": probe,
      }
      (partial / "review.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
      )
      if output.exists() or output.is_symlink():
        raise FileExistsError("review output appeared while exporting; partial retained")
      partial.rename(output)
      return summary
    except BaseException as error:
      if writer is not None:
        try:
          writer.abort()
        except BaseException as abort_error:
          error.add_note(f"review encoder cleanup failed: {abort_error}")
      try:
        (partial / "failure.json").write_text(
          json.dumps(
            {
              "completed": False,
              "source_hdf5": str(source),
              "error_type": type(error).__name__,
              "error": str(error),
            },
            indent=2,
          )
          + "\n",
          encoding="utf-8",
        )
      except BaseException as report_error:
        error.add_note(f"review failure report could not be written: {report_error}")
      raise


def export_card_review_video(
  source: str | Path,
  output_dir: str | Path,
  *,
  fps: float = 10,
  width: int = 1920,
  height: int = 1080,
  tolerance_s: float = 0.020,
) -> dict:
  """Export a successful Card episode with bilateral tactile video and curves."""
  return export_usb_review_video(
    source,
    output_dir,
    fps=fps,
    width=width,
    height=height,
    tolerance_s=tolerance_s,
    source_scene="poker-draw",
  )


__all__ = ["REVIEW_SCHEMA", "export_usb_review_video", "export_card_review_video"]
