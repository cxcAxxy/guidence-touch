"""Offline pick-place replay with recorded Genesis probe maps and proxy curves.

The original recordings contain no spatial taxel force in newtons.  Probe depth
and instantaneous contact are shown as measured geometric quantities; local
aggregate components remain explicitly labeled as Genesis force proxies.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw

from .config import FINGERTIP_LINK_NAMES
from .evaluation_video import (
  FINGER_LABELS,
  HAND_LABELS,
  _draw_camera,
  _draw_curve,
  _draw_heatmap,
  _font,
)
from .poker_review import (
  _json_attr,
  _probe_video,
  _sha256,
  _text,
  _validate_probe,
  _video_validation_backend,
  select_camera_frames,
)
from .recording import validate_episode
from .task_video import _FfmpegPipeWriter, _find_ffmpeg_executable

SCHEMA = "pickplace-genesis-probe-review-v1"


def _probe_indices(file: h5py.File) -> np.ndarray:
  """Require ten complete row-major 7x5 grids in canonical finger order."""
  group = file["tactile_genesis"]
  names = [_text(value) for value in group["probe_link_names"][:]]
  if len(names) != 350 or set(names) != set(FINGERTIP_LINK_NAMES):
    raise ValueError("Genesis probe layout must contain ten 7x5 fingertip grids")
  indices = []
  for name in FINGERTIP_LINK_NAMES:
    match = np.flatnonzero(np.asarray(names) == name)
    if len(match) != 35 or not np.array_equal(match, np.arange(match[0], match[0] + 35)):
      raise ValueError(f"Genesis probes for {name} are not one complete 7x5 block")
    indices.append(match)
  return np.asarray(indices, dtype=np.intp)


def _check_source(file: h5py.File) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  if "cameras/head" not in file or len(file["cameras"]) != 1:
    raise ValueError("pick-place review expects the recorded head-only camera contract")
  camera = file["cameras/head"]
  times = np.asarray(camera["timestamp"][:], dtype=np.float64)
  state_indices = np.asarray(camera["state_index"][:], dtype=np.int64)
  state_times = np.asarray(file["state/timestamp"][:], dtype=np.float64)
  if (
    times.ndim != 1
    or not len(times)
    or len(times) != len(state_indices)
    or camera["rgb"].shape != (len(times), 240, 320, 3)
    or camera["rgb"].dtype != np.uint8
    or not np.isfinite(times).all()
    or np.any(np.diff(times) <= 0)
    or np.any(state_indices < 0)
    or np.any(state_indices >= len(state_times))
  ):
    raise ValueError("head RGB, capture times, and state indices must be aligned")
  age = times - state_times[state_indices]
  if np.any(age < -1e-8) or np.any(age > 0.020):
    raise ValueError("head RGB must match a nonfuture state within 20 ms")
  group = file["tactile_genesis"]
  if (
    group["probe_depth"].shape != (len(state_times), 350)
    or group["probe_contact_instantaneous"].shape != (len(state_times), 350)
    or group["force_local"].shape != (len(state_times), 10, 3)
  ):
    raise ValueError("Genesis probe/force arrays must align with recorded states")
  link_names = [_text(value) for value in group["link_names"][:]]
  if link_names != list(FINGERTIP_LINK_NAMES):
    raise ValueError("Genesis force_local fingertip order must be canonical")
  return times, state_indices, state_times


def compose_probe_frame(
  head: np.ndarray,
  depth_mm: np.ndarray,
  contact: np.ndarray,
  proxy_history: np.ndarray,
  *,
  width: int,
  height: int,
  depth_max_mm: float,
  proxy_abs_max: float,
  timestamp_s: float,
  phase: str,
) -> Image.Image:
  """Use the USB/Card three-column layout without relabeling proxies as Fn/Ft."""
  if depth_mm.shape != (10, 7, 5) or contact.shape != (10, 7, 5):
    raise ValueError("probe maps must have shape (10,7,5)")
  if proxy_history.ndim != 3 or proxy_history.shape[1:] != (10, 3):
    raise ValueError("Genesis proxy history must have shape (T,10,3)")
  image = Image.new("RGB", (width, height), (14, 15, 19))
  draw = ImageDraw.Draw(image)
  draw.rectangle((0, 0, width, 68), fill=(27, 29, 35))
  draw.text(
    (12, 7),
    f"PICK-PLACE DATASET REPLAY | t={timestamp_s:.3f}s | phase={phase}",
    fill=(246, 246, 249),
    font=_font(22),
  )
  draw.text(
    (12, 38),
    "Genesis depth [mm] + instantaneous contact | local Fx/Fy/Fz proxy = depth x 1e4; NOT Newton force",
    fill=(190, 193, 204),
    font=_font(16),
  )
  margin, top, bottom = 8, 76, height - 8
  left_end, middle_end = int(width * 0.34), int(width * 0.61)
  _draw_camera(
    image,
    draw,
    head,
    (margin, top, left_end - margin, top + int((bottom - top) * 0.68)),
    "HEAD / RECORDED",
  )
  note_top = top + int((bottom - top) * 0.68) + 8
  draw.rectangle(
    (margin, note_top, left_end - margin, bottom),
    outline=(58, 61, 72),
    width=1,
  )
  for index, line in enumerate(
    (
      "SOURCE: saved HDF5 trajectory",
      "Camera: head only (no wrist/global RGB)",
      "Spatial maps: Genesis probe geometry",
      "Curves: recorded per-finger local proxy",
      "Not solver-contact taxel force in N",
    )
  ):
    draw.text(
      (margin + 10, note_top + 12 + index * 28),
      line,
      fill=(205, 207, 216),
      font=_font(15),
    )
  tactile_left, tactile_right = left_end + margin, middle_end - margin
  tactile_row_height = (bottom - top) // 4
  for side_index, hand in enumerate(HAND_LABELS):
    offset = side_index * 5
    rows = (
      (depth_mm[offset : offset + 5], depth_max_mm, f"{hand[0]} D"),
      (contact[offset : offset + 5], 1.0, f"{hand[0]} C"),
    )
    for quantity_index, (maps, maximum, label) in enumerate(rows):
      row = side_index * 2 + quantity_index
      row_top = top + row * tactile_row_height
      row_bottom = bottom if row == 3 else row_top + tactile_row_height - 4
      cell_width = (tactile_right - tactile_left) // 5
      for finger_index, finger in enumerate(FINGER_LABELS):
        cell_left = tactile_left + finger_index * cell_width
        cell_right = tactile_right if finger_index == 4 else cell_left + cell_width - 2
        _draw_heatmap(
          image,
          draw,
          maps[finger_index],
          (cell_left, row_top, cell_right, row_bottom),
          f"{label} {finger[:2]}",
          maximum,
        )
  curve_left, curve_right = middle_end + margin, width - margin
  curve_width = (curve_right - curve_left - 5) // 2
  curve_height = (bottom - top - 16) // 5
  for finger_index, finger in enumerate(FINGER_LABELS):
    row_top = top + finger_index * (curve_height + 4)
    for side_index, hand in enumerate(("L", "R")):
      box_left = curve_left + side_index * (curve_width + 5)
      _draw_curve(
        draw,
        proxy_history,
        side_index * 5 + finger_index,
        (box_left, row_top, box_left + curve_width, row_top + curve_height),
        f"{hand} {finger}",
        proxy_abs_max,
        channel_names=("Fx", "Fy", "Fz"),
      )
  return image


def export_pickplace_probe_review(
  source: str | Path,
  output_dir: str | Path,
  *,
  fps: int = 10,
  width: int = 1920,
  height: int = 1080,
) -> dict:
  """Export a new immutable review directory from recorded RGB and Genesis probes."""
  source = Path(source).expanduser().resolve()
  output = Path(output_dir).expanduser().absolute()
  partial = output.with_name(output.name + ".partial")
  if output.exists() or output.is_symlink() or partial.exists() or partial.is_symlink():
    raise FileExistsError("review or partial output already exists; choose a new path")
  if source.suffix != ".h5" or not source.is_file():
    raise ValueError("source must be a completed HDF5 episode")
  if width < 960 or height < 540 or width % 2 or height % 2:
    raise ValueError("review dimensions must be even and at least 960 by 540")
  if fps not in (5, 10):
    raise ValueError("fps must be 5 or 10")
  ffmpeg = _find_ffmpeg_executable()
  backend = _video_validation_backend(ffmpeg)
  validation = validate_episode(source)
  if not validation.valid:
    raise ValueError(f"source validation failed: {validation.errors}")
  source_sha256 = _sha256(source)
  with h5py.File(source, "r") as file:
    metadata, outcome = _json_attr(file, "metadata_json"), _json_attr(file, "outcome_json")
    if metadata.get("scene") != "pick-place":
      raise ValueError("source must be a pick-place episode")
    camera_times, state_indices, state_times = _check_source(file)
    probe_indices = _probe_indices(file)
    selected = select_camera_frames(camera_times, fps)
    group = file["tactile_genesis"]
    depth_max_mm = max(
      1.0, math.ceil(float(np.max(group["probe_depth"][:])) * 1000.0)
    )
    proxy_abs_max = max(
      1.0, math.ceil(float(np.max(np.abs(group["force_local"][:]))) / 10.0) * 10.0
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    partial.mkdir()
    writer = None
    history: list[np.ndarray] = []
    last_image = None
    try:
      writer = _FfmpegPipeWriter(
        partial / "review.mp4", fps=fps, width=width, height=height, executable=ffmpeg
      )
      with (partial / "frames.jsonl").open("x", encoding="utf-8") as log:
        for output_index, camera_index in enumerate(selected):
          state_index = int(state_indices[camera_index])
          depth = np.asarray(group["probe_depth"][state_index])[probe_indices].reshape(10, 7, 5)
          contact = np.asarray(group["probe_contact_instantaneous"][state_index])[
            probe_indices
          ].reshape(10, 7, 5)
          proxy = np.asarray(group["force_local"][state_index], dtype=np.float64)
          if not np.isfinite(depth).all() or not np.isfinite(proxy).all() or np.any(depth < 0):
            raise ValueError("probe data must be finite with nonnegative depth")
          history.append(proxy)
          phase = _text(file["commands/phase"][state_index])
          timestamp_s = float(camera_times[camera_index])
          image = compose_probe_frame(
            np.asarray(file["cameras/head/rgb"][camera_index]),
            depth * 1000.0,
            contact.astype(np.float64),
            np.asarray(history),
            width=width,
            height=height,
            depth_max_mm=depth_max_mm,
            proxy_abs_max=proxy_abs_max,
            timestamp_s=timestamp_s,
            phase=phase,
          )
          writer.write(np.asarray(image))
          if output_index == 0:
            image.save(partial / "first_frame.png")
          last_image = image
          log.write(
            json.dumps(
              {
                "output_index": output_index,
                "camera_index": int(camera_index),
                "camera_timestamp_s": timestamp_s,
                "state_index": state_index,
                "state_timestamp_s": float(state_times[state_index]),
                "phase": phase,
                "fingertip_order": list(FINGERTIP_LINK_NAMES),
                "probe_depth_mm": (depth * 1000.0).tolist(),
                "probe_contact_instantaneous": contact.tolist(),
                "genesis_force_local_proxy": proxy.tolist(),
              },
              ensure_ascii=False,
            ) + "\n"
          )
      writer.finish()
      assert last_image is not None
      last_image.save(partial / "last_frame.png")
      probe = _probe_video(partial / "review.mp4", backend=backend)
      _validate_probe(probe, count=len(selected), fps=fps, width=width, height=height)
      report = {
        "schema_version": SCHEMA,
        "completed": True,
        "task": "pick-place",
        "source_hdf5": str(source),
        "source_sha256": source_sha256,
        "exporter_source_sha256": _sha256(Path(__file__)),
        "source_episode_index": metadata.get("episode_index"),
        "source_outcome_success": outcome.get("success"),
        "source_validation": asdict(validation),
        "camera_names": ["head"],
        "fingertip_order": list(FINGERTIP_LINK_NAMES),
        "heatmaps": {
          "probe_depth_mm": {"source": "/tactile_genesis/probe_depth", "maximum": depth_max_mm},
          "instantaneous_contact": {"source": "/tactile_genesis/probe_contact_instantaneous", "maximum": 1.0},
        },
        "curve_channels": ["local Fx", "local Fy", "local Fz"],
        "curve_source": "/tactile_genesis/force_local",
        "curve_unit": "Genesis proxy: sum(probe_depth_m * probe_local_normal * 1e4); not newtons",
        "curve_abs_max": proxy_abs_max,
        "no_solver_contact_taxel_force": True,
        "frame_alignment": "camera/state_index selects the recorded nonfuture state and Genesis probe sample",
        "fps": fps,
        "output_size": [width, height],
        "source_camera_frame_count": len(camera_times),
        "output_frame_count": len(selected),
        "first_camera_timestamp_s": float(camera_times[selected[0]]),
        "last_camera_timestamp_s": float(camera_times[selected[-1]]),
        "video_validation": probe,
      }
      (partial / "review.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
      )
      if output.exists() or output.is_symlink():
        raise FileExistsError("output appeared during export; partial retained")
      partial.rename(output)
      return report
    except BaseException as error:
      if writer is not None:
        try:
          writer.abort()
        except BaseException as abort_error:
          error.add_note(f"encoder cleanup failed: {abort_error}")
      (partial / "failure.json").write_text(
        json.dumps(
          {"completed": False, "source_hdf5": str(source), "error": str(error)},
          indent=2,
        ) + "\n",
        encoding="utf-8",
      )
      raise


__all__ = ["SCHEMA", "compose_probe_frame", "export_pickplace_probe_review"]
