"""Offline review exports from saved camera RGB and recorded contact-force grids.

No model, renderer, or simulated motion is created. Image and video frames are
composed one at a time; signed tangential forces remain in both HDF5 and NPZ.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw

from .recording import validate_episode
from .tactile import RIGHT_FINGERTIP_LINK_NAMES
from .task_video import (
  _FINGER_LABELS,
  _HEATMAP_LUT,
  _MAX_NORMAL_TAXEL_FORCE_N,
  _MAX_TANGENT_TAXEL_FORCE_N,
  _draw_force_tactile_cell,
  _FfmpegPipeWriter,
  _find_ffmpeg_executable,
  _paste_fit,
)

REVIEW_SCHEMA = "poker-offline-review-v1"
_FORCE_GROUP = "tactile_contact_force"
_NORMAL = "normal_taxel_force_n"
_TANGENT = "tangent_taxel_force_n"


def review_cameras(file, second_camera: str | None = None) -> tuple[str, str]:
  """Prefer the new wrist stream; never relabel a legacy overhead image."""
  if second_camera is None:
    second_camera = "right_wrist" if "cameras/right_wrist" in file else "overhead"
  if second_camera not in ("right_wrist", "overhead", "front"):
    raise ValueError("second review camera must be right_wrist, overhead or front")
  for name in ("head", second_camera):
    if f"cameras/{name}" not in file:
      raise ValueError(f"saved source is missing requested camera {name!r}")
  return "head", second_camera


@dataclass(frozen=True)
class ReviewFrame:
  output_index: int
  camera_index: int
  camera_timestamp_s: float
  camera_pose_timestamp_s: float
  tactile_index: int
  tactile_timestamp_s: float
  tactile_age_s: float
  playback_timestamp_s: float
  playback_time_error_s: float


def _text(value) -> str:
  return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _json_attr(group, name):
  value = json.loads(_text(group.attrs.get(name, "{}")))
  if not isinstance(value, dict):
    raise ValueError(f"{name} must contain a JSON object")
  return value


def _clock(values, name: str, *, strict: bool) -> tuple[np.ndarray, np.ndarray]:
  times = np.asarray(values, dtype=float)
  if times.ndim != 1 or not len(times) or not np.isfinite(times).all():
    raise ValueError(f"{name} must be a nonempty finite timestamp vector")
  if np.any(times < 0) or np.max(times) > 1e8:
    raise ValueError(f"{name} contains timestamps outside the supported range")
  ns = np.rint(times * 1e9).astype(np.int64)
  if np.any(np.diff(ns) <= 0 if strict else np.diff(ns) < 0):
    raise ValueError(f"{name} timestamps are not ordered")
  return times, ns


def select_camera_frames(camera_times: np.ndarray, fps: float) -> np.ndarray:
  """Choose distinct existing RGB frames nearest the playback grid, including last."""
  times, _ = _clock(camera_times, "camera", strict=True)
  if not math.isfinite(fps) or fps <= 0:
    raise ValueError("video fps must be finite and positive")
  targets = (
    times[0] + np.arange(int(math.floor((times[-1] - times[0]) * fps + 1e-8)) + 1) / fps
  )
  right = np.searchsorted(times, targets, side="left").clip(0, len(times) - 1)
  left = np.maximum(0, right - 1)
  selected = np.where(
    np.abs(times[left] - targets) <= np.abs(times[right] - targets), left, right
  )
  return np.unique(np.r_[0, selected, len(times) - 1]).astype(np.int64)


def plan_review_frames(
  file,
  *,
  fps: float = 10,
  tolerance_s: float = 0.020,
  second_camera: str | None = None,
):
  """Synchronize each camera independently against causal solver-force samples."""
  if not math.isfinite(tolerance_s) or not 0 <= tolerance_s <= 0.020:
    raise ValueError("tactile matching tolerance must be between 0 and 20 ms")
  if not math.isfinite(fps) or not 0 < fps <= float(file.attrs.get("camera_hz", 0)):
    raise ValueError(
      "review fps must be positive and cannot exceed the saved camera rate"
    )
  force_times, force_ns = _clock(
    file[f"{_FORCE_GROUP}/timestamp"][:], "tactile", strict=False
  )
  clocks, poses, matches = {}, {}, {}
  cameras = review_cameras(file, second_camera)
  secondary = cameras[1]
  for camera in cameras:
    group = file[f"cameras/{camera}"]
    times, time_ns = _clock(group["timestamp"][:], f"{camera} camera", strict=True)
    pose_times, pose_ns = _clock(
      group["pose_timestamp"][:], f"{camera} pose", strict=True
    )
    rgb = group["rgb"]
    if rgb.ndim != 4 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
      raise ValueError(f"{camera} RGB must be an original uint8 [T,H,W,3] stream")
    if len(times) != len(pose_times) or rgb.shape[0] != len(times):
      raise ValueError(f"{camera} camera clock and RGB sample counts differ")
    if np.any(pose_ns > time_ns):
      raise ValueError(
        f"{camera} pose timestamp cannot follow its saved camera timestamp"
      )
    index = np.searchsorted(force_ns, pose_ns, side="right") - 1
    if np.any(index < 0):
      raise ValueError(f"{camera} frame has no nonfuture tactile sample")
    age_ns = pose_ns - force_ns[index]
    if np.any(age_ns > int(round(tolerance_s * 1e9))):
      raise ValueError(
        f"{camera} tactile sample is more than {tolerance_s * 1000:g} ms old"
      )
    clocks[camera], poses[camera], matches[camera] = times, pose_times, index
  if not np.array_equal(clocks["head"], clocks[secondary]) or not np.array_equal(
    poses["head"], poses[secondary]
  ):
    raise ValueError(
      f"head and {secondary} frames must have identical capture and pose times"
    )
  if not np.array_equal(matches["head"], matches[secondary]):
    raise ValueError(f"head and {secondary} do not select the same tactile frame")
  indices = select_camera_frames(clocks["head"], fps)
  start = float(poses["head"][indices[0]])
  result = []
  for output_index, index in enumerate(indices):
    tactile_index = int(matches["head"][index])
    pose_time = float(poses["head"][index])
    force_time = float(force_times[tactile_index])
    playback = output_index / fps
    result.append(
      ReviewFrame(
        output_index,
        int(index),
        float(clocks["head"][index]),
        pose_time,
        tactile_index,
        force_time,
        pose_time - force_time,
        playback,
        playback - (pose_time - start),
      )
    )
  return tuple(result)


def _force_layout(file):
  group = file[_FORCE_GROUP]
  names = [_text(value) for value in group["link_names"][:]]
  if len(names) != len(set(names)) or not set(RIGHT_FINGERTIP_LINK_NAMES).issubset(
    names
  ):
    raise ValueError("contact forces must identify all five distinct right fingertips")
  count = len(group["timestamp"])
  if group[_NORMAL].shape != (count, len(names), 7, 5):
    raise ValueError("normal contact-force grids must have shape [T,finger,7,5]")
  if group[_TANGENT].shape != (count, len(names), 7, 5, 2):
    raise ValueError("signed tangential grids must have shape [T,finger,7,5,2]")
  return np.array([names.index(name) for name in RIGHT_FINGERTIP_LINK_NAMES])


def force_panel(
  normal: np.ndarray,
  tangent: np.ndarray,
  *,
  width: int,
  height: int,
  quantity: str,
  scale_n: float | None = None,
) -> Image.Image:
  """Use the existing task-video colors and taxel orientation, with fixed scales."""
  if normal.shape != (5, 7, 5) or tangent.shape != (5, 7, 5, 2):
    raise ValueError("review panels require five 7-by-5 right-finger force grids")
  if (
    not np.isfinite(normal).all()
    or not np.isfinite(tangent).all()
    or np.any(normal < -1e-12)
  ):
    raise ValueError("contact-force grids must be finite with nonnegative normal force")
  if quantity == "normal":
    maps, totals = normal, normal.sum(axis=(1, 2))
    maximum, short = scale_n if scale_n is not None else _MAX_NORMAL_TAXEL_FORCE_N, "Fn"
    title = f"RIGHT 5 | NORMAL Fn [N/taxel] | fixed 0..{maximum:.2f}"
  elif quantity == "tangent":
    maps = np.linalg.norm(tangent, axis=-1)
    totals = np.linalg.norm(tangent.sum(axis=(1, 2)), axis=-1)
    maximum, short = (
      scale_n if scale_n is not None else _MAX_TANGENT_TAXEL_FORCE_N,
      "Ft",
    )
    title = f"RIGHT 5 | TANGENT |Ft| [N/taxel] | fixed 0..{maximum:.2f}"
  else:
    raise ValueError("force quantity must be normal or tangent")
  if not math.isfinite(maximum) or maximum <= 0:
    raise ValueError("force panel scale must be finite and positive")
  image = Image.new("RGB", (width, height), (15, 16, 20))
  draw = ImageDraw.Draw(image)
  draw.text((6, 5), title, fill=(235, 235, 240))
  for index, label in enumerate(_FINGER_LABELS):
    _draw_force_tactile_cell(
      image,
      draw,
      maps[index],
      float(totals[index]),
      (index * width // 5, 24, (index + 1) * width // 5 - 1, height - 38),
      label,
      quantity=short,
      scale_maximum=maximum,
    )
  bar_left, bar_right = 8, max(9, width - 150)
  bar = Image.fromarray(_HEATMAP_LUT[None, :, :]).resize(
    (bar_right - bar_left, 9), Image.Resampling.NEAREST
  )
  image.paste(bar, (bar_left, height - 29))
  draw.text((bar_left, height - 16), "0 N/taxel", fill=(205, 205, 215))
  draw.text(
    (max(bar_left + 85, bar_right - 100), height - 16),
    f"{maximum:g} N/taxel",
    fill=(205, 205, 215),
  )
  saturated = int(np.count_nonzero(maps > maximum))
  draw.text(
    (bar_right + 8, height - 29), f"SAT taxels: {saturated}", fill=(205, 205, 215)
  )
  return image


def compose_review_frame(
  head,
  overhead,
  normal,
  tangent,
  *,
  frame: ReviewFrame,
  width: int,
  height: int,
  phase: str,
  heading: str = "SAVED SUCCESS",
  caption: str = "Recorded RGB + solver contact spatial estimates | signed Ft preserved in H5/NPZ",
  second_camera_label: str = "OVERHEAD",
  normal_scale_n: float | None = None,
  tangent_scale_n: float | None = None,
):
  """Put head/secondary RGB on the left and normal/tangent on the right."""
  image = Image.new("RGB", (width, height), (15, 16, 20))
  draw = ImageDraw.Draw(image)
  draw.rectangle((0, 0, width, 37), fill=(28, 30, 36))
  draw.text(
    (8, 6),
    f"{heading} | image t={frame.camera_pose_timestamp_s:.3f}s | force t={frame.tactile_timestamp_s:.3f}s | {phase}",
    fill=(240, 240, 245),
  )
  draw.text(
    (8, 21),
    caption,
    fill=(185, 188, 200),
  )
  left_width = width * 2 // 5
  half = (height - 46) // 2
  for index, (name, rgb) in enumerate(
    (("HEAD", head), (second_camera_label, overhead))
  ):
    top = 42 + index * (half + 4)
    box = (4, top, left_width - 4, top + half)
    _paste_fit(image, rgb, box, Image.Resampling.BILINEAR)
    draw.rectangle(box, outline=(72, 76, 86), width=1)
    draw.rectangle((box[0], box[1], box[0] + 76, box[1] + 16), fill=(0, 0, 0))
    draw.text((box[0] + 4, box[1] + 2), name, fill=(240, 240, 240))
  panels = {}
  for index, quantity in enumerate(("normal", "tangent")):
    panels[quantity] = force_panel(
      normal,
      tangent,
      width=width - left_width - 8,
      height=half,
      quantity=quantity,
      scale_n=normal_scale_n if quantity == "normal" else tangent_scale_n,
    )
    image.paste(panels[quantity], (left_width + 4, 42 + index * (half + 4)))
  return image, panels


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def _video_validation_backend(ffmpeg_executable: str | None = None) -> tuple[str, str]:
  """Resolve validation dependencies before starting any export or encoding."""
  executable = shutil.which("ffprobe")
  if executable is not None:
    return "ffprobe", executable
  return "ffmpeg_framemd5_full_decode", ffmpeg_executable or _find_ffmpeg_executable()


def _parse_decoded_frames(lines, *, executable: str) -> dict:
  """Read frame checksums/PTS incrementally; no decoded pixel buffers are retained."""
  time_base = dimensions = period = first_pts = previous_end = None
  count = 0
  for raw_line in lines:
    line = _text(raw_line).strip()
    if line.startswith("#"):
      if match := re.fullmatch(r"#tb\s+0:\s*(\d+/\d+)", line):
        time_base = Fraction(match[1])
      elif match := re.fullmatch(r"#dimensions\s+0:\s*(\d+)x(\d+)", line):
        dimensions = int(match[1]), int(match[2])
      continue
    if not line:
      continue
    fields = [field.strip() for field in line.split(",")]
    if (
      len(fields) != 6
      or fields[0] != "0"
      or not re.fullmatch(r"[0-9a-fA-F]{32}", fields[5])
    ):
      raise ValueError("ffmpeg decoded-frame checksum record is malformed")
    if time_base is None or time_base <= 0 or dimensions is None:
      raise ValueError(
        "ffmpeg decoded-frame stream lacks a valid time base or dimensions"
      )
    _, _, pts, duration, size = map(int, fields[:5])
    if duration <= 0 or size != dimensions[0] * dimensions[1] * 3:
      raise ValueError("ffmpeg decoded-frame duration or RGB size is invalid")
    if period is None:
      first_pts, period = pts, duration
    if duration != period or (previous_end is not None and pts != previous_end):
      raise ValueError(
        "ffmpeg decoded-frame timestamps are not continuous constant-fps video"
      )
    previous_end = pts + duration
    count += 1
  if not count:
    raise ValueError("ffmpeg full decode yielded no complete video frames")
  return {
    "backend": "ffmpeg_framemd5_full_decode",
    "executable": executable,
    "full_decode": True,
    "decoder_threads": 1,
    "frame_count": count,
    "duration_s": float((previous_end - first_pts) * time_base),
    "fps": float(1 / (period * time_base)),
    "width": dimensions[0],
    "height": dimensions[1],
    "decoded_time_base": str(time_base),
    "duration_basis": "last decoded frame PTS plus duration minus first decoded frame PTS",
  }


def _decode_video_with_ffmpeg(path: Path, executable: str) -> dict:
  """Require a successful complete decode, retaining only a small checksum log on disk."""
  command = [
    executable,
    "-hide_banner",
    "-loglevel",
    "error",
    "-nostdin",
    "-xerror",
    "-err_detect",
    "explode",
    "-threads",
    "1",
    "-i",
    str(path),
    "-map",
    "0:v:0",
    "-an",
    "-sn",
    "-dn",
    "-vsync",
    "0",
    "-enc_time_base",
    "-1",
    "-c:v",
    "rawvideo",
    "-threads",
    "1",
    "-filter_threads",
    "1",
    "-pix_fmt",
    "rgb24",
    "-f",
    "framemd5",
    "pipe:1",
  ]
  with tempfile.TemporaryFile() as checksums, tempfile.TemporaryFile() as errors:
    try:
      subprocess.run(command, stdout=checksums, stderr=errors, check=True, timeout=60)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
      errors.seek(0, 2)
      errors.seek(max(0, errors.tell() - 65536))
      detail = errors.read().decode("utf-8", errors="replace").strip()
      raise RuntimeError(
        f"ffmpeg full video decode failed: {detail or error}"
      ) from error
    checksums.seek(0)
    return _parse_decoded_frames(checksums, executable=executable)


def _probe_video(path: Path, *, backend: tuple[str, str] | None = None) -> dict:
  name, executable = backend or _video_validation_backend()
  if name == "ffmpeg_framemd5_full_decode":
    return _decode_video_with_ffmpeg(path, executable)
  if name != "ffprobe":
    raise ValueError(f"unknown review video verification backend: {name}")
  process = subprocess.run(
    [
      executable,
      "-v",
      "error",
      "-threads",
      "1",
      "-select_streams",
      "v:0",
      "-count_frames",
      "-show_entries",
      "stream=nb_read_frames,duration,avg_frame_rate,width,height:format=duration",
      "-of",
      "json",
      str(path),
    ],
    check=True,
    capture_output=True,
    text=True,
    timeout=60,
  )
  payload = json.loads(process.stdout)
  streams = payload.get("streams", [])
  if len(streams) != 1:
    raise ValueError("review video must contain one readable video stream")
  stream = streams[0]
  return {
    "backend": "ffprobe",
    "executable": executable,
    "frame_count": int(stream["nb_read_frames"]),
    "duration_s": float(
      stream["duration"] if "duration" in stream else payload["format"]["duration"]
    ),
    "fps": float(Fraction(stream["avg_frame_rate"])),
    "width": int(stream["width"]),
    "height": int(stream["height"]),
  }


def _validate_probe(probe, *, count, fps, width, height):
  backend = probe.get("backend", "video verification")
  if (
    probe["frame_count"] != count
    or probe["width"] != width
    or probe["height"] != height
  ):
    raise ValueError(
      f"{backend} frame count or dimensions differ from the exported frames"
    )
  if not math.isclose(probe["fps"], fps, abs_tol=1e-6, rel_tol=0):
    raise ValueError(f"{backend} playback frame rate differs from the requested fps")
  if (
    not math.isfinite(probe["duration_s"])
    or abs(probe["duration_s"] - count / fps) > 0.002
  ):
    raise ValueError(f"{backend} duration differs from constant-fps playback duration")


def export_poker_review(
  source: str | Path,
  output_dir: str | Path,
  *,
  fps: float = 10,
  width: int = 1280,
  height: int = 720,
  tolerance_s: float = 0.020,
  second_camera: str | None = None,
) -> dict:
  """Export once into a new directory; failures retain an explicitly partial directory."""
  source, output = (
    Path(source).expanduser().resolve(),
    Path(output_dir).expanduser().absolute(),
  )
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
  with h5py.File(source, "r") as file:
    metadata, outcome = (
      _json_attr(file, "metadata_json"),
      _json_attr(file, "outcome_json"),
    )
    if metadata.get("scene") != "poker-draw" or outcome.get("success") is not True:
      raise ValueError("review export requires a successful poker-draw episode")
    cameras = review_cameras(file, second_camera)
    secondary = cameras[1]
    frames = plan_review_frames(
      file, fps=fps, tolerance_s=tolerance_s, second_camera=secondary
    )
    right = _force_layout(file)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial.mkdir()
    writer = None
    try:
      for modality in (*cameras, "normal", "tangent", "composite"):
        (partial / "frames" / modality).mkdir(parents=True)
      (partial / "raw").mkdir()
      writer = _FfmpegPipeWriter(
        partial / "review.mp4",
        fps=fps,
        width=width,
        height=height,
        executable=ffmpeg_executable,
      )
      with (partial / "frames.csv").open(
        "w", encoding="utf-8", newline=""
      ) as csv_stream:
        table = csv.DictWriter(csv_stream, fieldnames=[*asdict(frames[0]), "phase"])
        table.writeheader()
        for frame in frames:
          rgb = {
            name: np.asarray(file[f"cameras/{name}/rgb"][frame.camera_index])
            for name in cameras
          }
          group = file[_FORCE_GROUP]
          normal = np.asarray(group[_NORMAL][frame.tactile_index])[right]
          tangent = np.asarray(group[_TANGENT][frame.tactile_index])[right]
          phase = _text(file["commands/phase"][frame.tactile_index])
          composite, panels = compose_review_frame(
            rgb["head"],
            rgb[secondary],
            normal,
            tangent,
            frame=frame,
            width=width,
            height=height,
            phase=phase,
            second_camera_label=secondary.replace("_", " ").upper(),
          )
          stem = f"{frame.output_index:06d}"
          for name, array in rgb.items():
            Image.fromarray(array).save(partial / "frames" / name / f"{stem}.png")
          for name, panel in panels.items():
            panel.save(partial / "frames" / name / f"{stem}.png")
          composite.save(partial / "frames" / "composite" / f"{stem}.png")
          np.savez_compressed(
            partial / "raw" / f"{stem}.npz",
            normal_taxel_force_n=normal,
            tangent_taxel_force_n=tangent,
            tangent_magnitude_n=np.linalg.norm(tangent, axis=-1),
            right_fingertip_link_names=np.asarray(RIGHT_FINGERTIP_LINK_NAMES),
            **asdict(frame),
          )
          table.writerow({**asdict(frame), "phase": phase})
          writer.write(np.asarray(composite))
      writer.finish()
      probe = _probe_video(partial / "review.mp4", backend=video_backend)
      _validate_probe(probe, count=len(frames), fps=fps, width=width, height=height)
      summary = {
        "schema_version": REVIEW_SCHEMA,
        "completed": True,
        "source_hdf5": str(source),
        "source_sha256": _sha256(source),
        "exporter_source_sha256": _sha256(Path(__file__)),
        "source_episode_index": metadata.get("episode_index"),
        "source_seed": metadata.get("seed"),
        "source_preset": metadata.get("preset"),
        "source_validation": asdict(validation),
        "source_outcome_success": True,
        "camera_names": list(cameras),
        "right_fingertip_link_names": list(RIGHT_FINGERTIP_LINK_NAMES),
        "layout": f"left: head above {secondary}; right: normal Fn above tangential magnitude |Ft|",
        "force_sources": {
          "normal": f"/{_FORCE_GROUP}/{_NORMAL}",
          "signed_tangent": f"/{_FORCE_GROUP}/{_TANGENT}",
        },
        "force_visualization": "solver contact spatial estimate; tangent display is per-taxel vector magnitude",
        "normal_color_scale_n_per_taxel": [0, _MAX_NORMAL_TAXEL_FORCE_N],
        "tangent_color_scale_n_per_taxel": [0, _MAX_TANGENT_TAXEL_FORCE_N],
        "heatmap_orientation": "vertical flip, matching existing task-video helper",
        "raw_tangent_preserved": True,
        "rgb_temporal_interpolation_or_repetition": False,
        "camera_png_pixels": "lossless original RGB without resizing",
        "composite_camera_resizing": "bilinear spatial resize preserving aspect ratio",
        "frame_selection": "nearest distinct saved camera frames on requested playback grid, including last frame",
        "source_camera_frame_count": len(file["cameras/head/rgb"]),
        "output_frame_count": len(frames),
        "fps": fps,
        "first_camera_pose_timestamp_s": frames[0].camera_pose_timestamp_s,
        "last_camera_pose_timestamp_s": frames[-1].camera_pose_timestamp_s,
        "last_camera_timestamp_s": frames[-1].camera_timestamp_s,
        "terminal_state_timestamp_s": float(file["state/timestamp"][-1]),
        "last_tactile_timestamp_s": frames[-1].tactile_timestamp_s,
        "tactile_matching": "each camera independently chooses latest recorded solver sample not after image pose time",
        "tactile_tolerance_s": tolerance_s,
        "maximum_tactile_age_s": max(frame.tactile_age_s for frame in frames),
        "constant_fps_duration_s": len(frames) / fps,
        "last_frame_playback_time_error_s": frames[-1].playback_time_error_s,
        "maximum_absolute_playback_time_error_s": max(
          abs(frame.playback_time_error_s) for frame in frames
        ),
        "duration_minus_source_pose_span_s": len(frames) / fps
        - (frames[-1].camera_pose_timestamp_s - frames[0].camera_pose_timestamp_s),
        "last_frame_display_duration_s": 1 / fps,
        "video_validation": probe,
        "ffprobe": probe if probe["backend"] == "ffprobe" else None,
      }
      (partial / "review.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
      )
      if output.exists() or output.is_symlink():
        raise FileExistsError(
          "review output appeared while exporting; partial retained"
        )
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


__all__ = [
  "REVIEW_SCHEMA",
  "ReviewFrame",
  "export_poker_review",
  "plan_review_frames",
  "select_camera_frames",
]
