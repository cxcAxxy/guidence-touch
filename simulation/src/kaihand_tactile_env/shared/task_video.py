"""Low-overhead composite video recording for complete task episodes.

The recorder deliberately keeps rendering and encoding simple: one reusable
RGB-only MuJoCo renderer supplies both camera views, and completed composite
frames are streamed directly to a single-threaded ffmpeg process.  Only the
most recent frame is retained for terminal/error annotation.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import warnings
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from .config import FINGERTIP_LINK_NAMES
from .tactile import (
  RIGHT_FINGERTIP_LINK_NAMES,
  GenesisProbeTactileProvider,
  TactileProvider,
)

SCHEMA_VERSION = "kaihand_task_video_v1"
_MAIN_CAMERA_NAME = "front"
_OVERHEAD_CAMERA_NAME = "overhead"
_MAX_DEPTH_MM = 3.0
_MAX_NORMAL_TAXEL_FORCE_N = 0.10
_MAX_TANGENT_TAXEL_FORCE_N = 0.02
_STDERR_LIMIT = 64 * 1024
_FINGER_LABELS = ("Thumb", "Index", "Middle", "Ring", "Pinky")


def _heatmap_lut() -> np.ndarray:
  positions = np.arange(256, dtype=np.float64)
  knots = np.asarray([0, 48, 112, 176, 224, 255], dtype=np.float64)
  red = np.interp(positions, knots, [5, 40, 150, 235, 255, 255])
  green = np.interp(positions, knots, [4, 8, 15, 45, 145, 245])
  blue = np.interp(positions, knots, [12, 75, 115, 75, 20, 220])
  return np.column_stack((red, green, blue)).astype(np.uint8)


_HEATMAP_LUT = _heatmap_lut()


def _find_ffmpeg_executable() -> str:
  """Find system ffmpeg first, then imageio-ffmpeg's bundled executable."""
  executable = shutil.which("ffmpeg")
  if executable:
    return executable
  try:
    import imageio_ffmpeg

    executable = imageio_ffmpeg.get_ffmpeg_exe()
  except (ImportError, RuntimeError, OSError) as error:
    raise RuntimeError(
      "Video recording requires ffmpeg. Install the project dependencies "
      "(`pixi install`) or put an ffmpeg executable on PATH."
    ) from error
  if not executable or not Path(executable).is_file():
    raise RuntimeError(
      "Video recording requires ffmpeg, but neither PATH nor imageio-ffmpeg "
      "provided a usable executable. Run `pixi install`."
    )
  return executable


class _FfmpegPipeWriter:
  """Stream fixed-size RGB frames to ffmpeg without retaining them in memory."""

  def __init__(
    self,
    path: Path,
    *,
    fps: float,
    width: int,
    height: int,
    executable: str | None = None,
  ) -> None:
    self.path = path
    self.width = width
    self.height = height
    executable = executable or _find_ffmpeg_executable()
    command = [
      executable,
      "-hide_banner",
      "-loglevel",
      "error",
      "-nostdin",
      "-n",
      "-f",
      "rawvideo",
      "-pix_fmt",
      "rgb24",
      "-s:v",
      f"{width}x{height}",
      "-r",
      f"{fps:.8g}",
      "-i",
      "pipe:0",
      "-an",
      "-c:v",
      "libx264",
      "-preset",
      "veryfast",
      "-crf",
      "20",
      "-pix_fmt",
      "yuv420p",
      "-filter_threads",
      "1",
      "-threads",
      "1",
      "-movflags",
      "+faststart",
      str(path),
    ]
    try:
      self._process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        bufsize=0,
      )
    except OSError as error:
      raise RuntimeError(
        f"Could not start ffmpeg at {executable!r}: {error}"
      ) from error
    if self._process.stdin is None or self._process.stderr is None:
      self._terminate_process()
      raise RuntimeError("ffmpeg did not expose its input/error pipes")
    self._stdin: BinaryIO = self._process.stdin
    self._stderr: BinaryIO = self._process.stderr
    self._stderr_tail = bytearray()
    self._stderr_lock = threading.Lock()
    self._finished = False
    self._stderr_thread = threading.Thread(
      target=self._drain_stderr,
      name="task-video-ffmpeg-stderr",
      daemon=True,
    )
    self._stderr_thread.start()

  def _drain_stderr(self) -> None:
    try:
      while True:
        chunk = self._stderr.read(4096)
        if not chunk:
          break
        with self._stderr_lock:
          self._stderr_tail.extend(chunk)
          overflow = len(self._stderr_tail) - _STDERR_LIMIT
          if overflow > 0:
            del self._stderr_tail[:overflow]
    except (OSError, ValueError):
      # Pipe closure is expected during forced shutdown.
      return

  def _stderr_message(self) -> str:
    with self._stderr_lock:
      content = bytes(self._stderr_tail)
    return content.decode("utf-8", errors="replace").strip()

  def write(self, frame: np.ndarray) -> None:
    if self._finished:
      raise RuntimeError("cannot write a frame after ffmpeg has finished")
    array = np.asarray(frame)
    expected_shape = (self.height, self.width, 3)
    if array.shape != expected_shape or array.dtype != np.uint8:
      raise ValueError(
        f"video frame must be uint8 with shape {expected_shape}, "
        f"got {array.dtype} {array.shape}"
      )
    if self._process.poll() is not None:
      self._join_stderr()
      detail = self._stderr_message() or "ffmpeg exited without an error message"
      raise RuntimeError(f"ffmpeg stopped before receiving all frames: {detail}")
    try:
      payload = memoryview(np.ascontiguousarray(array)).cast("B")
      while payload:
        written = self._stdin.write(payload)
        if written is None or written <= 0:
          raise BrokenPipeError("ffmpeg pipe made no progress")
        payload = payload[written:]
    except (BrokenPipeError, OSError) as error:
      self._join_stderr()
      detail = self._stderr_message() or str(error)
      raise RuntimeError(f"ffmpeg rejected a video frame: {detail}") from error

  def _join_stderr(self) -> None:
    self._stderr_thread.join(timeout=2.0)

  def _terminate_process(self) -> None:
    process = getattr(self, "_process", None)
    if process is None or process.poll() is not None:
      return
    process.terminate()
    try:
      process.wait(timeout=3.0)
    except subprocess.TimeoutExpired:
      process.kill()
      process.wait(timeout=3.0)

  def finish(self) -> None:
    if self._finished:
      return
    self._finished = True
    try:
      self._stdin.close()
      try:
        return_code = self._process.wait(timeout=30.0)
      except subprocess.TimeoutExpired as error:
        self._terminate_process()
        raise RuntimeError(
          "ffmpeg did not finish the task video within 30 seconds"
        ) from error
    finally:
      self._join_stderr()
      try:
        self._stderr.close()
      except OSError:
        pass
    if return_code != 0:
      detail = self._stderr_message() or f"exit status {return_code}"
      raise RuntimeError(f"ffmpeg could not encode the task video: {detail}")

  def close(self) -> None:
    if not self._finished:
      self.finish()

  def abort(self) -> None:
    """Stop an encoder that cannot receive a valid recording."""
    if self._finished:
      return
    self._finished = True
    try:
      self._stdin.close()
    except OSError:
      pass
    self._terminate_process()
    self._join_stderr()
    try:
      self._stderr.close()
    except OSError:
      pass
    # The destination was proven absent before ffmpeg started, so any file at
    # this point is solely the unusable partial output from this process.
    self.path.unlink(missing_ok=True)


class _TkPreview:
  """One non-blocking Tk window displaying the already-composed video frame."""

  def __init__(self, *, width: int, height: int) -> None:
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
      raise RuntimeError(
        "video preview requires a desktop display; use --record-no-preview "
        "for headless recording"
      )
    import tkinter as tk

    from PIL import ImageTk

    self._tk = tk
    self._image_tk = ImageTk
    self._closed = False
    try:
      self._root = tk.Tk()
    except tk.TclError as error:
      raise RuntimeError(
        "could not open the video preview; use --record-no-preview"
      ) from error
    self._root.title("KaiHand task recording")
    self._root.protocol("WM_DELETE_WINDOW", self._on_close)
    self._label = tk.Label(self._root, width=width, height=height, borderwidth=0)
    self._label.pack()
    self._photo: Any | None = None

  def _on_close(self) -> None:
    self.close()

  def update(self, frame: np.ndarray) -> None:
    if self._closed:
      return
    try:
      self._photo = self._image_tk.PhotoImage(Image.fromarray(frame, mode="RGB"))
      self._label.configure(image=self._photo)
      self._root.update_idletasks()
      self._root.update()
    except self._tk.TclError:
      self._closed = True
      try:
        self._root.destroy()
      except self._tk.TclError:
        pass

  def close(self) -> None:
    if self._closed:
      return
    self._closed = True
    try:
      self._root.destroy()
    except self._tk.TclError:
      pass


def _create_poker_spatial_provider(simulation: Any) -> Any:
  """Build the independent right-hand solver field used by poker videos.

  The import is intentionally local: the legacy recorder interface remains
  usable while the task-neutral tactile module owns the optional spatial
  solver implementation.
  """
  from .contact_tactile import SolverDistributedTactileProvider

  return SolverDistributedTactileProvider(
    simulation.model,
    getattr(simulation, "genesis_probe_layout", None),
    target_geom_names=("card_core_geom",),
    link_names=RIGHT_FINGERTIP_LINK_NAMES,
  )


class TaskVideoRecorder:
  """Record two synchronized RGB views plus tactile; poker uses head/right wrist."""

  def __init__(
    self,
    simulation: Any,
    output_path: str | Path,
    *,
    fps: float = 15,
    width: int = 640,
    height: int = 480,
    tactile_provider: TactileProvider | None = None,
    preview: bool = True,
    metadata: Mapping[str, Any] | None = None,
  ) -> None:
    if not np.isfinite(fps) or fps <= 0.0:
      raise ValueError("fps must be finite and positive")
    if isinstance(width, bool) or isinstance(height, bool):
      raise ValueError("video width and height must be integers")
    if not isinstance(width, int) or not isinstance(height, int):
      raise ValueError("video width and height must be integers")
    if width < 160 or height < 120:
      raise ValueError("video width and height must be at least 160x120")
    if width % 2 or height % 2:
      raise ValueError("video width and height must be even for yuv420p encoding")

    self.simulation = simulation
    self.output_path = Path(output_path).expanduser().resolve()
    if self.output_path.suffix.lower() != ".mp4":
      raise ValueError("task video output path must end in .mp4")
    self.metadata_path = self.output_path.with_suffix(".json")
    conflicts = [
      str(path) for path in (self.output_path, self.metadata_path) if path.exists()
    ]
    if conflicts:
      raise FileExistsError(
        "task video recording never overwrites existing files: " + ", ".join(conflicts)
      )
    self.output_path.parent.mkdir(parents=True, exist_ok=True)

    self.fps = float(fps)
    self.width = width
    self.height = height
    self._period = 1.0 / self.fps
    self._metadata = _json_safe(dict(metadata or {}))
    self._task_outcome: dict[str, Any] | None = None
    poker = getattr(simulation, "scene", None) == "poker-draw"
    self._default_main_camera = "head" if poker else _MAIN_CAMERA_NAME
    self._default_secondary_camera = "right_wrist" if poker else _OVERHEAD_CAMERA_NAME
    self._secondary_camera = self._default_secondary_camera
    self._main_camera: str | mujoco.MjvCamera = self._default_main_camera
    self._main_camera_mode = f"fixed:{self._default_main_camera}"
    self._closed = False
    self._finished = False
    self._finishing = False
    self._frame_count = 0
    self._first_time: float | None = None
    self._last_time: float | None = None
    self._last_observed_time: float | None = None
    self._clock_origin: float | None = None
    self._next_sample_time: float | None = None
    self._phase_events: list[dict[str, Any]] = []
    self._last_phase = "initial"
    self._last_frame: np.ndarray | None = None
    self._status: tuple[str, bool | None] = ("RECORDING", None)
    self._terminal_error: str | None = None

    if tactile_provider is None:
      tactile_provider = GenesisProbeTactileProvider(
        simulation.model, simulation.genesis_probe_layout
      )
    if not tactile_provider.available:
      raise RuntimeError(f"tactile provider {tactile_provider.source!r} is unavailable")
    self.tactile_provider = tactile_provider
    self._grid_indices = _fingertip_grid_indices(tactile_provider)
    self._has_taxel_maps = all(
      indices is not None for indices in self._grid_indices.values()
    )
    self._force_unit = str(getattr(tactile_provider, "force_unit", "unspecified"))
    self._poker_spatial_provider: Any | None = None
    if getattr(simulation, "scene", None) == "poker-draw":
      self._poker_spatial_provider = _create_poker_spatial_provider(simulation)
      if not getattr(self._poker_spatial_provider, "available", True):
        raise RuntimeError(
          "right-hand solver-distributed tactile provider is unavailable"
        )
      provider_links = tuple(self._poker_spatial_provider.link_names)
      if provider_links != RIGHT_FINGERTIP_LINK_NAMES:
        raise RuntimeError(
          "poker video tactile provider must expose the five right fingertips "
          "in canonical order"
        )
    self._display_force_unit = (
      "N" if self._poker_spatial_provider is not None else self._force_unit
    )

    # 320x240 bounds rendering cost at the default 640x480 composite.  Both
    # named cameras share this one renderer and are resized only during layout.
    self._render_width = max(80, min(320, width // 2))
    self._render_height = max(60, min(240, int(round(self._render_width * 0.75))))
    executable = _find_ffmpeg_executable()
    self._renderer: Any | None = None
    self._writer: Any | None = None
    self._preview: Any | None = None
    try:
      self._renderer = mujoco.Renderer(
        simulation.model,
        height=self._render_height,
        width=self._render_width,
      )
      self._renderer.disable_depth_rendering()
      self._renderer.disable_segmentation_rendering()
      self._renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
      self._writer = _FfmpegPipeWriter(
        self.output_path,
        fps=self.fps,
        width=self.width,
        height=self.height,
        executable=executable,
      )
      if preview:
        self._preview = _TkPreview(width=width, height=height)
    except BaseException:
      self._abort_resources()
      raise

  def __enter__(self) -> TaskVideoRecorder:
    return self

  def __exit__(
    self,
    exception_type: type[BaseException] | None,
    exception: BaseException | None,
    _traceback: Any,
  ) -> None:
    if self._finished:
      self.close()
      return
    message = (
      str(exception)
      if exception is not None
      else "recording context closed before finish()"
    )
    try:
      self.finish(success=False, error=message)
    except Exception as cleanup_error:
      self.close()
      if exception_type is None:
        raise
      warnings.warn(
        f"task video cleanup also failed: {cleanup_error}",
        RuntimeWarning,
        stacklevel=2,
      )

  def follow_viewer_camera(self, camera: mujoco.MjvCamera | None) -> None:
    """Optionally use a native viewer camera for the main panel.

    Passing ``None`` restores the default head/wrist (poker) or front/overhead.
    Explicit free-camera diagnostic callers retain their old overhead pane. The
    camera object is referenced rather than copied so subsequent viewer moves
    are reflected in recorded frames without creating another renderer.
    """
    self._ensure_active()
    if camera is None:
      self._main_camera = self._default_main_camera
      self._main_camera_mode = f"fixed:{self._default_main_camera}"
      self._secondary_camera = self._default_secondary_camera
    else:
      self._main_camera = camera
      self._main_camera_mode = "viewer"
      self._secondary_camera = _OVERHEAD_CAMERA_NAME

  def observe(self, simulation: Any, phase: str) -> None:
    """Sample one frame when the simulation-time video clock is due."""
    self._ensure_active()
    if simulation is not self.simulation:
      raise ValueError("video recorder received a different simulation instance")
    sim_time = float(simulation.data.time)
    if not np.isfinite(sim_time):
      raise ValueError("simulation timestamp must be finite")
    if (
      self._last_observed_time is not None
      and sim_time + 1.0e-12 < self._last_observed_time
    ):
      raise RuntimeError("simulation time moved backwards during video recording")
    self._last_observed_time = sim_time
    normalized_phase = str(phase) or "unnamed"
    self._record_phase(sim_time, normalized_phase)
    due = self._next_sample_time is None or sim_time + 1.0e-12 >= self._next_sample_time
    if not due:
      return
    self._capture(sim_time, normalized_phase)
    if self._clock_origin is None:
      self._clock_origin = sim_time
      sample_index = 1
    else:
      elapsed = max(0.0, sim_time - self._clock_origin)
      sample_index = int(np.floor(elapsed / self._period + 1.0e-10)) + 1
    # Anchor every deadline at the first sample instead of adding a period to
    # a 500 Hz-quantized observation time.  Thus quantization never accrues.
    self._next_sample_time = self._clock_origin + sample_index * self._period

  def set_outcome(self, outcome: Mapping[str, Any]) -> None:
    """Attach measured task metrics to the sidecar before finalization."""
    if self._finished or self._closed:
      raise RuntimeError("cannot change the outcome of a finished task video")
    self._task_outcome = _json_safe(dict(outcome))

  def finish(
    self,
    success: bool,
    error: BaseException | str | None = None,
  ) -> tuple[Path, Path]:
    """Append a marked terminal frame and finalize the MP4 and JSON sidecar."""
    if self._finished:
      return self.output_path, self.metadata_path
    if self._closed:
      raise RuntimeError("cannot finish a closed task video recorder")
    if self._finishing:
      raise RuntimeError("task video recorder is already finishing")
    self._finishing = True
    terminal_error = None if error is None else str(error)
    self._terminal_error = terminal_error
    task_success = bool(success)
    self._status = ("SUCCESS" if task_success else "FAILED", task_success)
    sim_time = float(self.simulation.data.time)
    recording_issues: list[str] = []
    try:
      if self._frame_count == 0:
        self._status = ("RECORDING", None)
        fallback = self._capture(sim_time, "initial", allow_fallback=True)
        if fallback is not None:
          recording_issues.append(f"initial frame fallback: {fallback}")
        self._status = ("SUCCESS" if task_success else "FAILED", task_success)
      self._record_phase(sim_time, self._last_phase)
      fallback = self._capture(sim_time, self._last_phase, allow_fallback=True)
      if fallback is not None:
        recording_issues.append(f"terminal frame fallback: {fallback}")
    except Exception as caught:
      recording_issues.append(f"terminal frame write failed: {caught}")

    try:
      assert self._writer is not None
      self._writer.finish()
    except Exception as caught:
      recording_issues.append(f"ffmpeg finalization failed: {caught}")
    encoding_error = "; ".join(recording_issues) or None
    encoding_complete = encoding_error is None
    episode_completed = terminal_error is None
    recording_complete = encoding_complete and episode_completed
    sidecar = self._sidecar(
      task_success=task_success,
      task_error=terminal_error,
      recording_complete=recording_complete,
      episode_completed=episode_completed,
      encoding_complete=encoding_complete,
      encoding_error=encoding_error,
    )
    try:
      with self.metadata_path.open("x", encoding="utf-8") as stream:
        json.dump(sidecar, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    except Exception:
      self._release_resources()
      self._closed = True
      self._finishing = False
      raise
    self._finished = True
    self._finishing = False
    self._release_resources()
    self._closed = True
    if encoding_error is not None:
      raise RuntimeError(f"task video is incomplete: {encoding_error}")
    return self.output_path, self.metadata_path

  def close(self) -> None:
    """Release preview, renderer and encoder resources idempotently."""
    if self._closed:
      return
    try:
      self._release_resources()
    finally:
      self._closed = True

  def _ensure_active(self) -> None:
    if self._closed:
      raise RuntimeError("task video recorder is closed")
    if self._finished or self._finishing:
      raise RuntimeError("task video recorder has already finished")

  def _record_phase(self, sim_time: float, phase: str) -> None:
    self._last_phase = phase
    if not self._phase_events or self._phase_events[-1]["phase"] != phase:
      self._phase_events.append({"time": sim_time, "phase": phase})

  def _capture(
    self,
    sim_time: float,
    phase: str,
    *,
    allow_fallback: bool = False,
  ) -> str | None:
    fallback_message: str | None = None
    try:
      frame = self._compose_frame(sim_time, phase)
    except Exception as error:
      if not allow_fallback:
        raise
      fallback_message = str(error)
      frame = self._fallback_frame(sim_time, phase, error)
    assert self._writer is not None
    self._writer.write(frame)
    if self._preview is not None:
      self._preview.update(frame)
    self._last_frame = frame
    self._frame_count += 1
    if self._first_time is None:
      self._first_time = sim_time
    self._last_time = sim_time
    return fallback_message

  def _render(self, camera: str | mujoco.MjvCamera) -> np.ndarray:
    assert self._renderer is not None
    self._renderer.update_scene(self.simulation.data, camera=camera)
    return np.asarray(self._renderer.render(), dtype=np.uint8).copy()

  def _compose_frame(self, sim_time: float, phase: str) -> np.ndarray:
    if self._poker_spatial_provider is not None:
      sample = self._poker_spatial_provider.read(self.simulation.data)
      normal_maps, tangent_maps, normal_forces, tangent_forces = (
        self._poker_tactile_values(sample)
      )
      main_rgb = self._render(self._main_camera)
      overhead_rgb = self._render(self._secondary_camera)
      return _compose_poker_dashboard(
        main_rgb,
        overhead_rgb,
        normal_maps,
        tangent_maps,
        normal_forces,
        tangent_forces,
        width=self.width,
        height=self.height,
        sim_time=sim_time,
        phase=phase,
        status=self._status,
        error=self._terminal_error,
        tactile_source=str(self._poker_spatial_provider.source),
        main_camera_label="HEAD"
        if self._main_camera_mode == "fixed:head"
        else "GLOBAL",
        second_camera_label=self._secondary_camera.replace("_", " ").upper(),
      )

    sample = self.tactile_provider.read(self.simulation.data)
    depths, forces = self._tactile_values(sample)
    main_rgb = self._render(self._main_camera)
    overhead_rgb = self._render(_OVERHEAD_CAMERA_NAME)
    return _compose_dashboard(
      main_rgb,
      overhead_rgb,
      depths,
      forces,
      width=self.width,
      height=self.height,
      sim_time=sim_time,
      phase=phase,
      status=self._status,
      error=self._terminal_error,
      tactile_source=str(self.tactile_provider.source),
      force_unit=self._force_unit,
      has_taxel_maps=self._has_taxel_maps,
    )

  def _poker_tactile_values(
    self, sample: Any
  ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return canonical right-hand solver fields for the poker dashboard."""
    sample_links = tuple(sample.link_names)
    if len(sample_links) != len(set(sample_links)):
      raise RuntimeError("poker tactile sample contains duplicate link names")
    missing = set(RIGHT_FINGERTIP_LINK_NAMES) - set(sample_links)
    if missing:
      raise RuntimeError(
        "poker tactile sample is missing right fingertips: "
        + ", ".join(sorted(missing))
      )
    source_indices = np.asarray(
      [sample_links.index(name) for name in RIGHT_FINGERTIP_LINK_NAMES],
      dtype=np.intp,
    )

    normal_force = _finite_tactile_array(
      sample.normal_force_n,
      (len(sample_links),),
      "normal_force_n",
    )[source_indices]
    tangent_force_components = _finite_tactile_array(
      sample.tangent_force_n,
      (len(sample_links), 2),
      "tangent_force_n",
    )[source_indices]
    normal_maps = _finite_tactile_array(
      sample.normal_taxel_force_n,
      (len(sample_links), 7, 5),
      "normal_taxel_force_n",
    )[source_indices]
    tangent_components = _finite_tactile_array(
      sample.tangent_taxel_force_n,
      (len(sample_links), 7, 5, 2),
      "tangent_taxel_force_n",
    )[source_indices]

    # Normal is a non-negative compressive load.  The tangent display uses
    # the norm of the two stable grid-chart components, not the provider's
    # sum-of-magnitudes diagnostic, so each label is the true net resultant.
    normal_maps = np.maximum(normal_maps, 0.0)
    normal_force = np.maximum(normal_force, 0.0)
    tangent_maps = np.linalg.norm(tangent_components, axis=-1)
    tangent_force = np.linalg.norm(tangent_force_components, axis=-1)
    return normal_maps, tangent_maps, normal_force, tangent_force

  def _tactile_values(self, sample: Any) -> tuple[np.ndarray, np.ndarray]:
    depths = np.zeros((10, 7, 5), dtype=np.float64)
    probe_depth = getattr(self.tactile_provider, "probe_depth", None)
    if probe_depth is not None:
      probe_depth_array = np.asarray(probe_depth, dtype=np.float64)
      for grid_index, link_name in enumerate(FINGERTIP_LINK_NAMES):
        indices = self._grid_indices.get(link_name)
        if indices is not None:
          depths[grid_index] = probe_depth_array[indices].reshape(7, 5) * 1000.0
    forces = np.zeros(10, dtype=np.float64)
    sample_links = tuple(sample.link_names)
    sample_forces = np.asarray(sample.normal_force, dtype=np.float64)
    force_by_name = dict(zip(sample_links, sample_forces, strict=True))
    for index, link_name in enumerate(FINGERTIP_LINK_NAMES):
      forces[index] = force_by_name.get(link_name, 0.0)
    return depths, forces

  def _fallback_frame(
    self,
    sim_time: float,
    phase: str,
    render_error: Exception,
  ) -> np.ndarray:
    if self._last_frame is None:
      image = Image.new("RGB", (self.width, self.height), (15, 16, 20))
    else:
      image = Image.fromarray(self._last_frame.copy(), mode="RGB")
    draw = ImageDraw.Draw(image)
    message = _ascii_text(f"FINAL FRAME FALLBACK: {render_error}", 90)
    draw.rectangle((0, self.height - 30, self.width, self.height), fill=(125, 20, 20))
    draw.text((8, self.height - 23), message, fill=(255, 255, 255))
    _draw_header(
      draw,
      width=self.width,
      sim_time=sim_time,
      phase=phase,
      total_force=0.0,
      force_unit=self._display_force_unit,
      status=("INCOMPLETE", False),
    )
    return np.asarray(image, dtype=np.uint8)

  def _sidecar(
    self,
    *,
    task_success: bool,
    task_error: str | None,
    recording_complete: bool,
    episode_completed: bool,
    encoding_complete: bool,
    encoding_error: str | None,
  ) -> dict[str, Any]:
    duration = (
      None
      if self._first_time is None or self._last_time is None
      else max(0.0, self._last_time - self._first_time)
    )
    return {
      "schema_version": SCHEMA_VERSION,
      "created_utc": datetime.now(timezone.utc).isoformat(),
      "video_path": str(self.output_path),
      "success": task_success and recording_complete,
      "task_success": task_success,
      "recording_complete": recording_complete,
      "episode_completed": episode_completed,
      "encoding_complete": encoding_complete,
      "incomplete": not recording_complete,
      "error": task_error or encoding_error,
      "task_error": task_error,
      "encoding_error": encoding_error,
      "fps": self.fps,
      "width": self.width,
      "height": self.height,
      "render_width": self._render_width,
      "render_height": self._render_height,
      "frame_count": self._frame_count,
      "first_simulation_time": self._first_time,
      "last_simulation_time": self._last_time,
      "simulation_duration": duration,
      "phases": self._phase_events,
      "cameras": {
        "main": self._main_camera_mode,
        "secondary"
        if self._poker_spatial_provider is not None
        else "overhead": self._secondary_camera,
      },
      "tactile": self._tactile_metadata(),
      "metadata": self._metadata,
      "task_outcome": self._task_outcome,
    }

  def _tactile_metadata(self) -> dict[str, Any]:
    if self._poker_spatial_provider is None:
      return {
        "source": str(self.tactile_provider.source),
        "link_names": list(FINGERTIP_LINK_NAMES),
        "grid_shape": [7, 5],
        "depth_unit": "mm",
        "spatial_taxel_maps": self._has_taxel_maps,
        "normal_force_unit": self._force_unit,
      }

    provider = self._poker_spatial_provider
    provider_metadata = getattr(provider, "metadata", None)
    if callable(provider_metadata):
      provider_metadata = provider_metadata()
    if not isinstance(provider_metadata, Mapping):
      provider_metadata = {}
    safe_provider_metadata = _json_safe(dict(provider_metadata))
    assert isinstance(safe_provider_metadata, dict)
    distribution = {
      "algorithm": safe_provider_metadata.get(
        "kernel",
        safe_provider_metadata.get(
          "algorithm", "normalized_gaussian_3d_taxel_distance"
        ),
      ),
      "kernel_sigma_m": float(getattr(provider, "kernel_sigma_m", 0.003)),
      "taxel_force_semantics": safe_provider_metadata.get(
        "taxel_force_semantics",
        "normalized allocation of solver force; spatial estimate",
      ),
      "tangent_basis_semantics": safe_provider_metadata.get(
        "tangent_basis_semantics",
        "axis0=grid column increasing; axis1=grid row increasing",
      ),
    }
    return {
      "source": str(provider.source),
      "mode": "poker_right_hand_solver_spatial_estimate",
      "hand": "right",
      "link_names": list(RIGHT_FINGERTIP_LINK_NAMES),
      "grid_shape": [7, 5],
      "spatial_taxel_maps": True,
      "spatial_estimate": True,
      "normal_force_unit": "N",
      "taxel_force_unit": str(getattr(provider, "taxel_force_unit", "N")),
      "display_taxel_color_unit": "N/taxel",
      "display_rows": [
        "normal_taxel_force_n",
        "norm(tangent_taxel_force_n, axis=-1)",
      ],
      "color_scales": {
        "normal_taxel_force_n": {
          "minimum": 0.0,
          "maximum": _MAX_NORMAL_TAXEL_FORCE_N,
          "unit": "N/taxel",
          "fixed": True,
          "saturates_above_maximum": True,
        },
        "tangent_taxel_force_norm_n": {
          "minimum": 0.0,
          "maximum": _MAX_TANGENT_TAXEL_FORCE_N,
          "unit": "N/taxel",
          "fixed": True,
          "saturates_above_maximum": True,
        },
      },
      "spatial_distribution": distribution,
      "provider_metadata": safe_provider_metadata,
    }

  def _release_resources(self) -> None:
    preview, self._preview = self._preview, None
    renderer, self._renderer = self._renderer, None
    writer, self._writer = self._writer, None
    first_error: Exception | None = None
    for resource in (preview, renderer, writer):
      if resource is None:
        continue
      try:
        resource.close()
      except Exception as error:
        if first_error is None:
          first_error = error
    if first_error is not None:
      raise first_error

  def _abort_resources(self) -> None:
    preview, self._preview = self._preview, None
    renderer, self._renderer = self._renderer, None
    writer, self._writer = self._writer, None
    if writer is not None:
      abort = getattr(writer, "abort", None)
      try:
        if abort is not None:
          abort()
        else:
          writer.close()
      except Exception:
        pass
    for resource in (renderer, preview):
      if resource is not None:
        try:
          resource.close()
        except Exception:
          pass


def _fingertip_grid_indices(provider: Any) -> dict[str, np.ndarray | None]:
  layout = getattr(provider, "layout", None)
  if layout is None or not hasattr(provider, "probe_depth"):
    return {name: None for name in FINGERTIP_LINK_NAMES}
  body_names = np.asarray(layout.body_names)
  result: dict[str, np.ndarray | None] = {}
  all_indices: list[np.ndarray] = []
  for name in FINGERTIP_LINK_NAMES:
    indices = np.flatnonzero(body_names == name)
    if len(indices) != 35:
      raise RuntimeError(f"{name}: expected 35 tactile probes, got {len(indices)}")
    if layout.grid_shape[int(indices[0])] != (7, 5):
      raise RuntimeError(f"{name}: expected a 7x5 tactile grid")
    result[name] = indices
    all_indices.append(indices)
  combined = np.concatenate(all_indices)
  if len(np.unique(combined)) != len(combined) or len(combined) != layout.count:
    raise RuntimeError("fingertip grids do not cover the tactile layout exactly once")
  return result


def _finite_tactile_array(
  value: Any,
  expected_shape: tuple[int, ...],
  name: str,
) -> np.ndarray:
  array = np.asarray(value, dtype=np.float64)
  if array.shape != expected_shape:
    raise RuntimeError(
      f"poker tactile {name} must have shape {expected_shape}, got {array.shape}"
    )
  if not np.all(np.isfinite(array)):
    raise RuntimeError(f"poker tactile {name} contains non-finite values")
  return array


def _compose_poker_dashboard(
  main_rgb: np.ndarray,
  overhead_rgb: np.ndarray,
  normal_taxel_force_n: np.ndarray,
  tangent_taxel_force_n: np.ndarray,
  normal_force_n: np.ndarray,
  tangent_force_n: np.ndarray,
  *,
  width: int,
  height: int,
  sim_time: float,
  phase: str,
  status: tuple[str, bool | None],
  error: str | None,
  tactile_source: str,
  main_camera_label: str = "HEAD",
  second_camera_label: str = "RIGHT WRIST",
) -> np.ndarray:
  """Compose poker video with only the active right hand's Fn/Ft fields."""
  image = Image.new("RGB", (width, height), (15, 16, 20))
  draw = ImageDraw.Draw(image)
  header_height = max(26, min(40, height // 10))
  _draw_header(
    draw,
    width=width,
    sim_time=sim_time,
    phase=phase,
    total_force=float(np.sum(np.maximum(normal_force_n, 0.0))),
    force_unit="N",
    status=status,
  )

  margin = max(3, width // 160)
  content_top = header_height + margin
  content_bottom = height - margin
  split_x = int(round(width * 0.64))
  main_box = (margin, content_top, split_x - margin, content_bottom)
  sidebar = (split_x + margin, content_top, width - margin, content_bottom)
  _paste_fit(image, main_rgb, main_box, Image.Resampling.BILINEAR)
  draw.rectangle(main_box, outline=(72, 76, 86), width=1)
  draw.rectangle(
    (main_box[0], main_box[1], main_box[0] + 48, main_box[1] + 15),
    fill=(0, 0, 0),
  )
  draw.text((main_box[0] + 4, main_box[1] + 2), main_camera_label, fill=(235, 235, 235))

  sidebar_width = max(1, sidebar[2] - sidebar[0])
  sidebar_height = max(1, sidebar[3] - sidebar[1])
  overhead_height = max(40, int(round(sidebar_height * 0.36)))
  overhead_box = (
    sidebar[0],
    sidebar[1],
    sidebar[2],
    min(sidebar[3], sidebar[1] + overhead_height),
  )
  _paste_fit(image, overhead_rgb, overhead_box, Image.Resampling.BILINEAR)
  draw.rectangle(overhead_box, outline=(72, 76, 86), width=1)
  draw.rectangle(
    (overhead_box[0], overhead_box[1], overhead_box[0] + 68, overhead_box[1] + 15),
    fill=(0, 0, 0),
  )
  draw.text(
    (overhead_box[0] + 4, overhead_box[1] + 2),
    second_camera_label,
    fill=(235, 235, 235),
  )

  tactile_top = overhead_box[3] + margin
  tactile_bottom = max(tactile_top + 1, sidebar[3] - 11)
  tactile_height = max(1, tactile_bottom - tactile_top)
  row_height = max(1, tactile_height // 2)
  column_width = max(1, sidebar_width // 5)
  row_specs = (
    (
      "R Fn est [N/taxel] 0..0.10 SAT",
      normal_taxel_force_n,
      normal_force_n,
      "Fn",
      _MAX_NORMAL_TAXEL_FORCE_N,
    ),
    (
      "R Ft est [N/taxel] 0..0.02 SAT",
      tangent_taxel_force_n,
      tangent_force_n,
      "Ft",
      _MAX_TANGENT_TAXEL_FORCE_N,
    ),
  )
  for row_index, (caption, maps, forces, quantity, scale_maximum) in enumerate(
    row_specs
  ):
    row_top = tactile_top + row_index * row_height
    row_bottom = tactile_bottom if row_index == 1 else row_top + row_height
    draw.text((sidebar[0] + 2, row_top), caption, fill=(205, 207, 214))
    cell_top = min(row_bottom, row_top + 12)
    for finger_index, finger_label in enumerate(_FINGER_LABELS):
      cell_left = sidebar[0] + finger_index * column_width
      cell_right = sidebar[2] if finger_index == 4 else cell_left + column_width
      _draw_force_tactile_cell(
        image,
        draw,
        maps[finger_index],
        float(forces[finger_index]),
        (cell_left, cell_top, cell_right, row_bottom),
        finger_label[:2],
        quantity=quantity,
        scale_maximum=scale_maximum,
      )
  source = _ascii_text(
    f"{tactile_source} | spatial estimate",
    max(12, sidebar_width // 5),
  )
  draw.text((sidebar[0] + 2, sidebar[3] - 10), source, fill=(145, 148, 158))

  if error:
    message = _ascii_text(error, max(20, width // 7))
    draw.rectangle(
      (margin, content_bottom - 20, split_x - margin, content_bottom),
      fill=(105, 16, 20),
    )
    draw.text((margin + 5, content_bottom - 16), message, fill=(255, 235, 235))
  return np.asarray(image, dtype=np.uint8)


def _compose_dashboard(
  main_rgb: np.ndarray,
  overhead_rgb: np.ndarray,
  depths_mm: np.ndarray,
  forces: np.ndarray,
  *,
  width: int,
  height: int,
  sim_time: float,
  phase: str,
  status: tuple[str, bool | None],
  error: str | None,
  tactile_source: str,
  force_unit: str,
  has_taxel_maps: bool,
) -> np.ndarray:
  image = Image.new("RGB", (width, height), (15, 16, 20))
  draw = ImageDraw.Draw(image)
  header_height = max(26, min(40, height // 10))
  total_force = float(np.sum(np.maximum(forces, 0.0)))
  _draw_header(
    draw,
    width=width,
    sim_time=sim_time,
    phase=phase,
    total_force=total_force,
    force_unit=force_unit,
    status=status,
  )

  margin = max(3, width // 160)
  content_top = header_height + margin
  content_bottom = height - margin
  split_x = int(round(width * 0.64))
  main_box = (margin, content_top, split_x - margin, content_bottom)
  sidebar = (split_x + margin, content_top, width - margin, content_bottom)
  _paste_fit(image, main_rgb, main_box, Image.Resampling.BILINEAR)
  draw.rectangle(main_box, outline=(72, 76, 86), width=1)
  draw.rectangle(
    (main_box[0], main_box[1], main_box[0] + 48, main_box[1] + 15), fill=(0, 0, 0)
  )
  draw.text((main_box[0] + 4, main_box[1] + 2), "FRONT", fill=(235, 235, 235))

  sidebar_width = max(1, sidebar[2] - sidebar[0])
  sidebar_height = max(1, sidebar[3] - sidebar[1])
  overhead_height = max(40, int(round(sidebar_height * 0.39)))
  overhead_box = (
    sidebar[0],
    sidebar[1],
    sidebar[2],
    min(sidebar[3], sidebar[1] + overhead_height),
  )
  _paste_fit(image, overhead_rgb, overhead_box, Image.Resampling.BILINEAR)
  draw.rectangle(overhead_box, outline=(72, 76, 86), width=1)
  draw.rectangle(
    (overhead_box[0], overhead_box[1], overhead_box[0] + 68, overhead_box[1] + 15),
    fill=(0, 0, 0),
  )
  draw.text(
    (overhead_box[0] + 4, overhead_box[1] + 2), "OVERHEAD", fill=(235, 235, 235)
  )

  tactile_caption_top = overhead_box[3] + margin
  depth_caption = "depth[mm] 0..3" if has_taxel_maps else "depth map: N/A"
  draw.text((sidebar[0] + 2, tactile_caption_top), depth_caption, fill=(205, 207, 214))
  tactile_top = tactile_caption_top + 12
  tactile_bottom = max(tactile_top + 1, sidebar[3] - 11)
  tactile_height = max(1, tactile_bottom - tactile_top)
  row_height = max(1, tactile_height // 2)
  column_width = max(1, sidebar_width // 5)
  for side_index, side_label in enumerate(("L", "R")):
    row_top = tactile_top + side_index * row_height
    row_bottom = tactile_bottom if side_index == 1 else row_top + row_height
    for finger_index, finger_label in enumerate(_FINGER_LABELS):
      sensor_index = side_index * 5 + finger_index
      cell_left = sidebar[0] + finger_index * column_width
      cell_right = sidebar[2] if finger_index == 4 else cell_left + column_width
      _draw_tactile_cell(
        image,
        draw,
        depths_mm[sensor_index],
        float(forces[sensor_index]),
        (cell_left, row_top, cell_right, row_bottom),
        f"{side_label} {finger_label[:2]}",
        force_unit=force_unit,
        depth_available=has_taxel_maps,
      )
  source = _ascii_text(
    f"{tactile_source} | force:{force_unit}", max(12, sidebar_width // 5)
  )
  draw.text((sidebar[0] + 2, sidebar[3] - 10), source, fill=(145, 148, 158))

  if error:
    message = _ascii_text(error, max(20, width // 7))
    draw.rectangle(
      (margin, content_bottom - 20, split_x - margin, content_bottom),
      fill=(105, 16, 20),
    )
    draw.text((margin + 5, content_bottom - 16), message, fill=(255, 235, 235))
  return np.asarray(image, dtype=np.uint8)


def _draw_header(
  draw: ImageDraw.ImageDraw,
  *,
  width: int,
  sim_time: float,
  phase: str,
  total_force: float,
  force_unit: str,
  status: tuple[str, bool | None],
) -> None:
  status_text, succeeded = status
  status_color = (
    (36, 145, 73)
    if succeeded is True
    else (190, 47, 47)
    if succeeded is False
    else (44, 100, 176)
  )
  draw.rectangle((0, 0, width, 39), fill=(28, 30, 36))
  force_label = (
    f"normal={total_force:6.2f} N"
    if force_unit == "N"
    else f"normal_proxy={total_force:6.2f}"
  )
  label = _ascii_text(
    f"t={sim_time:7.3f}s  phase={phase}  {force_label}", max(18, width // 8)
  )
  draw.text((7, 7), label, fill=(238, 238, 242))
  status_width = min(78, max(48, width // 7))
  draw.rectangle((width - status_width, 0, width, 25), fill=status_color)
  draw.text((width - status_width + 5, 7), status_text, fill=(255, 255, 255))


def _draw_force_tactile_cell(
  image: Image.Image,
  draw: ImageDraw.ImageDraw,
  taxel_force_n: np.ndarray,
  resultant_force_n: float,
  box: tuple[int, int, int, int],
  label: str,
  *,
  quantity: str,
  scale_maximum: float,
) -> None:
  left, top, right, bottom = box
  width = max(1, right - left)
  height = max(1, bottom - top)
  draw.rectangle(box, outline=(52, 55, 64), width=1)
  draw.text((left + 2, top + 1), label, fill=(220, 220, 225))
  draw.text(
    (left + 2, top + 11),
    f"{quantity}{max(resultant_force_n, 0.0):.2g}N",
    fill=(180, 183, 190),
  )
  available_height = max(1, height - 25)
  heatmap_width = max(1, min(width - 4, int(available_height * 5 / 7)))
  heatmap_height = max(1, min(available_height, int(heatmap_width * 7 / 5)))
  heatmap_left = left + max(2, (width - heatmap_width) // 2)
  heatmap_top = top + 24
  normalized = np.clip(
    np.asarray(taxel_force_n, dtype=np.float64) / scale_maximum,
    0.0,
    1.0,
  )
  indices = np.rint(normalized * 255.0).astype(np.uint8)
  rgb = _HEATMAP_LUT[np.flipud(indices)]
  heatmap = Image.fromarray(rgb, mode="RGB").resize(
    (heatmap_width, heatmap_height), Image.Resampling.NEAREST
  )
  image.paste(heatmap, (heatmap_left, heatmap_top))


def _draw_tactile_cell(
  image: Image.Image,
  draw: ImageDraw.ImageDraw,
  depth_mm: np.ndarray,
  force: float,
  box: tuple[int, int, int, int],
  label: str,
  *,
  force_unit: str,
  depth_available: bool,
) -> None:
  left, top, right, bottom = box
  width = max(1, right - left)
  height = max(1, bottom - top)
  draw.rectangle(box, outline=(52, 55, 64), width=1)
  draw.text((left + 2, top + 2), label, fill=(220, 220, 225))
  short_unit = "N" if force_unit == "N" else "p"
  draw.text(
    (left + 2, top + 12),
    f"{max(force, 0.0):.1f}{short_unit}",
    fill=(180, 183, 190),
  )
  available_height = max(1, height - 28)
  heatmap_width = max(1, min(width - 4, int(available_height * 5 / 7)))
  heatmap_height = max(1, min(available_height, int(heatmap_width * 7 / 5)))
  heatmap_left = left + max(2, (width - heatmap_width) // 2)
  heatmap_top = top + 26
  if not depth_available:
    draw.rectangle(
      (
        heatmap_left,
        heatmap_top,
        heatmap_left + heatmap_width,
        heatmap_top + heatmap_height,
      ),
      fill=(38, 40, 46),
      outline=(92, 95, 104),
    )
    draw.text((heatmap_left + 2, heatmap_top + 2), "N/A", fill=(175, 177, 184))
    return
  normalized = np.clip(np.asarray(depth_mm) / _MAX_DEPTH_MM, 0.0, 1.0)
  indices = np.rint(normalized * 255.0).astype(np.uint8)
  rgb = _HEATMAP_LUT[np.flipud(indices)]
  heatmap = Image.fromarray(rgb, mode="RGB").resize(
    (heatmap_width, heatmap_height), Image.Resampling.NEAREST
  )
  image.paste(heatmap, (heatmap_left, heatmap_top))


def _paste_fit(
  target: Image.Image,
  pixels: np.ndarray,
  box: tuple[int, int, int, int],
  resampling: Image.Resampling,
) -> None:
  source = Image.fromarray(np.asarray(pixels, dtype=np.uint8), mode="RGB")
  left, top, right, bottom = box
  maximum_width = max(1, right - left)
  maximum_height = max(1, bottom - top)
  scale = min(maximum_width / source.width, maximum_height / source.height)
  fitted_width = max(1, int(round(source.width * scale)))
  fitted_height = max(1, int(round(source.height * scale)))
  resized = source.resize((fitted_width, fitted_height), resampling)
  x = left + (maximum_width - fitted_width) // 2
  y = top + (maximum_height - fitted_height) // 2
  target.paste(resized, (x, y))


def _ascii_text(value: Any, maximum_length: int) -> str:
  text = str(value).encode("ascii", errors="replace").decode("ascii")
  if len(text) <= maximum_length:
    return text
  return text[: max(0, maximum_length - 3)] + "..."


def _json_safe(value: Any) -> Any:
  if isinstance(value, np.ndarray):
    return _json_safe(value.tolist())
  if value is None or isinstance(value, (str, int, float, bool)):
    return value
  if isinstance(value, Path):
    return str(value)
  if isinstance(value, np.generic):
    return value.item()
  if isinstance(value, Mapping):
    return {str(key): _json_safe(item) for key, item in value.items()}
  if isinstance(value, (list, tuple)):
    return [_json_safe(item) for item in value]
  return str(value)


__all__ = ["SCHEMA_VERSION", "TaskVideoRecorder"]
