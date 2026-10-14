"""Raw episode and fingertip-force artifacts for poker pi0.5 rollouts."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from kaihand_tactile_env.shared.config import WorkcellConfig
from kaihand_tactile_env.shared.recording import EpisodeRecorder, validate_episode
from kaihand_tactile_env.shared.tactile import RIGHT_FINGERTIP_LINK_NAMES
from kaihand_tactile_env.shared.tactile import SolverContactTactileProvider


class PokerPi05RawCapture:
  """Record the executed simulation, including RGB and raw contact forces."""

  def __init__(self, simulation, output: Path, cameras, *, metadata: dict):
    self.output = Path(output)
    raw_dir = self.output / "raw"
    raw_dir.mkdir(exist_ok=False)
    self.path = raw_dir / "episode.h5"
    config = WorkcellConfig(
      model_path=simulation.model_path,
      physics_hz=round(1.0 / simulation.timestep),
      control_hz=100,
      camera_hz=30,
      cameras=tuple(cameras.values()),
      tactile_provider=SolverContactTactileProvider.source,
    )
    self.recorder = EpisodeRecorder(
      self.path,
      simulation,
      config,
      metadata={"recording_contract": "poker_pi05_policy_rollout_v1", **metadata},
      capture_taskspace=True,
      buffer_rows=128,
    )
    self.recorder.record_initial("awaiting_contact")

  def observe(self, simulation, phase: str) -> None:
    self.recorder.observe(simulation, phase)

  def finish(self, report: dict, phase: str) -> dict:
    """Finalize a complete or failed rollout while the renderer is still open."""
    self.recorder.record_terminal(phase)
    outcome = {
      "success": report.get("status") == "success",
      "status": report.get("status", "error"),
      "seed": report["seed"],
      "evaluation": report.get("evaluation"),
      "error": report.get("error"),
    }
    self.recorder.set_outcome(outcome)
    self.recorder.close()
    result_path = self.path.with_suffix(".result.json")
    result_path.write_text(
      json.dumps(outcome, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    validation = validate_episode(self.path)
    if not validation.valid:
      raise RuntimeError(f"invalid poker rollout Raw episode: {validation.errors}")
    curve_path = self.output / "curves/right_hand_force_curves.png"
    plot_right_fingertip_forces(self.path, curve_path)
    return {
      "episode": str(self.path.relative_to(self.output)),
      "result": str(result_path.relative_to(self.output)),
      "right_hand_force_curves": str(curve_path.relative_to(self.output)),
      "state_samples": validation.state_samples,
      "camera_samples": validation.camera_samples,
      "validated": True,
    }

  def close_incomplete(self) -> None:
    if not self.recorder._closed:
      self.recorder.close(finalize=False)


def plot_right_fingertip_forces(source: Path, output: Path) -> None:
  """Plot five fingers' total normal and tangential force from the saved Raw."""
  from PIL import Image, ImageDraw, ImageFont

  with h5py.File(source, "r") as file:
    force = file["tactile_contact_force"]
    names = tuple(force["link_names"].asstr()[:])
    indices = [names.index(name) for name in RIGHT_FINGERTIP_LINK_NAMES]
    timestamp = np.asarray(force["timestamp"], dtype=np.float64)
    normal = np.asarray(force["normal_force_n"][:, indices], dtype=np.float64)
    tangent_xy = np.asarray(force["tangent_force_n"][:, indices], dtype=np.float64)
    tangent = np.linalg.norm(tangent_xy, axis=-1)
    if not all(np.isfinite(value).all() for value in (timestamp, normal, tangent)):
      raise ValueError("recorded fingertip forces contain nonfinite values")
    if timestamp.shape[0] != normal.shape[0]:
      raise ValueError("force and timestamp sample counts differ")

  fingers = ("thumb", "index", "middle", "ring", "little")
  image = Image.new("RGB", (1600, 1120), "#ffffff")
  draw = ImageDraw.Draw(image)
  try:
    font = ImageFont.truetype("DejaVuSans.ttf", 19)
    small_font = ImageFont.truetype("DejaVuSans.ttf", 14)
  except OSError:
    font = small_font = ImageFont.load_default()
  draw.text((35, 16), "Poker pi0.5 rollout | right fingertip contact forces (unfiltered)",
            fill="#182331", font=font)
  start, end = float(timestamp[0]), float(timestamp[-1])
  span = max(end - start, 1e-9)
  for index, finger in enumerate(fingers):
    for column, (values, label, color) in enumerate(
      ((normal, "Normal Fn (N)", "#1264a3"),
       (tangent, "Tangential |Ft| (N)", "#c7651b"))
    ):
      left = 95 + 785 * column
      top = 78 + 202 * index
      right, bottom = left + 690, top + 152
      maximum = max(0.01, float(np.max(values[:, index])) * 1.05)
      draw.rectangle((left, top, right, bottom), outline="#86929f", width=2)
      for grid in (0.25, 0.5, 0.75):
        y = round(bottom - grid * (bottom - top))
        draw.line((left, y, right, y), fill="#d8dfe6", width=1)
      draw.text((left + 5, top + 5), f"{finger} | {label}", fill="#182331", font=small_font)
      draw.text((left - 83, top), f"{maximum:.2f}", fill="#465567", font=small_font)
      draw.text((left - 38, bottom - 17), "0", fill="#465567", font=small_font)
      samples = [
        (round(left + (float(t) - start) / span * (right - left)),
         round(bottom - float(value) / maximum * (bottom - top)))
        for t, value in zip(timestamp, values[:, index], strict=True)
      ]
      if len(samples) > 1:
        draw.line(samples, fill=color, width=2)
      elif samples:
        draw.ellipse((samples[0][0] - 2, samples[0][1] - 2,
                      samples[0][0] + 2, samples[0][1] + 2), fill=color)
      if index == 4:
        draw.text((left, bottom + 4), f"{start:.2f} s", fill="#465567", font=small_font)
        draw.text((right - 85, bottom + 4), f"{end:.2f} s", fill="#465567", font=small_font)
  output.parent.mkdir(exist_ok=False)
  image.save(output)
