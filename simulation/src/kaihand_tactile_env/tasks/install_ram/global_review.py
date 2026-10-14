"""Presentation-only global video reconstructed from saved physical states."""

import csv
import json

import mujoco
import numpy as np
from PIL import Image

from ...shared.poker_review import _probe_video, _validate_probe, plan_review_frames
from ...shared.render_backend import prepare_render_backend
from ...shared.task_video import _FfmpegPipeWriter
from .task import RamInstallSimulation


def export_global_review(file, output):
  """Use the composite video's clock; never advance or modify the raw episode."""
  sim = RamInstallSimulation()
  frames = plan_review_frames(file, fps=10, second_camera="right_wrist")
  camera = mujoco.MjvCamera()
  camera.type = mujoco.mjtCamera.mjCAMERA_FREE
  camera.lookat[:] = (0.25, 0.0, 0.85)
  camera.distance = 2.25
  camera.azimuth = 135
  camera.elevation = -22
  option = mujoco.MjvOption()
  option.geomgroup[3:] = 0
  prepare_render_backend()
  writer = _FfmpegPipeWriter(output / "global.mp4", fps=10, width=960, height=540)
  try:
    with mujoco.Renderer(sim.model, width=960, height=540) as renderer:
      with (output / "global_frames.csv").open("w", newline="") as stream:
        table = csv.DictWriter(
          stream, fieldnames=["output_index", "state_index", "timestamp_s"]
        )
        table.writeheader()
        for frame in frames:
          index = int(file["cameras/head/state_index"][frame.camera_index])
          sim.data.time = float(file["state/timestamp"][index])
          sim.data.qpos[:] = file["state/qpos"][index]
          sim.data.qvel[:] = file["state/qvel"][index]
          sim.data.ctrl[:] = file["install_ram/actuator_control"][index]
          if "eq_active" in file["install_ram"]:
            sim.data.eq_active[:] = file["install_ram/eq_active"][index]
          mujoco.mj_forward(sim.model, sim.data)
          renderer.update_scene(sim.data, camera=camera, scene_option=option)
          rgb = np.asarray(renderer.render()).copy()
          writer.write(rgb)
          if frame.output_index in (0, len(frames) - 1):
            label = "initial" if frame.output_index == 0 else "final"
            Image.fromarray(rgb).save(output / f"global_{label}.png")
          table.writerow(
            dict(
              output_index=frame.output_index,
              state_index=index,
              timestamp_s=sim.data.time,
            )
          )
    writer.finish()
  except BaseException:
    writer.abort()
    raise
  probe = _probe_video(output / "global.mp4")
  _validate_probe(probe, count=len(frames), fps=10, width=960, height=540)
  report = {
    "source": "../raw/install_ram_000000.h5",
    "method": "Offline rendering of exact saved states; no physics stepping; presentation camera only",
    "camera": {
      "lookat": camera.lookat.tolist(),
      "distance": camera.distance,
      "azimuth": camera.azimuth,
      "elevation": camera.elevation,
    },
    "timeline": "global_frames.csv",
    "video_validation": probe,
  }
  (output / "global_review.json").write_text(json.dumps(report, indent=2) + "\n")
  return report
