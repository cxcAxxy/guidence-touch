"""Read-only target-to-actual fingertip and actuator diagnostics."""

import mujoco
import numpy as np

FINGERS = ("thumb", "index", "middle", "ring", "pinky")


def execution_sample(sim, controller, targets, index):
  actual = []
  distances = []
  alignments = []
  normal = controller._card_rotation()[:, 2]
  for finger in FINGERS:
    link = f"hand_r_{finger}_{'link6' if finger == 'thumb' else 'link4'}"
    actual.append(sim.data.site_xpos[sim.model.site(f"{link}_site").id].copy())
    distances.append(
      float(
        mujoco.mj_geomDistance(
          sim.model,
          sim.data,
          sim.model.geom(f"{link}_tactile_pad_col").id,
          sim.model.geom("card_core_geom").id,
          1.0,
          None,
        )
      )
    )
    axis = sim.data.xmat[sim.model.body(link).id].reshape(3, 3)[:, 2]
    alignments.append(float(np.rad2deg(np.arcsin(np.clip(abs(axis @ normal), 0, 1)))))
  measured = np.asarray(actual)
  tip_error = np.linalg.norm(measured - targets.fingertips_world[index], axis=1)
  wrist, _ = sim.current_pose_matrix("right")
  return {
    "time_s": float(sim.data.time),
    "pose_time_s": float(sim.observation_time),
    "mode": controller.mode,
    "drive_limit_n": sim.drive_limit_n,
    "drive_state": sim.drive_state(),
    "fingers": list(FINGERS),
    "wrist_target_world": targets.site_positions_world[index].tolist(),
    "wrist_measured_world": wrist.tolist(),
    "wrist_error_m": float(np.linalg.norm(wrist - targets.site_positions_world[index])),
    "fingertip_target_world": targets.fingertips_world[index].tolist(),
    "fingertip_measured_world": measured.tolist(),
    "fingertip_error_m": tip_error.tolist(),
    "pad_card_distance_m_capped_at_1m": distances,
    "pad_plane_angle_deg": alignments,
    "force_n": controller.terminal_metrics()["force_n"],
    "card_pose": sim.object_pose("card").tolist(),
  }
