"""Actuator-only pickup, horizontal contact wiping, replacement and release."""

import numpy as np

from ...shared.simulation import HAND_JOINT_NAMES
from ..whiteboard_wipe.execution import WhiteboardWipeExecutor


class HorizontalWhiteboardWipeExecutor(WhiteboardWipeExecutor):
  """Use the shared force-feedback policy with the face-up task's board frame."""

  def run(self):
    result = super().run()
    result["mechanics_version"] = self.config.MECHANICS_VERSION
    result["board_placement"] = "fixed_horizontal_face_up"
    result["board_stroke_center_limit_m"] = self.config.BOARD_STROKE_CENTER_LIMIT_M
    result["board_preload_force_n"] = self.config.BOARD_PRELOAD_FORCE_N
    result["wipe_direction"] = "board_local_top_to_bottom"
    result["home_return_verified"] = False
    if not result["success"]:
      return result
    try:
      # The original replacement policy releases to its bent pre-grasp pose.
      # Restore the shared neutral hand while already safely above the table.
      self.move_hand(self.sim.spread_grip, 2.0, "open_for_home")
      self.move_wrist(self.position, 0.8, "settle_home_hand")
      # The wipe IK branch may finish with a negative third joint. A direct
      # interpolation to home folds that branch through robot self-contact.
      # Restore the calibrated, elevated pickup branch first, then retrace the
      # already verified shared-home approach with a neutral, empty hand.
      result["restore_approach_motion"] = self.move_arm_joints(
        self.sim.approach_arm, 8.0, "restore_approach_branch"
      )
      restore = result["restore_approach_motion"]
      if (
        restore["maximum_joint_speed_deg_s"] >= 70.0
        or restore["maximum_joint_acceleration_deg_s2"] >= 100.0
      ):
        raise RuntimeError("restore_approach_branch: arm return was not smooth")
      result["return_home_motion"] = self.move_arm_joints(
        self.config.ARM_HOME["right"], 5.0, "home"
      )
      motion = result["return_home_motion"]
      if (
        motion["maximum_joint_speed_deg_s"] >= 70.0
        or motion["maximum_joint_acceleration_deg_s2"] >= 100.0
      ):
        raise RuntimeError("home: arm joint return was not smooth")
      self.move_wrist(self.position, 0.5, "verify_home")
      right_error = float(
        np.max(
          abs(
            self.sim.data.qpos[self.sim._arm_qpos["right"]]
            - self.config.ARM_HOME["right"]
          )
        )
      )
      released = bool(
        self.sim.table_force > 0.2
        and np.linalg.norm(self.sim.object_twist("eraser")[:3]) < 0.01
        and self.sim.forces.read(self.sim.data).normal_force_n[5:].sum() < 0.05
      )
      hand = np.array(
        [
          self.sim.data.qpos[self.sim._qpos_address[name]]
          for name in HAND_JOINT_NAMES["right"]
        ]
      )
      hand_error = float(np.max(abs(hand - self.sim.spread_grip)))
      result["home_return_verified"] = bool(
        right_error < 0.01
        and hand_error < self.config.OPEN_HAND_MAX_ERROR_RAD
        and released
        and self.sim.peak_direct_hand_board_force < 1e-9
      )
      result["right_arm_home_error_rad"] = right_error
      result["right_hand_home_error_rad"] = hand_error
      if not result["home_return_verified"]:
        result["success"] = False
        result["reason"] = "Home return or freely supported eraser was not verified"
    except RuntimeError as exc:
      result["success"] = False
      result["reason"] = str(exc)
    result["elapsed_s"] = float(self.sim.data.time)
    result["peak_direct_hand_board_force_n"] = self.sim.peak_direct_hand_board_force
    result["solver_warnings"] = [int(w.number) for w in self.sim.data.warning]
    return result


def run_episode(simulation, observer=None):
  return HorizontalWhiteboardWipeExecutor(simulation, observer=observer).run()
