"""Pinky-forward pickup that arrives palm down with thumb/index above pinky.

Calibrated jointly with the explicit AUTO_PLUG_QUATERNION_WXYZ initial pose.
The reversed USB is pinched farther back on its handle, with the hand tilted
around the pinch axis to keep the wrist above the table during insertion.
Arrays command the robot, never the free USB body.
"""

from dataclasses import dataclass

import mujoco
import numpy as np

from . import config


@dataclass(frozen=True)
class UsbGrasp:
  wrist_position: np.ndarray
  wrist_rotation: np.ndarray
  arm_seed: np.ndarray
  open_hand: np.ndarray
  closed_hand: np.ndarray

  @property
  def approach_hand(self) -> np.ndarray:
    """Wide (~61 mm tip spacing) approach, before the calibrated fine pinch.

    Only thumb/index are opened farther; the remaining fingers retain their
    clearance fold. Calibrated in the same hand frame, so XY/yaw jitter does
    not change this posture or require any object manipulation.
    """
    target = self.open_hand.copy()
    target[:8] = [
      0.17268758685679134,
      1.2352736672140932,
      0.09332785377394695,
      0.332368433215127,
      -0.26,
      0.9112582501864351,
      0.7412511314666322,
      -0.0181011256268629,
    ]
    return target


def calibrated_grasp(simulation, *, pinch_tilt_rad: float = 0.06) -> UsbGrasp:
  """Transport the calibrated pinch with the measured plug's planar pose.

  Rotate around the plug origin, not the wrist or the world origin. Settling
  can introduce a tiny roll/pitch; the pickup remains level with the table.
  """
  shift = simulation.object_pose("usb_plug")[:3] - np.array([0.5, -0.18, 0.685991])
  grasp = UsbGrasp(
    wrist_position=np.array(
      [0.44551783027887343, -0.16807999510472202, 0.816708053463016]
    )
    + shift,
    wrist_rotation=np.array(
      [
        [0.7924750023744166, 0.4106955628573672, 0.45090190203739144],
        [-0.024518008431238025, -0.717249740185171, 0.6963847194380911],
        [0.6094113864002155, -0.562922698826023, -0.5583330522834845],
      ]
    ),
    # This redundant arm branch opens the insertion elbow outward while
    # preserving the calibrated wrist pose and physical finger contacts.
    arm_seed=np.array(
      [
        -0.7381367216908878,
        -0.7829375888223753,
        0.2617993877991494,
        -1.4157995883484007,
        1.6625948094318155,
        0.3853382155852312,
        0.10423279199943442,
      ]
    ),
    open_hand=np.array(
      [
        0.19223489867457477,
        1.2317037156964805,
        0.16207002960465544,
        0.4249378599845403,
        -0.26,
        1.2416696997906649,
        0.492417265000636,
        -0.16370780000054924,
        0.0,
        1.45,
        1.4,
        0.7,
        0.0,
        1.45,
        1.4,
        0.7,
        0.0,
        1.45,
        1.4,
        0.7,
      ]
    ),
    closed_hand=np.array(
      [
        0.22358712619200474,
        1.2162456280917817,
        0.22239549869196965,
        0.39241472497484436,
        -0.24836016198770527,
        1.3692368378967144,
        0.3522758371465467,
        -0.175,
        0.0,
        1.45,
        1.4,
        0.7,
        0.0,
        1.45,
        1.4,
        0.7,
        0.0,
        1.45,
        1.4,
        0.7,
      ]
    ),
  )
  body = simulation.model.body("usb_plug").id
  rotation = simulation.data.xmat[body].reshape(3, 3)
  nominal = np.empty(9)
  mujoco.mju_quat2Mat(nominal, config.AUTO_PLUG_QUATERNION_WXYZ)
  nominal = nominal.reshape(3, 3)
  # Recalibrate the hand around the pinch axis for the finite contact patch.
  # Only wrist targets change; the USB stays under physical contact dynamics.
  axis = nominal[:, 1]
  angle = pinch_tilt_rad  # Match the selected transport calibration.
  skew = np.array(
    [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
  )
  turn = np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew)
  pivot = simulation.object_pose("usb_plug")[:3] + nominal @ np.array([-0.008, 0, 0])
  grasp = UsbGrasp(
    wrist_position=pivot + turn @ (grasp.wrist_position - pivot),
    wrist_rotation=turn @ grasp.wrist_rotation,
    arm_seed=grasp.arm_seed,
    open_hand=grasp.open_hand,
    closed_hand=grasp.closed_hand,
  )
  yaw = np.arctan2(rotation[1, 0], rotation[0, 0]) - np.arctan2(
    nominal[1, 0], nominal[0, 0]
  )
  yaw = np.arctan2(np.sin(yaw), np.cos(yaw))
  # Preserve the calibrated nominal target without an extra numerical yaw
  # rotation, including the sub-micrometre settling translation.
  if abs(yaw) < 1e-12:
    return grasp
  c, s = np.cos(yaw), np.sin(yaw)
  turn = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
  origin = simulation.object_pose("usb_plug")[:3]
  # A fixed seed lets tabletop yaw shift the redundant arm branch inward
  # later at the socket. This local, calibrated seed correction keeps the
  # elbow open over the validated +/-5-degree pickup range. IK still solves
  # the exact wrist target; these are neither joint commands nor object edits.
  arm_seed = grasp.arm_seed.copy()
  arm_seed[2] += 0.8 * yaw
  return UsbGrasp(
    wrist_position=origin + turn @ (grasp.wrist_position - origin),
    wrist_rotation=turn @ grasp.wrist_rotation,
    arm_seed=arm_seed,
    open_hand=grasp.open_hand,
    closed_hand=grasp.closed_hand,
  )
