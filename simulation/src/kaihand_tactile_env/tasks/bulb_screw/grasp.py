"""Task-local pinch and five-finger grasps fitted to shared tactile surfaces."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BulbGrasp:
  wrist_position: np.ndarray
  wrist_rotation: np.ndarray
  open_hand: np.ndarray
  closed_hand: np.ndarray


def calibrated_grasp(simulation, *, five_finger: bool = False) -> BulbGrasp:
  """Place opposing pads around the 58 mm globe, with the wrist above it.

  Five-finger targets keep middle/ring abduction neutral to avoid finger
  collisions; a contact-force controller completes their closure in execution.
  Thumb/index targets were fitted to the physical probe surface centroids and
  normals. Other fingers curl out of the pinch. Only robot goals are returned;
  neither live joint state nor the bulb is moved by this function.
  """
  position, rotation = simulation.current_pose_matrix("right")
  hand = simulation.model.body("hand_r_base_link").id
  hand_rotation = simulation.data.xmat[hand].reshape(3, 3)
  ee_to_hand = rotation.T @ hand_rotation
  hand_offset = rotation.T @ (simulation.data.xpos[hand] - position)
  angle = np.arctan2(0.604, 0.797)
  target_hand_rotation = np.array(
    [
      [0, -np.sin(angle), -np.cos(angle)],
      [1, 0, 0],
      [0, -np.cos(angle), np.sin(angle)],
    ]
  )
  target_rotation = target_hand_rotation @ ee_to_hand.T
  bulb_center = simulation.object_pose("bulb")[:3] + [0, 0, 0.072]
  local_center = (
    np.array([-0.002542103750781093, 0.06556771771920465, -0.11947452673060487])
    if five_finger
    else np.array([0.035, 0.075, -0.110])
  )
  target_position = (
    bulb_center - target_hand_rotation @ local_center - target_rotation @ hand_offset
  )
  if five_finger:
    return BulbGrasp(
      target_position,
      target_rotation,
      np.array(
        [
          0.349,
          1.2838365038733797,
          0.4392531463250977,
          0.3560852725920311,
          -0.05,
          0.18254707429558065,
          0.6473034999362802,
          0.6819559756428133,
          0.0,
          0.07823425129608905,
          0.6284525402943059,
          0.7926713167802996,
          0.0,
          0.036943871096746544,
          0.5814019668479246,
          0.757808083576648,
          0.1,
          0.1491839753677753,
          0.5255267794348297,
          0.5795737054027612,
        ]
      ),
      np.array(
        [
          0.1084806651409939,
          1.745,
          0.698,
          0.2506213716750889,
          -0.05,
          0.41975287218700724,
          1.128222807697052,
          0.2955066877637926,
          0.0,
          0.0699244285588622,
          1.3095948091124703,
          0.1239672736067599,
          0.0,
          0.005823374044853006,
          1.3530455097020533,
          -0.030395955994185833,
          0.1,
          0.5113591279564549,
          0.6330320055153862,
          0.5048437068085506,
        ]
      ),
    )
  # The old identical tuck put ``middle_link3`` on the globe throughout every
  # wrist-turn tightening sample, making the nominal two-pad pinch depend on a
  # hidden third support.  Keep ring and pinky curled away, but unfold and
  # abduct only the middle finger far enough that the strict clean-contact
  # tightening window is carried by the thumb and index pads.
  middle_tucked = [-0.259, 0.626, 1.198, 1.570]
  outer_tucked = [0, 1.45, 1.4, 0.7]
  tucked = [*middle_tucked, *outer_tucked, *outer_tucked]
  return BulbGrasp(
    target_position,
    target_rotation,
    np.r_[
      [
        -0.1901870193,
        1.5556383849,
        0.2912164744,
        0.4197002694,
        0.1818568262,
        0.0451699808,
        1.0987649205,
        0.2493225120,
      ],
      tucked,
    ],
    np.r_[
      [
        -0.1904808559,
        1.5558956129,
        0.5113861958,
        0.3118115817,
        0.1830929115,
        0.2000552192,
        1.3114439441,
        -0.0821579274,
      ],
      tucked,
    ],
  )
