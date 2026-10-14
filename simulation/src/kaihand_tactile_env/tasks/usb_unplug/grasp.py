"""Calibrated thumb/index grasp of the exposed handle of a seated USB plug."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class UsbUnplugGrasp:
  wrist_position: np.ndarray
  wrist_rotation: np.ndarray
  arm_seed: np.ndarray
  approach_hand: np.ndarray
  closed_hand: np.ndarray


# Captured from the first physical bottom-out sample of the validated
# usb-insert expert.  Store the hand-to-object transform, not a world pose, so
# small passive settling remains observable and the target follows it.
_NOMINAL_OBJECT_POSITION = np.array([0.620000578, -0.179998232, 0.735035024])
_NOMINAL_OBJECT_QUATERNION = np.array(
  [0.707108567, -0.000208756805, 0.707104958, 0.0000917944901]
)
_NOMINAL_OBJECT_ROTATION_FLAT = np.empty(9)
mujoco.mju_quat2Mat(_NOMINAL_OBJECT_ROTATION_FLAT, _NOMINAL_OBJECT_QUATERNION)
_NOMINAL_OBJECT_ROTATION = _NOMINAL_OBJECT_ROTATION_FLAT.reshape(3, 3)
_NOMINAL_WRIST_POSITION = np.array([0.49162831, -0.16310119, 0.79412282])
_NOMINAL_WRIST_ROTATION = np.array(
  [
    [-0.66193843, 0.51320795, 0.54631046],
    [-0.35614909, 0.42597087, -0.83168903],
    [-0.65954177, -0.74509490, -0.09918794],
  ]
)
_WRIST_OFFSET_OBJECT = _NOMINAL_OBJECT_ROTATION.T @ (
  _NOMINAL_WRIST_POSITION - _NOMINAL_OBJECT_POSITION
)
_WRIST_ROTATION_OBJECT = _NOMINAL_OBJECT_ROTATION.T @ _NOMINAL_WRIST_ROTATION

_ARM_SEED = np.array(
  [
    -1.10016992,
    -1.73854029,
    1.45832646,
    -0.89936268,
    2.76673276,
    -0.62749173,
    1.07267684,
  ]
)

# The wide posture is the insertion expert's collision-free pre-contact shape;
# the closed posture is its verified bottom-out pinch.  No free-body state is
# encoded here and the executor still requires measured bilateral contact.
_APPROACH_HAND = np.array(
  [
    0.17268758685679134,
    1.2352736672140932,
    0.09332785377394695,
    0.332368433215127,
    -0.26,
    0.9112582501864351,
    0.7412511314666322,
    -0.0181011256268629,
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
)
_CLOSED_HAND = np.array(
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
)


def seated_grasp(simulation) -> UsbUnplugGrasp:
  """Follow the measured installed plug without changing its state."""
  body = simulation.model.body("usb_plug").id
  position = simulation.data.xpos[body].copy()
  rotation = simulation.data.xmat[body].reshape(3, 3).copy()
  return UsbUnplugGrasp(
    wrist_position=position + rotation @ _WRIST_OFFSET_OBJECT,
    wrist_rotation=rotation @ _WRIST_ROTATION_OBJECT,
    arm_seed=_ARM_SEED.copy(),
    approach_hand=_APPROACH_HAND.copy(),
    closed_hand=_CLOSED_HAND.copy(),
  )


__all__ = ["UsbUnplugGrasp", "seated_grasp"]
