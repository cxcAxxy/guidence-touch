"""Calibrated configuration for the poker-draw task."""

from __future__ import annotations

import numpy as np

from ...shared.posture import ARM_HOME as ARM_HOME

SCENE_NAME = "poker-draw"
OBJECT_NAMES = ("card",)
SCENE_GEOM_NAMES = (
  "poker_table_base",
  "poker_table_top",
  "card_core_geom",
  "card_back_visual",
  "card_face_visual",
)
OBJECT_STABILIZERS: dict[tuple[str, str], str] = {}
DEFAULT_PLAYBACK_SPEED = 1.25
# The force baseline is deterministic.  Re-enable pose randomization explicitly
# for separate robustness experiments, not during friction calibration.
DEFAULT_XY_JITTER = 0.0
DEFAULT_YAW_JITTER = 0.0
# Force-controlled four-finger press used before and throughout the draw.
# This is a measured card-pad normal-force target for each finger, not an
# actuator effort or a fixed joint offset.  The initial value is deliberately
# conservative and is calibrated by the bounded headless press experiment.
DEFAULT_PRESS_FORCE_PER_FINGER_N = 0.35

_SIDE = "right"
_FINGERS = ("index", "middle", "ring", "pinky")
_DRAW_FINGER_DEGREES = (19.70, 20.10, 20.40, 20.55)
_FINGER_ABDUCTION_DEGREES = (0.0, 0.0, 0.0, 0.0)
# Tip-pad draw posture v7: visibly curl the finger roots (about 47--49 degrees)
# and raise palm/wrist about 23 mm relative to v6. The distal tactile pads
# contact the card near their tips at about 38--40 degrees, not flush/flat.
# This user-requested contact mode has its own DRAW-only angle limit; physical
# force/contact checks and the original PINCH angle/alignment gates remain.
# Finger-specific q2/q3 sharing equalizes pad support height (mesh minima),
# not just pad centres, so the pitched tactile surfaces still meet the card.
# A small yaw keeps the curved thumb and palm clear of the mini-table;
# it does not change the world -X drawing direction.
# The historical _FLAT_DRAW_* names are retained for callers, not pad flatness.
_FLAT_DRAW_POSTURE_VERSION = "tip-pad-v7"
_FLAT_DRAW_MAXIMUM_FINGERTIP_PLANE_ANGLE_DEGREES = 50.0
_FLAT_DRAW_WRIST_PITCH_DEGREES = -8.0
_FLAT_DRAW_WRIST_YAW_DEGREES = 12.0
_FLAT_DRAW_FINGER_DEGREES = np.array(
  [
    [0.0, 47.175720, 9.264280, -10.0],
    [0.0, 49.365781, 7.634219, -10.0],
    [0.0, 47.990043, 9.569957, -10.0],
    [0.0, 46.648730, 10.731270, -10.0],
  ]
)
_FLAT_DRAW_PRECONTACT_OFFSET = np.array([-0.136498, -0.028086, 0.067226])
_FLAT_DRAW_THUMB_DEGREES = {
  "thumb_joint1": 0.0,
  "thumb_joint2": np.rad2deg(0.25),
  "thumb_joint3": np.rad2deg(0.05),
  "thumb_joint5": 12.0,
}
_OPEN_THUMB_DEGREES = {
  "thumb_joint1": 0.0,
  "thumb_joint2": np.rad2deg(0.25),
  "thumb_joint3": np.rad2deg(0.05),
  "thumb_joint5": 0.0,
}
# The proximal joints curl naturally around the exposed edge while joint3/4
# compensate that bend. Consequently each link4 longitudinal axis remains
# along the card plane and its volar taxel patch forms the upper jaw.
_FLAT_PINCH_FINGER_DEGREES = np.array(
  [
    [14.0, 75.011, 9.6, -10.0],
    [-5.0, 84.681, 0.0, -10.0],
    [3.0, 76.247, 8.3, -10.0],
    [10.0, 64.874, 19.6, -10.0],
  ]
)
_FLAT_PINCH_THUMB_DEGREES = {
  "thumb_joint1": 14.1985,
  "thumb_joint2": 87.955,
  "thumb_joint3": 38.6597,
  "thumb_joint5": 16.6825,
}
# Short-path edge pickup: the thumb scoops underneath the exposed card into
# the existing opposed pad jaw. The card may pitch naturally while closing;
# no object constraint or relaxed terminal contact/force gate is used.
# The old -55 degree world-horizontal jaw required the opposite joint-5
# branch, producing a ~300 degree forearm winding at the table edge.
_FLAT_PINCH_WRIST_PITCH_DEGREES = -35.0
_FLAT_PINCH_WRIST_YAW_DEGREES = 10.0
_FLAT_PINCH_CONTACT_OFFSET = np.array([-0.14679888, -0.02583891, 0.03341511])
_FACE_NORMAL_COSINE = 0.85
_INSPECTION_TARGET_CARD_POSITION = np.array([0.550, -0.100, 1.050])
_MINIMUM_PINCH_NORMAL_FORCE = 0.05
_MINIMUM_FINGER_PAD_ALIGNMENT = 0.94
_MINIMUM_THUMB_PAD_ALIGNMENT = 0.84
_MAXIMUM_FINGERTIP_PLANE_ANGLE_DEGREES = 21.0
_PRESS_FORCE_CONTACT_RATIO = 0.25
_PRESS_FORCE_TARGET_TOLERANCE_N = 0.075
_PRESS_FORCE_ESTABLISH_TOLERANCE_N = 0.035
_PRESS_FORCE_STABLE_DURATION = 0.08
_PRESS_FORCE_ESTABLISH_TIMEOUT = 12.0
_PRESS_FORCE_RECOVERY_STABLE_DURATION = 0.02
_PRESS_FORCE_RECOVERY_TIMEOUT = 0.60
_PRESS_FORCE_SLIDE_ACCELERATION_DURATION = 0.50
_PRESS_FORCE_SLIDE_SPEED_M_S = 0.015
# Successful baseline samples require all four pads to bear load at every
# physics sample; reaching the edge after a contact gap is not sufficient.
_PRESS_FORCE_MINIMUM_CONTACT_FRACTION = 1.0
_PRESS_FORCE_MINIMUM_TARGET_BAND_FRACTION = 0.95
_PRESS_FORCE_MINIMUM_ALL_TARGET_BAND_FRACTION = 0.90
_PRESS_FORCE_MAXIMUM_CONTACT_GAP = 0.0
_PRESS_FORCE_MEAN_TOLERANCE_N = 0.035
_PRESS_FORCE_CONTROL_PERIOD = 0.010
_PRESS_FORCE_FILTER_TIME_CONSTANT = 0.020
_PRESS_FORCE_INTEGRAL_GAIN_RAD_PER_N_S = np.deg2rad(2.0)
_PRESS_FORCE_MAXIMUM_OFFSET_RAD = np.deg2rad(5.0)
_PRESS_FORCE_MAXIMUM_OFFSET_RATE_RAD_S = np.deg2rad(1.5)
_PRESS_FORCE_CONTACT_RECOVERY_RATE_RAD_S = np.deg2rad(1.2)
_PRESS_FORCE_DEADBAND_N = 0.010

__all__ = [
  "ARM_HOME",
  "DEFAULT_PRESS_FORCE_PER_FINGER_N",
  "DEFAULT_PLAYBACK_SPEED",
  "DEFAULT_XY_JITTER",
  "DEFAULT_YAW_JITTER",
  "OBJECT_NAMES",
  "OBJECT_STABILIZERS",
  "SCENE_GEOM_NAMES",
  "SCENE_NAME",
]
