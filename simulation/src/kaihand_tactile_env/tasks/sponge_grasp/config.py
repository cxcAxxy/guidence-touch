"""Task-local constants for moving the vase-wipe sponge into a plate."""

import numpy as np

from ...shared.posture import ARM_HOME as ARM_HOME

SCENE_NAME = "sponge-grasp"
OBJECT_NAMES = ("sponge",)
OBJECT_STABILIZERS = {}
SCENE_GEOM_NAMES = (
  "plate_base",
  *(f"plate_rim_{i:02d}" for i in range(12)),
)
DEFAULT_PLAYBACK_SPEED = 1.0
DEFAULT_XY_JITTER = 0.0
DEFAULT_YAW_JITTER = 0.0
# Batch Raw only: preserve the calibrated fixed-pose compact example.
# Soft-contact acceptance is sensitive to millimetre-scale shifts.  Keep the
# production pickup variation within 1 mm per horizontal axis.
COLLECTION_XY_OFFSET_LOW_M = np.array([-0.001, -0.001])
COLLECTION_XY_OFFSET_HIGH_M = np.array([0.001, 0.001])

# The robot faces +X; negative Y shifts the upright sponge toward its right hand.
SPONGE_TABLE_XY = np.array([0.43, -0.07])
# Fixed dish to the robot's right (negative world Y), clear of the pickup site.
# Its flat interior is 8 mm above the tabletop; the raised lip is 17 mm high.
PLATE_TABLE_XY = np.array([0.62, -0.38])
PLATE_INTERIOR_TOP_Z_M = 0.688
PLATE_INTERIOR_RADIUS_M = 0.082
PICKUP_HAND_YAW_OFFSET_RAD = np.deg2rad(20.0)
PHYSICS_TIMESTEP_S = 0.00025
SPONGE_YOUNG_PA = 20000.0
SPONGE_CONTACT_TIME_S = 0.003
SPONGE_CONTACT_IMPEDANCE = 0.99
SPONGE_SELF_CONTACT_IMPEDANCE = 0.999
SPONGE_CONTACT_MARGIN_M = 0.0030
SPONGE_ELASTIC_DAMPING_S = 0.00005
GRASP_FORCE_TARGETS_N = np.array([2.0, 1.5, 1.2, 1.2, 0.8])
GRASP_CLOSE_SECONDS = 3.2
# Keep every pad physically loaded once the hand has closed, through placement.
MINIMUM_FIVE_FINGER_NORMAL_FORCE_N = 0.05
SETTLED_GRASP_CONTACT_SECONDS = 0.5
# A small initial curl seats the ring and little fingertips.  The ring finger
# then follows the sponge's elastic retreat as it leaves the tabletop, rather
# than crushing it at the initial grasp plane.
RING_GRASP_JOINT2_EXTRA_RAD = 0.04
PINKY_GRASP_JOINT2_EXTRA_RAD = 0.05
PINKY_GRASP_JOINT3_EXTRA_RAD = 0.06
RING_LIFT_FOLLOW_EXTRA_RAD = 0.025
RING_LIFT_FOLLOW_SECONDS = 1.2
GRASP_PATCH_RADIUS_M = 0.025
LIFT_SECONDS = 2.0
CARRY_SECONDS = 3.0
LOWER_TO_PLATE_SECONDS = 3.0
RELEASE_SECONDS = 2.0
RETREAT_SECONDS = 1.0
SETTLE_SECONDS = 1.0
PICK_TO_PLACE_SECONDS = (
  LIFT_SECONDS + CARRY_SECONDS + LOWER_TO_PLATE_SECONDS + RELEASE_SECONDS
)
