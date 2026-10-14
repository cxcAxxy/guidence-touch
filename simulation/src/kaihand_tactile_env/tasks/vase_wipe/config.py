"""Task-local dimensions and initial grasp; SI units throughout."""

import numpy as np

from ...shared.posture import ARM_HOME as ARM_HOME

SCENE_NAME = "vase-wipe"
OBJECT_NAMES = ("sponge",)
OBJECT_STABILIZERS = {}
DEFAULT_PLAYBACK_SPEED = 1.0
DEFAULT_XY_JITTER = 0.0
DEFAULT_YAW_JITTER = 0.0
VASE_CENTER = np.array([0.58, -0.18, 0.68])
SPONGE_TABLE_XY = np.array([0.43, -0.40])
PICKUP_HAND_YAW_OFFSET_RAD = np.deg2rad(-15.0)
# Height/radius in metres; the shoulder and neck have real matching collision.
INNER_PROFILE = (
  (0.022, 0.066),
  (0.038, 0.084),
  (0.057, 0.099),
  (0.076, 0.106),
  (0.095, 0.104),
  (0.111, 0.095),
  (0.127, 0.0855),
  (0.143, 0.076),
  (0.153, 0.074),
  (0.163, 0.076),
  (0.169, 0.080),
  (0.174, 0.080),
)
INNER_PROFILE = tuple((z * (0.220 / 0.174), r * 0.85) for z, r in INNER_PROFILE)
INNER_PROFILE = ((0.012, 0.050), *INNER_PROFILE)
FLOOR_HEIGHT = 0.100
WALL_COUNT = 24
STAIN_HEIGHTS = (0.194, 0.197, 0.200)
# Tilt the flat face into the wall; the cuboid itself has no flared tip.
SPONGE_WIPE_PITCH_RAD = np.deg2rad(-15.0)
SPONGE_WIPE_ROLL_RAD = np.deg2rad(-4.5)
SPONGE_WIPE_YAW_RAD = np.deg2rad(40.0)
ENTRY_HEIGHT_M = 1.055
STAIN_ANGLE_RAD = 0.14
STAIN_ANGLE_SPACING_RAD = 0.015
# Ivory ceramic and a cool light-black stain keep the stain readable without
# relying on the old red/green color contrast.
VASE_IVORY_RGB = (0.88, 0.84, 0.74)
VASE_CLAY_RGBA = (0.84, 0.80, 0.69, 1.0)
STAIN_COLOR = (0.32, 0.35, 0.38, 1.0)
STAIN_OPACITY_GAMMA = 0.35


def inner_radius(height):
  profile = np.array(INNER_PROFILE)
  return float(np.interp(height, profile[:, 0], profile[:, 1]))


PATCH_ROWS = 3
PATCH_COLUMNS = 7
TARGET_WALL_FORCE_N = 0.65
# Complete lateral cycles (2.8 s each). The initial sweep includes four cycles
# so the widened face covers the whole compact stain cluster without re-entry.
WIPE_SECONDS = 11.2
RESIDUAL_WIPE_SECONDS = 3.8
MAX_WIPE_PASSES = 6
MIN_WALL_NORMAL_N = 0.20
MAX_CLEAN_NORMAL_N = 5.0
MIN_WALL_FRICTION_N = 0.30
MIN_PATCH_FRICTION_N = 0.025
MIN_WIPE_SPEED_M_S = 0.005
# Light surface dirt: less accumulated work/travel, with sustained loaded
# sliding still required. Force and speed gates remain unchanged.
PATCH_WORK_REQUIRED_J = 0.000045
PATCH_STROKE_REQUIRED_M = 0.0016
PATCH_DWELL_REQUIRED_S = 0.14
MAX_MEAN_DIRT = 0.05
MAX_PATCH_DIRT = 0.10
SPONGE_YOUNG_PA = 20000.0
# Material elasticity supplies the softness; collision constraints must not
# stand in for foam compression. Positive margin activates contact before overlap.
PHYSICS_TIMESTEP_S = 0.00025
SPONGE_CONTACT_TIME_S = 0.003
SPONGE_CONTACT_IMPEDANCE = 0.99
SPONGE_SELF_CONTACT_IMPEDANCE = 0.999
SPONGE_CONTACT_MARGIN_M = 0.0030
SPONGE_ELASTIC_DAMPING_S = 0.00005
GRASP_FORCE_TARGETS_N = np.array([2.0, 1.5, 1.2, 1.2, 0.8])
GRASP_CLOSE_SECONDS = 3.2
# Effective radius of the distributed grasp contact patch, within the 30 mm
# half-width of the cuboid. This bounds a compliant moment, never a pose lock.
GRASP_PATCH_RADIUS_M = 0.025
SCENE_GEOM_NAMES = (
  "vase_floor",
  "vase_foot",
  "vase_visual",
  "sponge_reference",
  *(f"vase_wall_{i}" for i in range(WALL_COUNT * (len(INNER_PROFILE) - 1))),
  *(f"stain_{i}" for i in range(PATCH_ROWS * PATCH_COLUMNS)),
)
