"""Face-up fixed tabletop board; SI units and the original physical ink gates."""

import numpy as np

from ...shared.posture import ARM_HOME as ARM_HOME

SCENE_NAME = "whiteboard-wipe-horizontal"
MECHANICS_VERSION = "horizontal_fixed_board_z_contact_v1"
OBJECT_NAMES = ("eraser",)
OBJECT_STABILIZERS = {}
DEFAULT_PLAYBACK_SPEED = 1.0
CONTROL_PERIOD_S = 0.01
BOARD_CONTROL_PERIOD_S = 0.005
HAND_VELOCITY_GAIN = 1.0
PICKUP_CLEARANCE_M = 0.12
OPEN_HAND_SETTLE_S = 0.6
OPEN_HAND_MAX_ERROR_RAD = 0.06
MOVE_ABOVE_ERASER_S = 3.0
DESCEND_TO_ERASER_S = 2.0
SHAPE_GRASP_S = 1.5
TACTILE_GRASP_S = 4.0
BOARD_TARGET_FORCE_N = 1.85
BOARD_PRELOAD_FORCE_N = 1.6
# A 2 mm longer loaded lead at extreme ink translations avoids an edge patch
# missing its 30 mm physical stroke gate. The felt still stays inside the frame.
BOARD_STROKE_CENTER_LIMIT_M = 0.142
DEFAULT_XY_JITTER = DEFAULT_YAW_JITTER = 0.0
SCENE_GEOM_NAMES = ("board_surface", "eraser_handle", "eraser_pad")
TABLE_HEIGHT = 0.68
# The frame underside rests at table height. The fixed board cannot be pushed
# away; this task intentionally changes placement, not board attachment.
BOARD_CENTER = np.array([0.52, -0.12, 0.692])
BOARD_ROTATION = np.eye(3)
BOARD_NORMAL = BOARD_ROTATION[:, 2]
BOARD_SURFACE = BOARD_CENTER + 0.008 * BOARD_NORMAL
PAD_BOTTOM = np.array([0.0, 0.0, -0.015])
INK_COUNT = 25
INK_PATTERN = "z_top_down_v1"
# "Top down" is board-local, not world-Z, on the face-up variant.
INK_Z_VERTICES = np.array([[0.07, 0.08], [0.07, -0.08], [-0.07, 0.08], [-0.07, -0.08]])
INK_Z_SEGMENTS = (8, 9, 8)
INK_CENTER_RANGE_M = np.array([0.04, 0.06])
MIN_SLIDING_SPEED = 0.005
REQUIRED_SLIDING_DISTANCE = 0.03

INITIAL_ERASER_ROTATION = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
WIPE_ROTATION = BOARD_ROTATION @ INITIAL_ERASER_ROTATION
