"""Independent top-bar ink layout on the unchanged face-up tabletop board."""

import numpy as np

from ...shared.posture import ARM_HOME as ARM_HOME

SCENE_NAME = "whiteboard-wipe-horizontal-line"
MECHANICS_VERSION = "horizontal_fixed_board_top_line_contact_v1"
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
# Loaded lead-in/out must retain the original felt/frame clearance at y +/-6cm.
BOARD_STROKE_CENTER_LIMIT_M = 0.142
DEFAULT_XY_JITTER = DEFAULT_YAW_JITTER = 0.0
SCENE_GEOM_NAMES = ("board_surface", "eraser_handle", "eraser_pad")
TABLE_HEIGHT = 0.68
BOARD_CENTER = np.array([0.52, -0.12, 0.692])
BOARD_ROTATION = np.eye(3)
BOARD_NORMAL = BOARD_ROTATION[:, 2]
BOARD_SURFACE = BOARD_CENTER + 0.008 * BOARD_NORMAL
PAD_BOTTOM = np.array([0.0, 0.0, -0.015])
INK_COUNT = 8
INK_PATTERN = "z_top_bar_v1"
# Keep the top bar at its original Z-relative location; do not recenter it.
INK_LINE_VERTICES = np.array([[0.07, 0.08], [0.07, -0.08]])
# The shared result schema historically names its geometry field INK_Z_VERTICES.
INK_Z_VERTICES = INK_LINE_VERTICES
INK_CENTER_RANGE_M = np.array([0.04, 0.06])
MIN_SLIDING_SPEED = 0.005
REQUIRED_SLIDING_DISTANCE = 0.03

INITIAL_ERASER_ROTATION = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
WIPE_ROTATION = BOARD_ROTATION @ INITIAL_ERASER_ROTATION
