"""Task-local geometry and contact-driven ink removal, in SI units."""

import numpy as np

from ...shared.posture import ARM_HOME as ARM_HOME

SCENE_NAME = "whiteboard-wipe"
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
BOARD_TARGET_FORCE_N = 2.2
BOARD_PRELOAD_FORCE_N = 1.8
DEFAULT_XY_JITTER = DEFAULT_YAW_JITTER = 0.0
SCENE_GEOM_NAMES = ("board_surface", "eraser_handle", "eraser_pad")
TABLE_HEIGHT = 0.68
BOARD_CENTER = np.array([0.62, -0.12, 0.825])
BOARD_ROTATION = np.array([[2**-0.5, 0, -(2**-0.5)], [0, 1, 0], [2**-0.5, 0, 2**-0.5]])
BOARD_NORMAL = BOARD_ROTATION[:, 2]
BOARD_SURFACE = BOARD_CENTER + 0.008 * BOARD_NORMAL
PAD_BOTTOM = np.array([0.0, 0.0, -0.015])
INK_COUNT = 25
INK_PATTERN = "z_top_down_v1"
# Board-local x points up the tilted board; y points left when viewed head-on.
# Same nominal 16 cm width / 14 cm height as the former two horizontal strokes.
INK_Z_VERTICES = np.array([[0.07, 0.08], [0.07, -0.08], [-0.07, 0.08], [-0.07, -0.08]])
INK_Z_SEGMENTS = (8, 9, 8)
# Ink center translation in the board plane; keep the frame outside the stroke.
INK_CENTER_RANGE_M = np.array([0.04, 0.06])
MIN_SLIDING_SPEED = 0.005
REQUIRED_SLIDING_DISTANCE = 0.03

INITIAL_ERASER_ROTATION = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
WIPE_ROTATION = BOARD_ROTATION @ INITIAL_ERASER_ROTATION
