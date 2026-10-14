"""Task contract for removing the installed bulb and placing it on bare table."""

from __future__ import annotations

import numpy as np

from ..bulb_screw import config as screw

# This task deliberately reuses the exact v15 socket, bulb, and contact model.
# The unused pickup cradle is absent; bulb/socket contact mechanics stay exact.
SCENE_NAME = "bulb-unscrew"
OBJECT_NAMES = screw.OBJECT_NAMES
SCENE_GEOM_NAMES = tuple(
  name for name in screw.SCENE_GEOM_NAMES if not name.startswith("bulb_pickup_cradle_")
)
OBJECT_STABILIZERS = screw.OBJECT_STABILIZERS
DEFAULT_PLAYBACK_SPEED = screw.DEFAULT_PLAYBACK_SPEED
DEFAULT_XY_JITTER = screw.DEFAULT_XY_JITTER
DEFAULT_YAW_JITTER = screw.DEFAULT_YAW_JITTER

ARM_HOME = screw.ARM_HOME
TABLETOP_HEIGHT_M = screw.TABLETOP_HEIGHT_M
BULB_INITIAL_POSITION_M = screw.BULB_INITIAL_POSITION_M.copy()
BULB_GLOBE_RADIUS_XY_M = screw.BULB_GLOBE_RADIUS_XY_M
SOCKET_MOUTH_POSITION_M = screw.SOCKET_MOUTH_POSITION_M.copy()
SOCKET_FUNNEL_TOP_Z_M = screw.SOCKET_FUNNEL_TOP_Z_M
SOCKET_APPROACH_CLEARANCE_M = screw.SOCKET_APPROACH_CLEARANCE_M
SOCKET_TRANSFER_CLEARANCE_M = screw.SOCKET_TRANSFER_CLEARANCE_M
THREAD_ENTRY_POSITION_M = screw.THREAD_ENTRY_POSITION_M.copy()
THREAD_ENTRY_DEPTH_M = screw.THREAD_ENTRY_DEPTH_M
THREAD_CAPTURE_LOAD_N = screw.THREAD_CAPTURE_LOAD_N
MAX_THREAD_ERROR_M = screw.MAX_THREAD_ERROR_M
THREAD_PITCH_M = screw.THREAD_PITCH_M
THREAD_TRAVEL_M = screw.THREAD_TRAVEL_M
TARGET_TURNS = screw.TARGET_TURNS
SEATED_DEPTH_M = screw.SEATED_DEPTH_M
CAPTURE_LATERAL_M = screw.CAPTURE_LATERAL_M
CAPTURE_DEPTH_M = screw.CAPTURE_DEPTH_M
CAPTURE_TILT_RAD = screw.CAPTURE_TILT_RAD
CAPTURE_SPEED_M_S = screw.CAPTURE_SPEED_M_S
CAPTURE_ANGULAR_SPEED_RAD_S = screw.CAPTURE_ANGULAR_SPEED_RAD_S
THREAD_RELEASE_DEPTH_OFFSET_M = screw.THREAD_RELEASE_DEPTH_OFFSET_M
SEATED_DEPTH_TOLERANCE_M = screw.SEATED_DEPTH_TOLERANCE_M
EXTERNAL_THREAD_SEGMENTS = screw.EXTERNAL_THREAD_SEGMENTS
INTERNAL_THREAD_SEGMENTS = screw.INTERNAL_THREAD_SEGMENTS
MAX_FINGERTIP_LOAD_N = screw.MAX_FINGERTIP_LOAD_N

MECHANICS_VERSION = (
  "bulb_unscrew_inherits_phase_aware_strict_completion_v15_reverse_tabletop_v2"
)
ROTATION_DRIVER = "five_finger_counterclockwise_fixed_wrist_v1"

# Reset-only installed preset. It is settled under the unchanged passive
# carriage/weld mechanics, then velocities and the episode clock are zeroed.
INSTALLED_SETTLE_S = 0.30
SETTLED_LINEAR_SPEED_M_S = 0.001
SETTLED_ANGULAR_SPEED_RAD_S = 0.05
INSTALLED_LINEAR_SPEED_M_S = 1e-9
INSTALLED_ANGULAR_SPEED_RAD_S = 1e-9
INSTALLED_MIN_CONTACT_LOAD_N = 0.01
INITIAL_STABLE_DWELL_S = 0.05

# Ordered physical milestones.  The release point follows directly from the
# shared pitch and auto-release threshold; it is not inferred from end pose.
GRASP_MIN_FORCE_N = 0.025
GRASP_STABLE_DWELL_S = 0.10
GRIP_GAP_TIMEOUT_S = 0.10
BREAKAWAY_MIN_TURNS = 30 / 360
BREAKAWAY_MIN_COUNTERCLOCKWISE_TORQUE_NM = 0.002
EXPECTED_RELEASE_TURNS = TARGET_TURNS - THREAD_RELEASE_DEPTH_OFFSET_M / THREAD_PITCH_M
RELEASE_TURN_TOLERANCE = 10 / 360
FINGER_STROKE_TURNS = 30 / 360
FULL_CLEAR_DEPTH_M = -0.003
FULL_CLEAR_SOCKET_LOAD_N = 0.01
FULL_CLEAR_DWELL_S = 0.08

# Return to the original reset-randomized pickup position on the bare table.
# Only actual bulb/table contact counts as placement support.
TRANSFER_HEIGHT_M = SOCKET_MOUTH_POSITION_M[2] + SOCKET_TRANSFER_CLEARANCE_M
# The nominal freejoint pose is 0.5 mm above the plane. Command a shallow,
# bounded 0.2 mm compliant preload so actual tabletop support is established
# before finger unloading; the body itself remains collision-limited.
PLACEMENT_COMMAND_Z_M = TABLETOP_HEIGHT_M - 0.0002
PLACEMENT_XY_TOLERANCE_M = 0.0015
PLACEMENT_Z_TOLERANCE_M = 0.004
PLACEMENT_TILT_RAD = np.deg2rad(8.0)
PLACEMENT_LINEAR_SPEED_M_S = 0.004
PLACEMENT_ANGULAR_SPEED_RAD_S = 0.20
PLACEMENT_MIN_SUPPORT_LOAD_N = 0.01
MAX_PLACEMENT_SUPPORT_LOAD_N = 12.0
PLACEMENT_DWELL_S = 0.20
RELEASE_FINGERTIP_LOAD_N = screw.RELEASE_FINGERTIP_LOAD_N
RELEASE_HAND_LOAD_N = screw.RELEASE_HAND_LOAD_N
RELEASE_DWELL_S = 0.10
RETREAT_CLEARANCE_M = 0.10
RETREAT_DWELL_S = 0.10
ARM_HOME_TOLERANCE_RAD = np.deg2rad(1.0)
# Velocity actuators settle against joint friction with a measured 3.20 degree
# residual after the physical episode, while stationary and contact-free.
HAND_HOME_TOLERANCE_RAD = np.deg2rad(4.0)
HOME_SPEED_RAD_S = np.deg2rad(2.0)

__all__ = [name for name in globals() if name.isupper()]
