"""Geometry, safety limits, and terminal criteria for USB unplugging."""

from __future__ import annotations

import numpy as np

from ...shared.posture import ARM_HOME as ARM_HOME
from ..usb_insert import config as insertion

SCENE_NAME = "usb-unplug"
OBJECT_NAMES = insertion.OBJECT_NAMES
SCENE_GEOM_NAMES = insertion.SCENE_GEOM_NAMES
OBJECT_STABILIZERS: dict[tuple[str, str], str] = {}
DEFAULT_PLAYBACK_SPEED = 1.0
DEFAULT_XY_JITTER = 0.0
DEFAULT_YAW_JITTER = 0.0

# Reuse the exact connector model.  The independent scene includes the frozen
# insertion MJCF and changes only reset state and task semantics in Python.
CONTACT_MODEL_VERSION = insertion.CONTACT_MODEL_VERSION
MECHANICS_VERSION = "usb_unplug_passive_spring_v1"
PLUG_TIP_LOCAL_M = insertion.PLUG_TIP_LOCAL_M.copy()
PLUG_MASS_KG = insertion.PLUG_MASS_KG
SOCKET_MOUTH_POSITION_M = insertion.SOCKET_MOUTH_POSITION_M.copy()
SOCKET_QUATERNION_WXYZ = insertion.SOCKET_QUATERNION_WXYZ.copy()
SOCKET_HALF_WIDTH_M = insertion.SOCKET_HALF_WIDTH_M
SOCKET_HALF_HEIGHT_M = insertion.SOCKET_HALF_HEIGHT_M
PLUG_SHELL_HALF_WIDTH_M = insertion.PLUG_SHELL_HALF_WIDTH_M
PLUG_SHELL_HALF_HEIGHT_M = insertion.PLUG_SHELL_HALF_HEIGHT_M
PLUG_SHELL_LENGTH_M = insertion.PLUG_SHELL_LENGTH_M

# A reset-only initializer loads this configuration before the first step.  It
# represents the same 0.30 mm radial shoe deflection produced by insertion;
# springref, stiffness, damping, friction, and collision geometry stay intact.
INITIAL_INSERTION_DEPTH_M = insertion.BACKSTOP_DEPTH_M
INITIAL_SPRING_DISPLACEMENT_M = 0.00030
INITIAL_DEPTH_TOLERANCE_M = 0.00012
INITIAL_LATERAL_TOLERANCE_M = 0.00012
INITIAL_ORIENTATION_TOLERANCE_RAD = np.deg2rad(1.0)
INITIAL_SETTLE_S = 0.30
INITIAL_STABLE_DWELL_S = 0.10

MAX_SOCKET_PENETRATION_M = insertion.MAX_SOCKET_PENETRATION_M
MAX_SOCKET_NORMAL_LOAD_N = insertion.MAX_SOCKET_NORMAL_LOAD_N
MAX_SOCKET_WALL_LOAD_N = insertion.MAX_SOCKET_WALL_LOAD_N
MAX_WITHDRAWAL_RESISTANCE_N = 1.5
MAX_INSERTION_RESISTANCE_N = insertion.MAX_SOCKET_AXIAL_FORCE_N

GRASP_MIN_FORCE_N = 0.05
GRASP_STABLE_DWELL_S = 0.15
GRIP_GAP_TIMEOUT_S = 0.05
FULL_CLEAR_DEPTH_M = -0.002
FULL_CLEAR_DWELL_S = 0.10

# Return to the calibrated mark-down pickup pose used by usb-insert.  Height is
# evaluated after physical settling, not forced to this reset seed.
HOME_POSITION_M = insertion.PLUG_INITIAL_POSITION_M.copy()
HOME_QUATERNION_WXYZ = insertion.AUTO_PLUG_QUATERNION_WXYZ.copy()
HOME_XY_TOLERANCE_M = 0.008
HOME_HEIGHT_RANGE_M = (0.6845, 0.6890)
HOME_ORIENTATION_TOLERANCE_RAD = np.deg2rad(10.0)
HOME_LINEAR_SPEED_M_S = 0.002
HOME_ANGULAR_SPEED_RAD_S = 0.10
HOME_STABLE_DWELL_S = 0.30
RELEASE_DWELL_S = 0.10
RETREAT_CLEARANCE_M = 0.10
RETREAT_DWELL_S = 0.10

__all__ = [name for name in globals() if name.isupper()]
