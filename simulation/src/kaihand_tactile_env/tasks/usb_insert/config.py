"""Task-local geometry and defaults for the simplified USB-A insertion scene.

Lengths are metres. Both connector-local frames use +X for insertion,
+Y for width, and +Z for the keyed side. The socket is rotated so local
+X points along world -Z: its opening faces upwards. This is a mechanical benchmark,
not an electrical connector model or a USB dimensional compliance model.
"""

from __future__ import annotations

import numpy as np

from ...shared.posture import ARM_HOME as ARM_HOME

SCENE_NAME = "usb-insert"
OBJECT_NAMES = ("usb_plug",)
SCENE_GEOM_NAMES = (
  "usb_fixture_base",
  "usb_fixture_post",
  "usb_fixture_rim_x_pos",
  "usb_fixture_rim_x_neg",
  "usb_fixture_rim_y_pos",
  "usb_fixture_rim_y_neg",
  "usb_socket_wall_left",
  "usb_socket_wall_right",
  "usb_socket_wall_top",
  "usb_socket_wall_bottom",
  "usb_socket_backstop",
  "usb_socket_tongue",
  "usb_socket_contact_1_visual",
  "usb_socket_contact_2_visual",
  "usb_socket_contact_3_visual",
  "usb_socket_contact_4_visual",
  "usb_socket_spring_top_pad_left",
  "usb_socket_spring_top_pad_right",
  "usb_socket_spring_bottom_pad_left",
  "usb_socket_spring_bottom_pad_right",
  "usb_plug_handle",
  "usb_plug_neck",
  "usb_plug_orientation_mark_visual",
  "usb_plug_shell_left",
  "usb_plug_shell_right",
  "usb_plug_shell_top",
  "usb_plug_shell_bottom",
  "usb_plug_insulator",
  "usb_plug_contact_1_visual",
  "usb_plug_contact_2_visual",
  "usb_plug_contact_3_visual",
  "usb_plug_contact_4_visual",
)
OBJECT_STABILIZERS: dict[tuple[str, str], str] = {}
DEFAULT_PLAYBACK_SPEED = 1.0
DEFAULT_XY_JITTER = 0.0
DEFAULT_YAW_JITTER = 0.0

TABLETOP_HEIGHT_M = 0.680
PLUG_INITIAL_POSITION_M = np.array([0.500, -0.180, 0.6865])
# Explicit automatic-episode preset; the idle scene keeps its original pose.
# Mark down, connector pointing away from the robot and toward its right.
# The reversed grip reaches the socket palm down, with thumb/index above pinky.
AUTO_PLUG_QUATERNION_WXYZ = np.array([0.0, -0.9659258262890683, 0.258819045102521, 0.0])
PLUG_GRASP_LOCAL_M = np.array([0.0, 0.0, 0.0])
PLUG_TIP_LOCAL_M = np.array([0.027, 0.0, 0.0])
PLUG_HANDLE_HALF_SIZE_M = np.array([0.0145, 0.009, 0.006])
PLUG_HANDLE_CENTER_LOCAL_M = np.array([-0.0005, 0.0, 0.0])
PLUG_SHELL_HALF_WIDTH_M = 0.006
PLUG_SHELL_HALF_HEIGHT_M = 0.00225
PLUG_SHELL_LENGTH_M = 0.012
PLUG_SHELL_WALL_THICKNESS_M = 0.00035
PLUG_MASS_KG = 0.018

SOCKET_MOUTH_POSITION_M = np.array([0.620, -0.180, 0.720])
SOCKET_QUATERNION_WXYZ = np.array([np.sqrt(0.5), 0.0, np.sqrt(0.5), 0.0])
SOCKET_HALF_WIDTH_M = 0.0063
SOCKET_HALF_HEIGHT_M = 0.00255
CLEARANCE_PER_SIDE_M = 0.0003
TARGET_INSERTION_DEPTH_M = 0.0119
MAX_INSERTION_DEPTH_M = 0.012
CONTACT_MODEL_VERSION = "passive_spring_shoes_bottom_out_v3"
BACKSTOP_DEPTH_M = 0.012
SEATED_DEPTH_TOLERANCE_M = 0.0001
MAX_SOCKET_PENETRATION_M = 0.0001
SEATED_BACKSTOP_MIN_LOAD_N = 0.0001
SEATED_LINEAR_SPEED_M_S = 0.0005
SEATED_DWELL_S = 0.1
# Passive spring preload is included in total normal load; axial/side loads
# have separate checks so intended symmetric preload is not a jam detector.
MAX_SOCKET_NORMAL_LOAD_N = 4.0
MAX_SOCKET_AXIAL_FORCE_N = 2.0
MAX_SOCKET_WALL_LOAD_N = 1.5
INSERTION_FORCE_LIMIT_N = 1.5
BOTTOM_OUT_TARGET_FORCE_N = 1.0
BOTTOM_OUT_MIN_FORCE_N = 0.6
BOTTOM_OUT_HOLD_S = 0.15
BOTTOM_OUT_MAX_COMMAND_DEPTH_M = 0.015
SOCKET_TONGUE_FRONT_DEPTH_M = 0.003
SOCKET_TONGUE_HALF_WIDTH_M = 0.0054
SOCKET_TONGUE_Z_RANGE_M = (0.0001, 0.0014)

__all__ = [name for name in globals() if name.isupper()]
