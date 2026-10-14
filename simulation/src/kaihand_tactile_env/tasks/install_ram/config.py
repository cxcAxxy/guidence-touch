"""Real-size DDR4 UDIMM insertion benchmark; all lengths are metres.

The module and socket use X along the long edge, Y through the PCB, and
Z up. Insertion is along -Z. Tolerances/contact mechanics are simulation
approximations documented in docs/tasks/install_ram_dimensions.md.
"""

from __future__ import annotations

import numpy as np

from ...shared.posture import ARM_HOME as ARM_HOME

SCENE_NAME = "install-ram"
OBJECT_NAMES = ("ram",)
OBJECT_STABILIZERS: dict[tuple[str, str], str] = {}
DEFAULT_PLAYBACK_SPEED = 1.0
DEFAULT_XY_JITTER = 0.0
DEFAULT_YAW_JITTER = 0.0
TABLETOP_HEIGHT_M = 0.680
RAM_LENGTH_M = 0.13335
RAM_HEIGHT_M = 0.03125
RAM_THICKNESS_M = 0.00140
RAM_CHIP_ENVELOPE_THICKNESS_M = 0.00340
RAM_MASS_KG = 0.020
RAM_INITIAL_POSITION_M = np.array([0.470, -0.290, 0.755635])
RAM_INITIAL_QUATERNION_WXYZ = np.array([1.0, 0.0, 0.0, 0.0])
RAM_GRASP_LOCAL_M = np.array([0.0, 0.0, 0.006])
RAM_BOTTOM_LOCAL_M = np.array([0.0, 0.0, -0.015625])
RAM_KEY_X_M = 0.005575
RAM_KEY_NOTCH_WIDTH_M = 0.0015
RAM_KEY_NOTCH_HEIGHT_M = 0.0038
SOCKET_MOUTH_POSITION_M = np.array([0.500, -0.150, 0.704])
SOCKET_QUATERNION_WXYZ = np.array([1.0, 0.0, 0.0, 0.0])
SOCKET_INNER_LENGTH_M = 0.13395
SOCKET_INNER_WIDTH_M = 0.00200
SOCKET_CLEARANCE_PER_SIDE_M = 0.0003
SOCKET_KEY_WIDTH_M = 0.001
SOCKET_KEY_HEIGHT_M = 0.0035
TARGET_INSERTION_DEPTH_M = 0.006
MAX_INSERTION_DEPTH_M = 0.006
SEATED_DEPTH_TOLERANCE_M = 0.0002
MAX_SOCKET_PENETRATION_M = 0.0002
SEATED_BACKSTOP_MIN_LOAD_N = 0.01
SEATED_LINEAR_SPEED_M_S = 0.001
SEATED_DWELL_S = 0.3
INSERTION_FORCE_LIMIT_N = 5.0
BOTTOM_OUT_TARGET_FORCE_N = 1.5
BOTTOM_OUT_MIN_FORCE_N = 1.2
BOTTOM_OUT_HOLD_S = 0.4
SPRING_STIFFNESS_N_M = 3000.0
SPRING_FRICTION = 0.35
MECHANICS_VERSION = "aligned_axial_insertion_frozen_grip_v3"
CONTACT_MODEL_VERSION = MECHANICS_VERSION
DIMENSIONS_M = {
  "ram_length": RAM_LENGTH_M,
  "ram_height": RAM_HEIGHT_M,
  "ram_pcb_thickness": RAM_THICKNESS_M,
  "ram_chip_envelope_thickness": RAM_CHIP_ENVELOPE_THICKNESS_M,
  "socket_inner_length": SOCKET_INNER_LENGTH_M,
  "socket_inner_width": SOCKET_INNER_WIDTH_M,
  "insertion_depth": TARGET_INSERTION_DEPTH_M,
  "key_notch_x": RAM_KEY_X_M,
  "key_notch_width": RAM_KEY_NOTCH_WIDTH_M,
  "key_notch_height": RAM_KEY_NOTCH_HEIGHT_M,
}
DIMENSION_SOURCES = (
  "https://www.te.com/content/dam/te-com/documents/consumer-devices/global/memory-sockets-final-flyer.pdf",
  "https://www.kingston.com/datasheets/KVR32N22S8_8.pdf",
  "https://www.te.com/en/product-1-2308107-1.html",
  "https://www.te.com/content/dam/te-com/documents/consumer-devices/global/ddr4-flyer-en.pdf",
)

SCENE_GEOM_NAMES = (
  "ram_motherboard_pcb",
  "ram_motherboard_standoff_0",
  "ram_motherboard_standoff_1",
  "ram_motherboard_standoff_2",
  "ram_motherboard_standoff_3",
  "ram_cpu_package",
  "ram_cpu_lid_visual",
  "ram_motherboard_capacitor_0",
  "ram_motherboard_capacitor_1",
  "ram_motherboard_capacitor_2",
  "ram_motherboard_capacitor_3",
  "ram_motherboard_capacitor_4",
  "ram_motherboard_capacitor_5",
  "ram_socket_base",
  "ram_socket_wall_front",
  "ram_socket_wall_back",
  "ram_socket_end_left",
  "ram_socket_end_right",
  "ram_socket_backstop",
  "ram_socket_key",
  "ram_socket_spring_front_left",
  "ram_socket_spring_front_right",
  "ram_socket_spring_back_left",
  "ram_socket_spring_back_right",
  "ram_latch_left_pivot",
  "ram_latch_left_arm",
  "ram_latch_left_tab",
  "ram_latch_right_pivot",
  "ram_latch_right_arm",
  "ram_latch_right_tab",
  "ram_stand_base",
  "ram_stand_left_post",
  "ram_stand_left_front",
  "ram_stand_left_back",
  "ram_stand_right_post",
  "ram_stand_right_front",
  "ram_stand_right_back",
  "ram_pcb_upper",
  "ram_pcb_lower_left",
  "ram_pcb_lower_right",
  "ram_label_visual",
) + (
  tuple(f"ram_board_trace_{i}_visual" for i in range(12))
  + tuple(
    f"ram_socket_contact_{side}_{i:03d}_visual"
    for side in ("front", "back")
    for i in range(144)
  )
  + tuple(f"ram_chip_{side}_{i}" for side in ("front", "back") for i in range(8))
  + tuple(
    f"ram_chip_mark_{side}_{i}_visual" for side in ("front", "back") for i in range(8)
  )
  + tuple(
    f"ram_contact_{side}_{i:03d}_visual"
    for side in ("front", "back")
    for i in range(144)
  )
  + tuple(
    f"ram_resistor_{side}_{i}_visual" for side in ("front", "back") for i in range(36)
  )
  + tuple(
    f"ram_trace_{side}_{i}_visual" for side in ("front", "back") for i in range(20)
  )
  + tuple(f"ram_label_barcode_{i}_visual" for i in range(16))
)

__all__ = [name for name in globals() if name.isupper()]
