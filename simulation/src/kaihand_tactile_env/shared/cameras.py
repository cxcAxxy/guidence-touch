"""Camera names shared by tasks and lightweight collection CLIs.

No numerical/physics imports: USB configures its worker thread limits before
loading NumPy or MuJoCo. Geometry and extrinsics remain in shared/mjcf/robot.xml.
"""

ROBOT_CAMERA_NAMES = ("head", "right_wrist", "left_wrist")
# Stable model-facing order. Keep it separate from the historical MJCF order
# above so collection, conversion and deployment use one explicit contract.
TRAINING_CAMERA_NAMES = ("head", "left_wrist", "right_wrist")
SHARED_CAMERA_NAMES = ("overhead", "front", *ROBOT_CAMERA_NAMES)
LEGACY_CAPTURE_CAMERA_NAMES = ("overhead", "front", "head")
