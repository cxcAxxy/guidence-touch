"""Broad-side five-finger pinch fitted to the shared tactile surface centroids.

These constants move only the robot; the eraser remains a free body.
"""

import numpy as np

WRIST_POSITION = np.array([0.4211100043830603, -0.3496796339931472, 0.8239280010396304])
WRIST_ROTATION = np.array(
  [
    [-0.9989732194646478, 0.0058710229915921175, 0.044922576523027676],
    [-0.03481067516910482, 0.5351389173151638, -0.8440465366726078],
    [-0.028995235586319255, -0.844743671336924, -0.5343850728168847],
  ]
)
CLOSED_HAND = np.array(
  [
    0.10364272762208986,
    1.7124868133994127,
    0.5280940330611715,
    0.2801761586282702,
    -0.0015665554661028495,
    0.3664323203855517,
    0.960830705192481,
    0.1625422952289503,
    0.006989163935531512,
    0.3192937410638449,
    1.3486888229156877,
    -0.175,
    0.020730737696587384,
    0.31741254758798343,
    1.086095315300132,
    0.08218562251886663,
    0.09976933917856236,
    0.48493146070760507,
    0.40010947233542843,
    0.5976667094640339,
  ]
)
# Fully spread pose used from reset until the wrist has descended to the
# eraser.  Keep this task-local: other environments have different approach
# geometry, even though this matches the shared neutral hand pose.
SPREAD_HAND = np.array(
  [
    0.0,
    0.25,
    0.05,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
  ]
)
# Collision-free hand shape immediately before tactile-regulated closure.
# It is reached only after the fully spread hand has finished descending.
OPEN_HAND = np.array(
  [
    0.10456546109177035,
    1.7121897305254827,
    0.26489486120979155,
    0.41816410002705395,
    -0.002925457016741118,
    0.37345855599596245,
    0.33156340799355094,
    0.7560851648961188,
    0.009473755621689449,
    0.1723577973038165,
    1.0147368322726735,
    0.2975049957066408,
    0.02555738777891963,
    0.2771750490076343,
    0.558370611414201,
    0.6309955966440928,
    0.09281903292994688,
    0.43900923608906944,
    0.04553581573586078,
    0.7398626047723772,
  ]
)
ARM_SEED = np.array(
  [
    -1.1180220408011665,
    -1.8363483221835097,
    2.1638267259111106,
    -1.3490136019762757,
    1.113204652793543,
    -1.0471975512,
    -0.5832794981144664,
  ]
)

# Same calibrated pre-grasp pose as the original branch, but reached with the
# fifth arm joint near -177 degrees. From shared home (-150 degrees), this
# makes the forearm turn about 27 degrees counterclockwise instead of taking
# the equivalent +296-degree clockwise route to +146 degrees.
APPROACH_ARM_SEED = np.deg2rad(
  [-152.5, 2.6, 128.0, -105.3, -177.1, -14.4, -7.1]
)

# Negative-joint-five IK branch for the nominal, safely lifted transfer to the
# first board line. The executor resolves the exact target at runtime so ink
# translation and the measured in-hand tool transform remain authoritative.
BOARD_APPROACH_ARM_SEED = np.deg2rad(
  [-13.1, -35.2, -34.3, -98.9, -158.9, -35.3, 68.1]
)
