"""Compatibility alias for the shared simulation implementation."""

import sys

from kaihand_tactile_env.shared import simulation as _implementation

sys.modules[__name__] = _implementation
