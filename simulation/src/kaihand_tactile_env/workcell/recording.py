"""Compatibility alias for the shared recording implementation."""

import sys

from kaihand_tactile_env.shared import recording as _implementation

sys.modules[__name__] = _implementation
