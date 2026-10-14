"""Compatibility alias for the shared tactile implementation."""

import sys

from kaihand_tactile_env.shared import tactile as _implementation

sys.modules[__name__] = _implementation
