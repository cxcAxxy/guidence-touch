"""Compatibility alias for the shared rendering implementation."""

import sys

from kaihand_tactile_env.shared import rendering as _implementation

sys.modules[__name__] = _implementation
