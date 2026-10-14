"""Compatibility alias for the shared config implementation."""

import sys

from kaihand_tactile_env.shared import config as _implementation

sys.modules[__name__] = _implementation
