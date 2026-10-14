"""Compatibility alias for the shared egosteer_adapter implementation."""

import sys

from kaihand_tactile_env.shared import egosteer_adapter as _implementation

sys.modules[__name__] = _implementation
