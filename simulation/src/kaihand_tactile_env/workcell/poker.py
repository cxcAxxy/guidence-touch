"""Compatibility alias for the poker-draw task implementation."""

import sys

from kaihand_tactile_env.tasks.poker_draw import task as _implementation

sys.modules[__name__] = _implementation
