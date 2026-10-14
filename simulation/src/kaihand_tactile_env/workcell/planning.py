"""Compatibility alias for the pick-and-place task implementation."""

import sys

from kaihand_tactile_env.tasks.pick_place import task as _implementation

sys.modules[__name__] = _implementation
