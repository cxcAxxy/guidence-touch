"""Compatibility alias for the shared egosteer_client implementation."""

import sys

from kaihand_tactile_env.shared import egosteer_client as _implementation

sys.modules[__name__] = _implementation
