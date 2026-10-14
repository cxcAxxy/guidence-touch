"""Compatibility alias for the shared inference_dashboard implementation."""

import sys

from kaihand_tactile_env.shared import inference_dashboard as _implementation

sys.modules[__name__] = _implementation
