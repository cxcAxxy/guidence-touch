"""KaiHand tactile manipulation and data collection environment."""

__all__ = [
  "ArmHandSimulation",
  "CameraConfig",
  "GenesisProbeTactileProvider",
  "WorkcellConfig",
]


def __getattr__(name):
  """Keep the public API without loading physics for lightweight CLI config.

  In particular, importing shared.cameras must not initialize numerical thread
  pools before a collection entry point has applied its worker limits.
  """
  from importlib import import_module

  modules = {
    "CameraConfig": ".workcell",
    "WorkcellConfig": ".workcell",
    "ArmHandSimulation": ".workcell.simulation",
    "GenesisProbeTactileProvider": ".workcell.tactile",
  }
  if name not in modules:
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
  value = getattr(import_module(modules[name], __name__), name)
  globals()[name] = value
  return value


def __dir__():
  return sorted(set(globals()) | set(__all__))
