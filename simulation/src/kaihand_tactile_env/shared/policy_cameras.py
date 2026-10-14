"""Validated camera contracts shared by policy evaluation adapters."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

CANONICAL_POLICY_CAMERAS = ("head", "left_wrist", "right_wrist")


def policy_camera_names(
  observation_contract: Mapping[str, Any],
  *,
  require_head: bool = True,
) -> tuple[str, ...]:
  """Return one canonical, duplicate-free camera tuple from a frozen contract."""
  declared = observation_contract.get("cameras", ("head",))
  if not isinstance(declared, (list, tuple)) or not declared:
    raise RuntimeError("observation_contract.cameras must be a nonempty list")
  names = tuple(str(name) for name in declared)
  if len(names) != len(set(names)):
    raise RuntimeError(f"policy camera names must be unique, got {list(names)}")
  unknown = sorted(set(names) - set(CANONICAL_POLICY_CAMERAS))
  if unknown:
    raise RuntimeError(f"unsupported policy cameras: {unknown}")
  if require_head and "head" not in names:
    raise RuntimeError(
      "policy evaluation supports cameras containing head; the current action "
      "reference contract requires the head camera"
    )
  canonical = tuple(name for name in CANONICAL_POLICY_CAMERAS if name in names)
  if names != canonical:
    raise RuntimeError(
      "policy cameras must use canonical order: head left_wrist right_wrist"
    )
  return names


def include_model_right_wrist_panel(
  camera_names: tuple[str, ...] | list[str], second_camera: str
) -> bool:
  """Show the model's right-wrist view unless the second panel already does."""
  return "right_wrist" in camera_names and second_camera != "right_wrist"


def image_shape_hwc(observation_contract: Mapping[str, Any]) -> tuple[int, int, int]:
  """Read and validate the shared RGB image shape."""
  shape = observation_contract.get("image_shape_hwc", (240, 320, 3))
  try:
    height, width, channels = (int(value) for value in shape)
  except (TypeError, ValueError) as error:
    raise RuntimeError("image_shape_hwc must contain three integers") from error
  if height <= 0 or width <= 0 or channels != 3:
    raise RuntimeError(
      f"policy RGB image shape must be positive HxWx3, got {list(shape)}"
    )
  return height, width, channels


def pi05_image_payload(
  images: Mapping[str, Any], observation_contract: Mapping[str, Any]
) -> dict[str, Any]:
  """Map synchronized camera images to the exact frozen pi0.5 request keys."""
  names = policy_camera_names(observation_contract)
  if tuple(images) != names:
    raise RuntimeError(
      f"captured cameras {list(images)} differ from contract {list(names)}"
    )
  declared_keys = observation_contract.get("request_image_keys")
  if declared_keys is None:
    keys = (
      {"head": "image"}
      if names == ("head",)
      else {name: f"{name}_image" for name in names}
    )
  else:
    if not isinstance(declared_keys, Mapping):
      raise RuntimeError("request_image_keys must be an object mapping camera to key")
    keys = {str(name): str(key) for name, key in declared_keys.items()}
    if tuple(keys) != names or len(set(keys.values())) != len(keys):
      raise RuntimeError(
        "request_image_keys must cover contract cameras once in canonical order"
      )
  return {keys[name]: images[name] for name in names}
