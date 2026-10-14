"""Small, model-free client helpers for the EgoSteer WebSocket protocol."""

from __future__ import annotations

import asyncio
import io
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import msgpack
import numpy as np
from PIL import Image
from websockets.asyncio.client import ClientConnection, connect


class PolicyProtocolError(RuntimeError):
  """The remote policy returned data that violates the serving contract."""


def _pack_array(value: Any) -> Any:
  if isinstance(value, np.ndarray):
    if value.dtype.kind in ("V", "O", "c"):
      raise ValueError(f"unsupported NumPy dtype: {value.dtype}")
    return {
      b"__ndarray__": True,
      b"data": value.tobytes(),
      b"dtype": value.dtype.str,
      b"shape": value.shape,
    }
  if isinstance(value, np.generic):
    if value.dtype.kind in ("V", "O", "c"):
      raise ValueError(f"unsupported NumPy dtype: {value.dtype}")
    return {
      b"__npgeneric__": True,
      b"data": value.item(),
      b"dtype": value.dtype.str,
    }
  return value


def _unpack_array(value: dict[Any, Any]) -> Any:
  if b"__ndarray__" in value:
    # copy() detaches the result from msgpack's immutable bytes buffer.  The
    # server's own implementation returns a read-only view, which triggers a
    # PyTorch warning when reused as input on a later request.
    return np.ndarray(
      buffer=value[b"data"],
      dtype=np.dtype(value[b"dtype"]),
      shape=value[b"shape"],
    ).copy()
  if b"__npgeneric__" in value:
    return np.dtype(value[b"dtype"]).type(value[b"data"])
  return value


def pack_payload(value: Any) -> bytes:
  """Pack values exactly like EgoSteer's ``msgpack_numpy`` module."""
  return msgpack.packb(value, default=_pack_array)


def unpack_payload(value: bytes) -> Any:
  """Unpack values exactly like EgoSteer's ``msgpack_numpy`` module."""
  return msgpack.unpackb(value, object_hook=_unpack_array)


def encode_jpeg_sequence(images: Any, *, quality: int = 90) -> dict[str, Any]:
  """Encode ``uint8 [T,H,W,3]`` RGB using EgoSteer's JPEG wire schema."""
  array = np.asarray(images)
  if array.ndim != 4 or array.shape[-1] != 3 or array.dtype != np.uint8:
    raise ValueError(
      "images must have uint8 shape (T, H, W, 3), "
      f"got shape={array.shape}, dtype={array.dtype}"
    )
  if not 1 <= quality <= 100:
    raise ValueError("JPEG quality must be in [1, 100]")

  frames: list[bytes] = []
  for frame in array:
    buffer = io.BytesIO()
    Image.fromarray(frame).save(
      buffer,
      format="JPEG",
      quality=quality,
      # The training dataset was exported without chroma subsampling.
      subsampling=0,
    )
    frames.append(buffer.getvalue())
  return {
    "__image_encoding__": "jpeg_sequence",
    "format": "jpeg",
    "quality": int(quality),
    "shape": tuple(array.shape),
    "dtype": "uint8",
    "color_order": "rgb",
    "frames": frames,
    "raw_nbytes": int(array.nbytes),
    "encoded_nbytes": sum(len(frame) for frame in frames),
  }


@dataclass(frozen=True)
class HistoryBatch:
  """Causal image/raw-state history selected at the model's stride."""

  images: np.ndarray
  raw_states: np.ndarray


class ObservationHistory:
  """Thirty-Hz base buffer with EgoSteer's repeat-left-padding semantics."""

  def __init__(self, *, horizon: int = 6, stride: int = 30) -> None:
    if horizon <= 0 or stride <= 0:
      raise ValueError("history horizon and stride must be positive")
    self.horizon = int(horizon)
    self.stride = int(stride)
    self.capacity = (self.horizon - 1) * self.stride + 1
    self._images: deque[np.ndarray] = deque(maxlen=self.capacity)
    self._raw_states: deque[np.ndarray] = deque(maxlen=self.capacity)

  def __len__(self) -> int:
    return len(self._images)

  def append(self, image: Any, raw_state: Any) -> None:
    image_array = np.asarray(image)
    state_array = np.asarray(raw_state)
    if (
      image_array.ndim != 3
      or image_array.shape[-1] != 3
      or image_array.dtype != np.uint8
    ):
      raise ValueError("history image must have uint8 shape (H, W, 3)")
    if state_array.shape != (48,) or not np.all(np.isfinite(state_array)):
      raise ValueError("history raw_state must contain 48 finite values")
    self._images.append(image_array.copy())
    self._raw_states.append(state_array.astype(np.float32, copy=True))

  def select(self) -> HistoryBatch:
    if not self._images:
      raise RuntimeError("observation history is empty")
    count = len(self._images)
    indices = [
      max(0, count - 1 - offset)
      for offset in range((self.horizon - 1) * self.stride, -1, -self.stride)
    ]
    images = tuple(self._images)
    raw_states = tuple(self._raw_states)
    return HistoryBatch(
      images=np.stack([images[index] for index in indices]),
      raw_states=np.stack([raw_states[index] for index in indices]),
    )


@dataclass(frozen=True)
class PolicyResponse:
  pred_actions: np.ndarray
  server_timing: dict[str, float]


class EgoSteerPolicyClient:
  """One-request-at-a-time client for EgoSteer's policy server."""

  def __init__(
    self,
    uri: str,
    *,
    open_timeout: float = 30.0,
    response_timeout: float = 600.0,
    allow_nonfinite_actions: bool = False,
  ) -> None:
    if not uri.startswith(("ws://", "wss://")):
      raise ValueError("policy URI must start with ws:// or wss://")
    if open_timeout <= 0.0 or response_timeout <= 0.0:
      raise ValueError("WebSocket timeouts must be positive")
    self.uri = uri
    self.open_timeout = float(open_timeout)
    self.response_timeout = float(response_timeout)
    # Observation runners validate and reject these outputs transactionally,
    # then advance a hold tick.  Shape/protocol checks always remain enabled.
    self.allow_nonfinite_actions = bool(allow_nonfinite_actions)
    self.websocket: ClientConnection | None = None
    self.metadata: dict[str, Any] | None = None

  async def __aenter__(self) -> EgoSteerPolicyClient:
    self.websocket = await connect(
      self.uri,
      max_size=None,
      compression=None,
      open_timeout=self.open_timeout,
      # Inference and slow SSH uploads block the server event loop long enough
      # to trip websockets' default 20-second keepalive on first-time graphs.
      ping_interval=None,
    )
    raw_metadata = await asyncio.wait_for(
      self.websocket.recv(), timeout=self.response_timeout
    )
    if isinstance(raw_metadata, str):
      raise PolicyProtocolError(raw_metadata)
    metadata = unpack_payload(raw_metadata)
    if not isinstance(metadata, dict):
      raise PolicyProtocolError("server metadata must be a mapping")
    action_horizon = metadata.get("action_horizon")
    action_dim = metadata.get("action_dim")
    if action_dim != 48 or not isinstance(action_horizon, int) or action_horizon <= 0:
      raise PolicyProtocolError(
        f"expected action_dim=48 and positive action_horizon, got {metadata!r}"
      )
    self.metadata = metadata
    return self

  async def __aexit__(self, *_: object) -> None:
    if self.websocket is not None:
      await self.websocket.close()
    self.websocket = None

  async def infer(
    self,
    *,
    images: Any,
    states: Any,
    camera_intrinsics: Any,
    instruction: str,
    image_format: str = "jpeg",
    jpeg_quality: int = 90,
  ) -> PolicyResponse:
    if self.websocket is None or self.metadata is None:
      raise RuntimeError("policy client is not connected")
    state_array = np.asarray(states, dtype=np.float32)
    declared_cameras = self.metadata.get("cameras")
    if declared_cameras is None:
      declared_cameras = self.metadata.get("observation_contract", {}).get(
        "cameras", ["head"]
      )
    expected_cameras = tuple(str(name) for name in declared_cameras)
    if not expected_cameras:
      raise PolicyProtocolError("server metadata declares no observation cameras")

    multi_camera = isinstance(images, Mapping)
    if multi_camera:
      if not isinstance(camera_intrinsics, Mapping):
        raise ValueError("multi-camera images require camera_intrinsics mapping")
      image_arrays = {str(name): np.asarray(value) for name, value in images.items()}
      intrinsic_arrays = {
        str(name): np.asarray(value, dtype=np.float64)
        for name, value in camera_intrinsics.items()
      }
      if set(image_arrays) != set(expected_cameras):
        raise ValueError(
          "image cameras must exactly match server metadata: "
          f"expected={list(expected_cameras)}, got={sorted(image_arrays)}"
        )
      if set(intrinsic_arrays) != set(expected_cameras):
        raise ValueError(
          "intrinsic cameras must exactly match server metadata: "
          f"expected={list(expected_cameras)}, got={sorted(intrinsic_arrays)}"
        )
    else:
      if expected_cameras != ("head",):
        raise ValueError(
          f"server requires multi-camera images: expected={list(expected_cameras)}"
        )
      image_arrays = {"head": np.asarray(images)}
      intrinsic_arrays = {"head": np.asarray(camera_intrinsics, dtype=np.float64)}

    frame_counts = set()
    for camera in expected_cameras:
      image_array = image_arrays[camera]
      intrinsic_array = intrinsic_arrays[camera]
      if image_array.ndim != 4 or image_array.shape[-1] != 3:
        raise ValueError(
          f"{camera} images must have shape (T, H, W, 3), got {image_array.shape}"
        )
      if image_array.dtype != np.uint8:
        raise ValueError(f"{camera} images must use uint8, got {image_array.dtype}")
      if intrinsic_array.shape not in ((4,), (3, 3)):
        raise ValueError(f"{camera} camera_intrinsics must have shape (4,) or (3, 3)")
      frame_counts.add(int(image_array.shape[0]))
    if len(frame_counts) != 1:
      raise ValueError("all camera histories must contain the same frame count")
    frame_count = frame_counts.pop()
    if state_array.shape != (frame_count, 48):
      raise ValueError("states must have shape (T, 48) matching camera histories")
    if not instruction.strip():
      raise ValueError("instruction must be non-empty")
    if image_format not in ("jpeg", "raw"):
      raise ValueError("image_format must be 'jpeg' or 'raw'")

    image_value: Any = image_arrays if multi_camera else image_arrays["head"]
    intrinsic_value: Any = (
      intrinsic_arrays if multi_camera else intrinsic_arrays["head"]
    )
    observation: dict[str, Any] = {
      "instruction": instruction,
      "states": state_array,
      "action_rtc": None,
      "camera_intrinsics": intrinsic_value,
    }
    if image_format == "jpeg":
      image_value = (
        {
          camera: encode_jpeg_sequence(image_arrays[camera], quality=jpeg_quality)
          for camera in expected_cameras
        }
        if multi_camera
        else encode_jpeg_sequence(image_arrays["head"], quality=jpeg_quality)
      )
      observation["image_compression"] = {
        "format": "jpeg",
        "quality": int(jpeg_quality),
        "color_order": "rgb",
        "field": "image",
      }
    observation["image"] = image_value

    await self.websocket.send(pack_payload(observation))
    raw_response = await asyncio.wait_for(
      self.websocket.recv(), timeout=self.response_timeout
    )
    if isinstance(raw_response, str):
      raise PolicyProtocolError(raw_response)
    response = unpack_payload(raw_response)
    if not isinstance(response, dict) or "pred_actions" not in response:
      raise PolicyProtocolError("policy response has no pred_actions")
    actions = np.asarray(response["pred_actions"], dtype=np.float32).copy()
    expected_shape = (int(self.metadata["action_horizon"]), 48)
    if actions.shape != expected_shape or (
      not self.allow_nonfinite_actions and not np.all(np.isfinite(actions))
    ):
      raise PolicyProtocolError(
        f"pred_actions must contain finite values with shape {expected_shape}, "
        f"got {actions.shape}"
      )
    timing_raw = response.get("server_timing", {})
    timing = {
      str(key): float(value)
      for key, value in timing_raw.items()
      if isinstance(value, (int, float))
    }
    return PolicyResponse(pred_actions=actions, server_timing=timing)


__all__ = [
  "EgoSteerPolicyClient",
  "HistoryBatch",
  "ObservationHistory",
  "PolicyProtocolError",
  "PolicyResponse",
  "encode_jpeg_sequence",
  "pack_payload",
  "unpack_payload",
]
