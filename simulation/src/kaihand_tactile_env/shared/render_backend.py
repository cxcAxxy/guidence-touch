"""Inspect the actual GL context, not just the requested environment variables."""

from __future__ import annotations

import os

_SOFTWARE_EGL_DEVICE_ID = None


def _software_device_index(device_extensions):
  """Match an advertised software device, never guess from the ordinal."""
  for index, extensions in enumerate(device_extensions):
    if "EGL_MESA_device_software" in extensions.split():
      return index
  raise RuntimeError(
    "Software rendering requested, but EGL exposes no software device; "
    "refusing to initialize a hardware GPU"
  )


def _egl_device_extensions(egl_api):
  import ctypes

  address = egl_api.eglGetProcAddress('eglQueryDeviceStringEXT')
  if not address:
    raise RuntimeError("Cannot identify EGL software devices: eglQueryDeviceStringEXT unavailable")
  query = ctypes.CFUNCTYPE(ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int)(address)
  return [
    (query(device, 0x3055) or b'').decode('ascii', errors='replace')
    for device in egl_api.eglQueryDevicesEXT()
  ]


def prepare_render_backend():
  """Select CPU EGL before MuJoCo initializes its process-global display.

  LIBGL_ALWAYS_SOFTWARE alone cannot override EGL_PLATFORM_DEVICE_EXT when
  MuJoCo chooses a physical GPU. Enumerating extensions creates no GL context.
  Explicit software requests override inherited GPU device indices; other modes
  retain their existing behavior. The actual renderer is still checked below.
  """
  global _SOFTWARE_EGL_DEVICE_ID
  if (os.environ.get("KAIHAND_RENDER_BACKEND", "auto") != "software"
      or os.environ.get("MUJOCO_GL", "").lower() != "egl"):
    return
  from mujoco import egl

  if egl.EGL_DISPLAY is not None:
    if (_SOFTWARE_EGL_DEVICE_ID is not None
        and os.environ.get("MUJOCO_EGL_DEVICE_ID") == str(_SOFTWARE_EGL_DEVICE_ID)):
      return
    raise RuntimeError(
      "Software EGL selection must happen before the first GL context; "
      "start a fresh process instead of reusing an initialized GPU display"
    )
  extensions = _egl_device_extensions(egl.EGL)
  index = _software_device_index(extensions)
  os.environ['MUJOCO_EGL_DEVICE_ID'] = str(index)
  _SOFTWARE_EGL_DEVICE_ID = index
  print(f'Software EGL device selected: index={index} (EGL_MESA_device_software)', flush=True)


def describe_backend(renderer: str, vendor: str, version: str, requested: str) -> dict:
  if requested not in ("auto", "hardware", "software"):
    raise ValueError(f"unknown KAIHAND_RENDER_BACKEND: {requested!r}")
  software = any(token in renderer.lower() for token in (
    "llvmpipe", "softpipe", "swrast", "software", "swr rasterizer"))
  if requested != "auto" and not renderer:
    raise RuntimeError("Cannot identify the active OpenGL renderer")
  if requested == "hardware" and software:
    raise RuntimeError(
      f"Hardware rendering required, but actual renderer is {renderer!r}. "
      "Check GPU device access (/dev/dri), EGL and Mesa; software fallback is disabled."
    )
  if requested == "software" and not software:
    raise RuntimeError(f"Software rendering requested, but actual renderer is {renderer!r}")
  return {"requested": requested, "renderer": renderer, "vendor": vendor,
          "version": version, "software": software if renderer else None}


def current_backend() -> dict:
  """Call only with a current GL context; strict modes fail closed."""
  requested = os.environ.get("KAIHAND_RENDER_BACKEND", "auto")
  try:
    from OpenGL import GL

    def string(name):
      value = GL.glGetString(name)
      return value.decode("utf-8", errors="replace") if value else ""

    values = (string(GL.GL_RENDERER), string(GL.GL_VENDOR), string(GL.GL_VERSION))
  except Exception as error:
    if requested != "auto":
      raise RuntimeError("Cannot inspect the active OpenGL renderer") from error
    return {**describe_backend("", "", "", requested), "query_error": str(error)}
  return describe_backend(*values, requested)
