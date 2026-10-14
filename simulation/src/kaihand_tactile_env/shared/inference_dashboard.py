"""Live overhead-camera and fingertip-tactile display for policy rollouts."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from .config import CameraConfig
from .rendering import WorkcellRenderer
from .simulation import ArmHandSimulation
from .tactile import GenesisProbeTactileProvider

FINGERS = (
  ("thumb", "link6"),
  ("index", "link4"),
  ("middle", "link4"),
  ("ring", "link4"),
  ("pinky", "link4"),
)


def fingertip_grid_indices(
  tactile: GenesisProbeTactileProvider,
) -> dict[str, np.ndarray]:
  """Return each canonical fingertip's 7x5 probe indices."""
  body_names = np.asarray(tactile.layout.body_names)
  result: dict[str, np.ndarray] = {}
  for side in ("l", "r"):
    for finger, link in FINGERS:
      name = f"hand_{side}_{finger}_{link}"
      indices = np.flatnonzero(body_names == name)
      if len(indices) != 35:
        raise RuntimeError(f"{name}: expected 35 tactile probes, got {len(indices)}")
      if tactile.layout.grid_shape[int(indices[0])] != (7, 5):
        raise RuntimeError(f"{name}: expected a 7x5 tactile grid")
      result[name] = indices
  if len(set(np.concatenate(tuple(result.values())).tolist())) != tactile.layout.count:
    raise RuntimeError("fingertip grids do not cover the tactile layout exactly once")
  return result


class InferenceSensorDashboard:
  """Non-blocking overhead RGB and bilateral 2x5 tactile heatmaps."""

  def __init__(
    self,
    simulation: ArmHandSimulation,
    tactile: GenesisProbeTactileProvider,
    renderer: WorkcellRenderer,
    overhead_camera: CameraConfig,
    *,
    max_depth_mm: float = 3.0,
    tactile_refresh_hz: float = 10.0,
    camera_refresh_hz: float = 5.0,
  ) -> None:
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
      raise RuntimeError(
        "sensor dashboard requires a desktop display or X11 forwarding"
      )
    if max_depth_mm <= 0.0:
      raise ValueError("max_depth_mm must be positive")
    if tactile_refresh_hz <= 0.0 or camera_refresh_hz <= 0.0:
      raise ValueError("dashboard refresh rates must be positive")
    import matplotlib

    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    self.simulation = simulation
    self.tactile = tactile
    self.renderer = renderer
    self.overhead_camera = overhead_camera
    self.plt = plt
    self.closed = False
    self._indices = fingertip_grid_indices(tactile)
    self._tactile_period = 1.0 / tactile_refresh_hz
    self._camera_period = 1.0 / camera_refresh_hz
    self._next_tactile_time = float(simulation.data.time)
    self._next_camera_time = float(simulation.data.time)
    self._peak_depth = np.zeros(tactile.layout.count, dtype=np.float64)
    self._latest_sample = tactile.read(simulation.data)

    self.figure = self.plt.figure(figsize=(18, 8.5))
    grid = self.figure.add_gridspec(
      2,
      6,
      width_ratios=(1, 1, 1, 1, 1, 2.35),
      wspace=0.48,
      hspace=0.65,
    )
    axes = np.asarray(
      [
        [self.figure.add_subplot(grid[row, column]) for column in range(5)]
        for row in range(2)
      ]
    )
    overhead_axis = self.figure.add_subplot(grid[:, 5])
    overhead_axis.set_title("Overhead camera (visualization only)")
    overhead_axis.axis("off")
    self._overhead_image = overhead_axis.imshow(
      np.zeros((overhead_camera.height, overhead_camera.width, 3), dtype=np.uint8)
    )

    self._heatmaps: dict[str, Any] = {}
    self._empty_labels: dict[str, Any] = {}
    for row, side in enumerate(("l", "r")):
      for column, (finger, link) in enumerate(FINGERS):
        name = f"hand_{side}_{finger}_{link}"
        axis = axes[row, column]
        heatmap = axis.imshow(
          np.zeros((7, 5)),
          origin="lower",
          interpolation="nearest",
          cmap="inferno",
          vmin=0.0,
          vmax=max_depth_mm,
          aspect="equal",
        )
        axis.set_title(f"{'L' if side == 'l' else 'R'} {finger.title()}", fontsize=9)
        axis.set_xticks(range(5))
        axis.set_yticks(range(7))
        axis.tick_params(labelsize=7)
        if column == 0:
          axis.set_ylabel("Left" if side == "l" else "Right")
        self._heatmaps[name] = heatmap
        self._empty_labels[name] = axis.text(
          0.5,
          0.5,
          "NO CONTACT",
          color="0.7",
          fontsize=8,
          ha="center",
          va="center",
          transform=axis.transAxes,
        )
    colorbar = self.figure.colorbar(
      next(iter(self._heatmaps.values())),
      ax=axes.ravel().tolist(),
      shrink=0.82,
    )
    colorbar.set_label("probe depth [mm]")
    self.figure.suptitle(
      "EgoSteer rollout sensors — 10 fingertip tactile arrays + overhead RGB"
    )
    self._sensor_status = self.figure.text(
      0.5, 0.055, "tactile ready", ha="center", fontsize=10
    )
    self._policy_status = self.figure.text(
      0.5,
      0.025,
      "policy: connecting",
      ha="center",
      fontsize=10,
    )
    self.figure.subplots_adjust(
      left=0.045,
      right=0.95,
      bottom=0.11,
      top=0.90,
      wspace=0.48,
      hspace=0.65,
    )
    self.figure.canvas.manager.set_window_title("EgoSteer rollout sensor dashboard")
    self.figure.canvas.mpl_connect("close_event", self._on_close)
    self.plt.show(block=False)
    self.sync(force=True)

  def _on_close(self, _event: Any) -> None:
    self.closed = True

  def is_running(self) -> bool:
    return not self.closed and self.plt.fignum_exists(self.figure.number)

  def set_policy_status(self, message: str) -> None:
    self._policy_status.set_text(message)
    self.figure.canvas.draw_idle()

  def sample_physics(self) -> None:
    """Read 500 Hz tactile state and retain peaks until the next GUI draw."""
    self._latest_sample = self.tactile.read(self.simulation.data)
    np.maximum(self._peak_depth, self.tactile.probe_depth, out=self._peak_depth)

  def sync(self, *, force: bool = False) -> None:
    """Process GUI events and redraw sensors at their configured rates."""
    if not self.is_running():
      return
    canvas = self.figure.canvas
    canvas.flush_events()
    sim_time = float(self.simulation.data.time)
    tolerance = 0.5 * self.simulation.timestep
    tactile_due = force or sim_time + tolerance >= self._next_tactile_time
    camera_due = force or sim_time + tolerance >= self._next_camera_time
    if not tactile_due and not camera_due:
      return

    if tactile_due:
      displayed_depth = np.maximum(self._peak_depth, self.tactile.probe_depth)
      for name, indices in self._indices.items():
        depth_mm = displayed_depth[indices].reshape(7, 5) * 1000.0
        self._heatmaps[name].set_data(depth_mm)
        self._empty_labels[name].set_visible(not np.any(depth_mm > 0.0))
        link_index = self._latest_sample.link_names.index(name)
        self._heatmaps[name].axes.title.set_text(
          f"{'L' if '_l_' in name else 'R'} {name.split('_')[2].title()}\n"
          f"active={self._latest_sample.contact_count[link_index]}  "
          f"max={depth_mm.max():.2f} mm"
        )
      active = int(np.count_nonzero(self.tactile.probe_contact))
      self._sensor_status.set_text(
        f"sim_t={sim_time:.3f}s | active probes={active} | "
        f"peak depth={displayed_depth.max() * 1000.0:.3f} mm"
      )
      self._peak_depth.fill(0.0)
      self._next_tactile_time = sim_time + self._tactile_period

    if camera_due:
      overhead = self.renderer.capture(self.simulation.data, self.overhead_camera)[
        "rgb"
      ]
      self._overhead_image.set_data(overhead)
      self._next_camera_time = sim_time + self._camera_period

    canvas.draw_idle()
    canvas.flush_events()

  def close(self) -> None:
    if not self.closed:
      self.closed = True
      self.plt.close(self.figure)


__all__ = ["InferenceSensorDashboard", "fingertip_grid_indices"]
