"""Stream synchronized camera and signed three-channel tactile videos."""

from __future__ import annotations

import json
from pathlib import Path

from kaihand_tactile_env.shared.task_video import _FfmpegPipeWriter
import numpy as np
from PIL import Image
from PIL import ImageDraw

FINGERS = ("thumb", "index", "middle", "ring", "pinky")


def tactile_frame(maps: np.ndarray, timestamp: float, scale: float) -> np.ndarray:
    """Fixed-scale visualization only; the original forces sent to CPM remain untouched."""
    canvas = Image.new("RGB", (1040, 620), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (18, 10), f"Right fingertips | {timestamp:.3f} s | N per taxel | fixed display scale {scale:g} N", fill="black"
    )
    for channel, label in enumerate(("normal", "tangent0 (signed)", "tangent1 (signed)")):
        draw.text((18, 38 + channel * 190), label, fill="black")
        for finger, name in enumerate(FINGERS):
            values = np.flipud(maps[finger, channel])
            if channel == 0:
                intensity = np.clip(values / scale, 0.0, 1.0)
                pixels = np.stack([1 - intensity, 1 - 0.6 * intensity, np.ones_like(intensity)], axis=-1)
            else:
                intensity = np.clip(values / scale, -1.0, 1.0)
                pixels = np.stack(
                    [1 - np.maximum(-intensity, 0), 1 - np.abs(intensity), 1 - np.maximum(intensity, 0)], axis=-1
                )
            x, y = 18 + finger * 204, 59 + channel * 190
            image = Image.fromarray((pixels * 255).astype(np.uint8)).resize((185, 135), Image.Resampling.NEAREST)
            canvas.paste(image, (x, y))
            draw.text((x, y + 140), f"{name}: sum={maps[finger, channel].sum():+.4f} N", fill="black")
    return np.asarray(canvas)


class SynchronizedVideos:
    """Use one renderer/provider snapshot; never step or forward simulation physics."""

    def __init__(
        self, output: Path, renderer, cameras: dict, provider, *, fps: int, scale: float = 0.05,
        include_tactile: bool = True,
    ):
        if fps not in (5, 10) or not np.isfinite(scale) or scale <= 0:
            raise ValueError("Use 5/10 FPS and a positive finite tactile display scale")
        self.output = output
        output.mkdir(exist_ok=False)
        self.renderer, self.cameras, self.provider = renderer, cameras, provider
        self.fps, self.stride, self.scale = fps, 30 // fps, scale
        self.include_tactile = include_tactile
        self.writers = {}
        self.count = 0
        self.first_time = self.last_time = None
        self.closed = False
        self.log = None
        try:
            for name, camera in cameras.items():
                self.writers[name] = _FfmpegPipeWriter(
                    output / f"{name}.mp4", fps=fps, width=camera.width, height=camera.height
                )
            if self.include_tactile:
                self.writers["tactile"] = _FfmpegPipeWriter(output / "tactile.mp4", fps=fps, width=1040, height=620)
            self.log = (output / "frames.jsonl").open("x")
        except BaseException:
            self.abort()
            raise

    def capture(self, simulation, tick: int, *, force: bool = False) -> None:
        timestamp = float(simulation.data.time)
        if (not force and tick % self.stride) or (self.last_time is not None and timestamp <= self.last_time + 1e-10):
            return
        for name, camera in self.cameras.items():
            self.writers[name].write(self.renderer.capture(simulation.data, camera)["rgb"])
        if self.include_tactile:
            sample = self.provider.read(simulation.data)
            maps = np.stack(
                [sample.normal_taxel_force_n, sample.tangent_taxel_force_n[..., 0], sample.tangent_taxel_force_n[..., 1]],
                axis=1,
            ).astype(np.float32)
            self.writers["tactile"].write(tactile_frame(maps, timestamp, self.scale))
        self.log.write(json.dumps({"frame": self.count, "sim_time_s": timestamp}) + "\n")
        self.log.flush()
        self.first_time = timestamp if self.first_time is None else self.first_time
        self.last_time = timestamp
        self.count += 1

    def finish(self) -> dict:
        try:
            for writer in self.writers.values():
                writer.close()
            metadata = {
                "fps": self.fps,
                "frames": self.count,
                "first_time_s": self.first_time,
                "last_time_s": self.last_time,
                "camera_names": list(self.cameras),
                "model_input_cameras": ["head", "right_wrist"],
                "review_only_cameras": ["overhead"],
                "tactile_channels": ["normal", "tangent0", "tangent1"],
                "tactile_video_saved": self.include_tactile,
                "finger_order": list(FINGERS),
                "tactile_target_geometry": "card_core_geom",
                "tactile_unit": "N per taxel",
                "tactile_display_scale_n": self.scale,
                "display_clipping_only": True,
                "clock": "simulation time; final off-grid frame allowed",
                "completed": True,
            }
            (self.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
            return metadata
        finally:
            self.abort()

    def abort(self) -> None:
        if self.closed:
            return
        for writer in self.writers.values():
            writer.abort()
        if self.log is not None:
            self.log.close()
        self.closed = True
