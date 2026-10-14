"""Offline RGB evidence around physical USB tactile onset and release.

This module reads existing HDF5 arrays and selected RGB frames. It neither
imports MuJoCo nor advances/re-renders the simulation. Human review remains
necessary: a timestamp match alone cannot prove that visible contact is right.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _text(value: Any) -> str:
  return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def event_bracket(times: np.ndarray, event_time: float) -> dict[str, int | None]:
  """Bracket an event without silently substituting a future causal image."""
  before = int(np.searchsorted(times, event_time + 1e-9, side="right")) - 1
  after = int(np.searchsorted(times, event_time - 1e-9, side="left"))
  return {
    "at_or_before": before if before >= 0 else None,
    "at_or_after": after if after < len(times) else None,
  }


def active_intervals(active: np.ndarray) -> list[tuple[int, int]]:
  """Inclusive row intervals; empty contact and one-row events are retained."""
  active = np.asarray(active, dtype=bool)
  changes = np.diff(np.r_[False, active, False].astype(np.int8))
  return list(
    zip(
      np.flatnonzero(changes == 1).tolist(),
      (np.flatnonzero(changes == -1) - 1).tolist(),
      strict=True,
    )
  )


def _sample(
  index: int,
  state_times: np.ndarray,
  force_times: np.ndarray,
  normal: np.ndarray,
  tangent: np.ndarray,
  tangent_load: np.ndarray,
  link_names: list[str],
  threshold: float,
) -> dict[str, Any]:
  active = (normal[index] > threshold) | (tangent_load[index] > threshold)
  return {
    "raw_state_index": int(index),
    "solver_timestamp_s": float(force_times[index]),
    "state_timestamp_s": float(state_times[index]),
    "normal_sum_n": float(normal[index].sum()),
    "tangent_load_sum_n": float(tangent_load[index].sum()),
    "active_link_names": [
      name for name, flag in zip(link_names, active, strict=True) if flag
    ],
    "active_mask": active.tolist(),
    "normal_force_n": normal[index].tolist(),
    "tangent_force_n": tangent[index].tolist(),
    "tangent_load_n": tangent_load[index].tolist(),
  }


def _contact_summary(
  active: np.ndarray, state_times: np.ndarray, force_times: np.ndarray
) -> dict[str, Any]:
  intervals = active_intervals(active)
  if not intervals:
    return {
      "has_signal": False,
      "first_active_index": None,
      "last_active_index": None,
      "first_inactive_after_last_index": None,
      "interval_count": 0,
      "intervals": [],
      "active_samples": 0,
      "onset_censored_by_episode_start": False,
      "offset_censored_by_episode_end": False,
    }
  first, last = intervals[0][0], intervals[-1][1]
  return {
    "has_signal": True,
    "first_active_index": first,
    "last_active_index": last,
    "first_inactive_after_last_index": last + 1 if last + 1 < len(active) else None,
    "first_active_solver_timestamp_s": float(force_times[first]),
    "last_active_solver_timestamp_s": float(force_times[last]),
    "first_active_state_timestamp_s": float(state_times[first]),
    "last_active_state_timestamp_s": float(state_times[last]),
    "interval_count": len(intervals),
    "inactive_gaps_between_first_and_last": len(intervals) - 1,
    "active_samples": int(np.count_nonzero(active)),
    "onset_censored_by_episode_start": bool(active[0]),
    "offset_censored_by_episode_end": bool(active[-1]),
    "intervals": [
      {
        "first_active_index": a,
        "last_active_index": b,
        "first_solver_timestamp_s": float(force_times[a]),
        "last_solver_timestamp_s": float(force_times[b]),
      }
      for a, b in intervals
    ],
  }


def _crop_bounds(
  camera: h5py.Group, index: int, size: tuple[int, int]
) -> tuple[int, int, int, int]:
  """A derived view around recorded right-hand sites, with broad context.

  MuJoCo camera axes are +x right, +y up, -z forward. The crop does not change
  the full-frame evidence and is explicitly only a magnified visual aid.
  """
  width, height = size
  required = (
    "world_from_fingertip",
    "world_from_wrist",
    "world_from_camera",
    "intrinsic",
  )
  if any(name not in camera for name in required):
    return width // 2, 0, width, height
  points = np.vstack(
    (
      camera["world_from_fingertip"][index, 1, :, :3, 3],
      camera["world_from_wrist"][index, 1, :3, 3],
    )
  )
  world_from_camera = np.asarray(camera["world_from_camera"][index])
  local = (points - world_from_camera[:3, 3]) @ world_from_camera[:3, :3]
  local = local[-local[:, 2] > 1e-6]
  if not len(local):
    return width // 2, 0, width, height
  intrinsic = np.asarray(camera["intrinsic"])
  u = intrinsic[0, 0] * local[:, 0] / -local[:, 2] + intrinsic[0, 2]
  v = intrinsic[1, 2] - intrinsic[1, 1] * local[:, 1] / -local[:, 2]
  # Broad square: retain plug/socket context beyond the fingertip sites.
  radius = max(48.0, float(np.ptp(u) + 30.0) / 2, float(np.ptp(v) + 30.0) / 2)
  cx, cy = float((u.min() + u.max()) / 2), float((v.min() + v.max()) / 2)
  left, top = max(0, int(cx - radius)), max(0, int(cy - radius))
  right, bottom = (
    min(width, int(np.ceil(cx + radius))),
    min(height, int(np.ceil(cy + radius))),
  )
  if left >= right or top >= bottom:
    return width // 2, 0, width, height
  return left, top, right, bottom


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
  for path in (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
  ):
    if Path(path).exists():
      return ImageFont.truetype(path, size)
  return ImageFont.load_default()


def _fit(
  image: Image.Image, size: tuple[int, int], *, magnify: bool = False
) -> Image.Image:
  result = Image.new("RGB", size, (235, 238, 242))
  view = image.copy()
  ratio = min(size[0] / view.width, size[1] / view.height)
  if not magnify:
    ratio = min(1.0, ratio)
  view = view.resize(
    (max(1, round(view.width * ratio)), max(1, round(view.height * ratio))),
    Image.Resampling.NEAREST,
  )
  result.paste(view, ((size[0] - view.width) // 2, (size[1] - view.height) // 2))
  return result


def _sheet(output: Path, group: dict[str, Any], source_name: str) -> str:
  rows = group["events"]
  cell_w, cell_h, header_h = 532, 374, 72
  sheet = Image.new("RGB", (cell_w * 2, header_h + max(1, len(rows)) * cell_h), "white")
  draw = ImageDraw.Draw(sheet)
  large, small = _font(17), _font(13)
  draw.text(
    (12, 8),
    f"{source_name} | physical USB pad contacts > {group['threshold_n']:g} N",
    font=large,
    fill="black",
  )
  draw.text(
    (12, 34),
    "Full recorded RGB + derived crop (nearest-pixel enlargement). BEFORE/AFTER use render pose time.",
    font=small,
    fill="black",
  )
  draw.text(
    (12, 52),
    "AFTER is visual context, not a causal training observation. Human inspection required.",
    font=small,
    fill="black",
  )
  if not rows:
    draw.text(
      (20, header_h + 20),
      "No signal at this threshold. See overview start/middle/end images in HTML.",
      font=large,
      fill="black",
    )
  for row, event in enumerate(rows):
    for column, relation in enumerate(("at_or_before", "at_or_after")):
      x, y = column * cell_w, header_h + row * cell_h
      draw.rectangle(
        (x + 2, y + 2, x + cell_w - 3, y + cell_h - 3), outline=(160, 160, 160)
      )
      draw.text(
        (x + 8, y + 7), f"{event['kind']} | {relation}", font=large, fill="black"
      )
      draw.text(
        (x + 8, y + 31),
        f"event row {event['raw_state_index']} | solver {event['solver_timestamp_s']:.6f} s",
        font=small,
        fill="black",
      )
      frame = event["images"][relation]
      if frame is None:
        draw.text(
          (x + 12, y + 88),
          "No bracketing recorded image (episode boundary)",
          font=small,
          fill="black",
        )
        continue
      with Image.open(output / frame["rgb_path"]) as image:
        sheet.paste(_fit(image, (320, 240)), (x + 8, y + 72))
      with Image.open(output / frame["crop_path"]) as image:
        sheet.paste(_fit(image, (188, 188), magnify=True), (x + 336, y + 92))
      draw.text((x + 336, y + 73), "Derived right-hand crop", font=small, fill="black")
      draw.text(
        (x + 8, y + 51),
        f"RGB {frame['camera_frame_index']} | pose {frame['pose_timestamp_s']:.6f} s | delta {frame['event_delta_ms']:+.1f} ms",
        font=small,
        fill="black",
      )
      matched = frame["physical_force_at_image_pose"]
      draw.text(
        (x + 8, y + 318),
        f"RGB touch row {matched['raw_state_index']}: Fn {matched['normal_sum_n']:.4f} N | Ft load {matched['tangent_load_sum_n']:.4f} N",
        font=small,
        fill="black",
      )
      names = ", ".join(
        name.replace("hand_", "").replace("_link4", "").replace("_link6", "")
        for name in matched["active_link_names"]
      )
      draw.text(
        (x + 8, y + 338),
        f"Active at RGB epoch: {names or 'none'}",
        font=small,
        fill="black",
      )
      draw.text(
        (x + 8, y + 355),
        f"Event: normal {event['normal_sum_n']:.4f} N | shear load {event['tangent_load_sum_n']:.4f} N",
        font=small,
        fill="black",
      )
  name = f"contact_sheet_{group['name']}.png"
  sheet.save(output / name)
  return name


def _html(report: dict[str, Any]) -> str:
  escape = html.escape
  sections = [
    '<!doctype html><html lang="zh"><meta charset="utf-8">',
    "<title>USB tactile / RGB review</title><style>body{font:15px sans-serif;max-width:1250px;margin:24px auto;padding:0 16px}table{border-collapse:collapse;width:100%;margin:16px 0}td,th{border:1px solid #ccc;padding:7px;text-align:left;vertical-align:top}img{max-width:100%;image-rendering:pixelated}code{word-break:break-all}.small{font-size:13px;color:#444}</style>",
    "<h1>USB 物理触觉与视觉对应检查</h1>",
    f"<p>原始数据：<code>{escape(report['source'])}</code></p>",
    "<p><b>待人工查看图像。</b>时间一致只能证明索引对应；还需确认接触发生时手和 USB 的视觉关系合理。整图是原始 RGB，放大裁剪只作查看辅助，不是新采集视角。</p>",
    f"<p>物理触觉 {report['physics_hz']:g} Hz；相机配置 {report['configured_camera_hz']:g} Hz；RGB {report['rgb_shape'][1]}×{report['rgb_shape'][0]}。相机画面对应 pose_timestamp；采集 timestamp 是积分后时刻。</p>",
    "<p>事件使用每指法向力或非抵消切向载荷超过阈值。表格保留 10 指顺序、掩码、法向力、带符号切向力和切向载荷于 JSON。事件前图是因果上下文；事件后图仅用于人工对照，不能作为该时刻训练输入。</p>",
    "<p>空间分辨率限制：头部整图和局部裁剪可检查抓住、拿起和松手是否对应；裁剪不增加原始像素，无法据此判定微米级间隙或精确触觉形变。接触精确时刻应结合物理接触记录与时间检查。</p>",
  ]
  if report["warnings"]:
    sections.append(
      "<ul>"
      + "".join(f"<li>{escape(value)}</li>" for value in report["warnings"])
      + "</ul>"
    )
  for group in report["thresholds"]:
    sections.extend(
      (
        f"<h2>{escape(group['name'])} — 阈值 {group['threshold_n']:g} N</h2>",
        f"<p>有效触觉样本 {group['global']['active_samples']}；接触区间 {group['global']['interval_count']}。末次有信号与其后首次无信号分别列出。</p>",
        f'<a href="{group["contact_sheet"]}"><img src="{group["contact_sheet"]}" alt="contact sheet"></a>',
        "<table><thead><tr><th>事件 / 触觉帧</th><th>事件前或同刻 RGB</th><th>事件后或同刻 RGB</th></tr></thead><tbody>",
      )
    )
    for event in group["events"]:
      sections.append(
        f"<tr><td>{escape(event['kind'])}<br>row {event['raw_state_index']}<br>solver {event['solver_timestamp_s']:.6f} s<br>state {event['state_timestamp_s']:.6f} s<br>Σ Fn={event['normal_sum_n']:.6g} N<br>Σ Ft load={event['tangent_load_sum_n']:.6g} N</td>"
      )
      for relation in ("at_or_before", "at_or_after"):
        frame = event["images"][relation]
        if frame is None:
          sections.append("<td>无对应边界图像</td>")
        else:
          sections.append(
            f'<td>RGB {frame["camera_frame_index"]}; raw row {frame["raw_state_index"]}<br>pose {frame["pose_timestamp_s"]:.6f} s; Δ {frame["event_delta_ms"]:+.1f} ms<br><a href="{frame["rgb_path"]}"><img width="320" src="{frame["rgb_path"]}" alt="original RGB"></a><br><a href="{frame["crop_path"]}">查看派生的右手局部裁剪</a></td>'
          )
      sections.append("</tr>")
    sections.extend(
      (
        "</tbody></table>",
        "<table><thead><tr><th>触觉链接</th><th>开始帧 / 时刻</th><th>最后有信号帧 / 时刻</th><th>区间数</th></tr></thead><tbody>",
      )
    )
    for name, summary in group["per_link"].items():
      onset = (
        f"{summary['first_active_index']} / {summary['first_active_solver_timestamp_s']:.6f} s"
        if summary["has_signal"]
        else "无信号"
      )
      offset = (
        f"{summary['last_active_index']} / {summary['last_active_solver_timestamp_s']:.6f} s"
        if summary["has_signal"]
        else "无信号"
      )
      sections.append(
        f"<tr><td>{escape(name)}</td><td>{onset}</td><td>{offset}</td><td>{summary['interval_count']}</td></tr>"
      )
    sections.append("</tbody></table>")
  sections.append(
    "<h2>原始场景概览</h2><p>用于空触觉场景和开始 / 中间 / 结束状态检查。</p>"
  )
  for frame in report["overview_images"]:
    sections.append(
      f'<a href="{frame["rgb_path"]}"><img width="320" src="{frame["rgb_path"]}" alt="overview frame {frame["camera_frame_index"]}"></a> '
    )
  sections.append(
    '<p><a href="review.json">完整时刻、逐指力量和索引 JSON</a></p></html>'
  )
  return "\n".join(sections) + "\n"


def review_usb_tactile(
  source: Path | str,
  output: Path | str,
  *,
  camera_name: str = "head",
  signal_threshold_n: float = 1e-6,
  practical_threshold_n: float = 0.01,
) -> dict[str, Any]:
  """Write immutable selected-frame evidence to a new output directory."""
  source, output = Path(source).resolve(), Path(output)
  thresholds = (("physical", signal_threshold_n), ("practical", practical_threshold_n))
  if any(not np.isfinite(value) or value < 0 for _, value in thresholds):
    raise ValueError("force thresholds must be finite and nonnegative")
  if output.exists():
    raise FileExistsError(f"review output already exists: {output}")
  with h5py.File(source, "r") as file:
    if (
      _text(file.attrs.get("contact_force_source", ""))
      != "solver_contact_distributed_taxel_v1"
    ):
      raise ValueError("review requires the physical solver tactile stream")
    force = file["tactile_contact_force"]
    if (
      _text(force.attrs.get("timestamp_reference", ""))
      != "/tactile_contact_force/timestamp"
    ):
      raise ValueError("explicit solver force timestamps are required")
    if _text(force.attrs.get("force_unit", "")) != "N":
      raise ValueError("physical force must be stored in newtons")
    state_times = np.asarray(file["state/timestamp"][:], dtype=float)
    force_times = np.asarray(force["timestamp"][:], dtype=float)
    normal = np.asarray(force["normal_force_n"][:], dtype=float)
    tangent = np.asarray(force["tangent_force_n"][:], dtype=float)
    tangent_load = np.asarray(force["tangent_load_n"][:], dtype=float)
    link_names = [_text(name) for name in force["link_names"][:]]
    expected = [
      f"hand_{side}_{finger}_link{6 if finger == 'thumb' else 4}"
      for side in ("l", "r")
      for finger in ("thumb", "index", "middle", "ring", "pinky")
    ]
    if len(link_names) != 10 or set(link_names) != set(expected):
      raise ValueError("expected all ten left/right physical fingertip channels")
    samples = len(state_times)
    if (
      samples == 0
      or force_times.shape != (samples,)
      or normal.shape != (samples, 10)
      or tangent.shape != (samples, 10, 2)
      or tangent_load.shape != (samples, 10)
    ):
      raise ValueError("physical force/state arrays have inconsistent shapes")
    if any(
      not np.isfinite(value).all()
      for value in (state_times, force_times, normal, tangent, tangent_load)
    ):
      raise ValueError("physical force/state timestamps must be finite")
    if np.any(normal < 0) or np.any(tangent_load < 0):
      raise ValueError("force magnitudes must be nonnegative")
    if np.any(np.diff(state_times) <= 0) or np.any(np.diff(force_times) < -1e-9):
      raise ValueError("state/force clocks are not monotonic")
    camera = file[f"cameras/{camera_name}"]
    camera_times = np.asarray(camera["pose_timestamp"][:], dtype=float)
    acquisition_times = np.asarray(camera["timestamp"][:], dtype=float)
    state_indices = np.asarray(camera["state_index"][:], dtype=np.int64)
    rgb = camera["rgb"]
    if (
      not len(camera_times)
      or rgb.shape[0] != len(camera_times)
      or acquisition_times.shape != camera_times.shape
      or state_indices.shape != camera_times.shape
    ):
      raise ValueError("camera arrays are empty or have inconsistent lengths")
    if (
      not np.isfinite(camera_times).all()
      or not np.isfinite(acquisition_times).all()
      or np.any(np.diff(camera_times) <= 0)
      or np.any(np.diff(acquisition_times) <= 0)
    ):
      raise ValueError("camera clocks must be finite and strictly increasing")
    if np.any(state_indices < 0) or np.any(state_indices >= samples):
      raise ValueError("camera raw state index is out of range")
    if not np.allclose(force_times[state_indices], camera_times, rtol=0, atol=1e-9):
      raise ValueError("RGB pose clock does not match indexed physical tactile clock")
    if not np.allclose(
      state_times[state_indices], acquisition_times, rtol=0, atol=1e-9
    ):
      raise ValueError("RGB acquisition clock does not match indexed state clock")
    if len(rgb.shape) != 4 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
      raise ValueError("expected uint8 RGB camera frames")
    output.mkdir(parents=True)
    (output / "frames").mkdir()
    written: dict[int, dict[str, Any]] = {}

    def save_frame(index: int, event_time: float, threshold: float) -> dict[str, Any]:
      if index not in written:
        image = Image.fromarray(np.asarray(rgb[index]), mode="RGB")
        rgb_path, crop_path = (
          f"frames/rgb_{index:05d}.png",
          f"frames/crop_{index:05d}.png",
        )
        crop = _crop_bounds(camera, index, image.size)
        image.save(output / rgb_path)
        image.crop(crop).save(output / crop_path)
        written[index] = {
          "camera_frame_index": index,
          "raw_state_index": int(state_indices[index]),
          "pose_timestamp_s": float(camera_times[index]),
          "acquisition_timestamp_s": float(acquisition_times[index]),
          "rgb_path": rgb_path,
          "crop_path": crop_path,
          "crop_box_xyxy": list(crop),
          "crop_kind": "derived right-hand context; nearest-pixel magnification in sheet",
        }
      return {
        **written[index],
        "event_delta_ms": 1000 * float(camera_times[index] - event_time),
        "physical_force_at_image_pose": _sample(
          int(state_indices[index]),
          state_times,
          force_times,
          normal,
          tangent,
          tangent_load,
          link_names,
          threshold,
        ),
      }

    report: dict[str, Any] = {
      "schema": "usb_physical_tactile_visual_review_v1",
      "source": str(source),
      "camera": camera_name,
      "raw_state_samples": samples,
      "rgb_frames": len(camera_times),
      "rgb_shape": list(rgb.shape[1:]),
      "physics_hz": float(file.attrs.get("physics_hz", 0)),
      "configured_camera_hz": float(file.attrs.get("camera_hz", 0)),
      "link_names": link_names,
      "review_status": "requires_human_review",
      "signal_definition": "any physical USB pad: normal_force_n > threshold OR tangent_load_n > threshold; tangent_load is non-cancelling contact magnitude, tangent_force_n is signed pad XY",
      "clock_semantics": "events and RGB are paired by recorded solver/render pose_timestamp; accepted USB post_step_forward_v1 recordings have equal state, solver and indexed camera acquisition/pose timestamps after the controller's forward refresh; this review pairs recorded arrays and does not establish clock provenance; after-event images are visual review context only",
      "visual_limitations": "The head RGB is a whole-scene view. A crop magnifies existing pixels and does not add spatial resolution; micrometre contact gaps or exact tactile pad deformation cannot be established visually. Inspect gross grasp/release correspondence and use contact/clock audits for sub-frame timing.",
      "camera_pose_delta_s": np.diff(camera_times).tolist(),
      "rgb_tactile_solver_clock_max_error_s": float(
        np.max(np.abs(force_times[state_indices] - camera_times))
      ),
      "warnings": [],
      "thresholds": [],
      "overview_images": [],
    }
    force_metadata = json.loads(_text(force.attrs.get("metadata_json", "{}")))
    targets = force_metadata.get("target_geom_names", [])
    if not targets or any(not name.startswith("usb_plug_") for name in targets):
      report["warnings"].append(
        "Force metadata does not prove that every target is a USB plug collider; inspect the raw source audit."
      )
    for name, threshold in thresholds:
      active = (normal > threshold) | (tangent_load > threshold)
      any_active = np.any(active, axis=1)
      summary = _contact_summary(any_active, state_times, force_times)
      group: dict[str, Any] = {
        "name": name,
        "threshold_n": threshold,
        "global": summary,
        "per_link": {
          link: _contact_summary(active[:, j], state_times, force_times)
          for j, link in enumerate(link_names)
        },
        "shear_only_active_samples": int(
          np.count_nonzero(
            np.any((normal <= threshold) & (tangent_load > threshold), axis=1)
          )
        ),
        "events": [],
      }
      if summary["has_signal"]:
        total = normal.sum(axis=1)
        peak_candidates = np.flatnonzero(any_active)
        peak = int(peak_candidates[np.argmax(total[peak_candidates])])
        event_rows = [
          ("first_signal", summary["first_active_index"]),
          ("peak_normal_load", peak),
          ("last_signal", summary["last_active_index"]),
          ("first_inactive_after_last", summary["first_inactive_after_last_index"]),
        ]
        if summary["first_inactive_after_last_index"] is not None:
          context_time = float(force_times[summary["last_active_index"]]) + 0.25
          context_index = min(
            int(np.searchsorted(force_times, context_time)), samples - 1
          )
          if context_index > summary["first_inactive_after_last_index"]:
            event_rows.append(("post_release_context", context_index))
        for kind, index in event_rows:
          if index is None:
            continue
          event = _sample(
            index,
            state_times,
            force_times,
            normal,
            tangent,
            tangent_load,
            link_names,
            threshold,
          )
          event["kind"] = kind
          event["images"] = {
            relation: None
            if frame_index is None
            else save_frame(frame_index, event["solver_timestamp_s"], threshold)
            for relation, frame_index in event_bracket(
              camera_times, event["solver_timestamp_s"]
            ).items()
          }
          group["events"].append(event)
        if summary["offset_censored_by_episode_end"]:
          report["warnings"].append(
            f"{name}: force remains active at episode end; an observed release boundary is unavailable."
          )
        if summary["onset_censored_by_episode_start"]:
          report["warnings"].append(
            f"{name}: force is already active at episode start; an observed onset boundary is unavailable."
          )
      else:
        report["warnings"].append(
          f"{name}: no physical tactile signal above {threshold:g} N; inspect overview and collection outcome."
        )
      group["contact_sheet"] = _sheet(output, group, source.name)
      report["thresholds"].append(group)
    for index in sorted(set((0, len(camera_times) // 2, len(camera_times) - 1))):
      report["overview_images"].append(
        save_frame(index, float(camera_times[index]), signal_threshold_n)
      )
    report["selected_rgb_frames"] = len(written)
    (output / "review.json").write_text(
      json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
      encoding="utf-8",
    )
    (output / "index.html").write_text(_html(report), encoding="utf-8")
  return report
