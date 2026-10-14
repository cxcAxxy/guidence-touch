"""Load recorded Whiteboard spatial tactile grids without rerunning physics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from .contact_tactile import SOURCE, _stable_pad_basis
from .tactile import load_fingertip_layout
from .whiteboard_contact_reference import (
  DEFAULT_DEMO_ROOT,
  _right_indices,
  _text,
  load_historical_whiteboard_contact_trial,
  load_whiteboard_contact_reference,
)
from .whiteboard_tactile_comparison import TactileTrace

_SOURCE_ROOT = Path(__file__).resolve().parents[3]
_LAYOUT_PATH = (
  "src/kaihand_tactile_env/tactile/taxels/kaihand_bimanual_fingertip_genesis.json"
)
_COORDINATE_SOURCE_FILES = (
  _LAYOUT_PATH,
  "src/kaihand_tactile_env/shared/tactile.py",
  "src/kaihand_tactile_env/shared/contact_tactile.py",
)


def _canonical_coordinates() -> tuple[np.ndarray, np.ndarray]:
  layout = load_fingertip_layout(_SOURCE_ROOT / _LAYOUT_PATH)
  names = np.asarray(layout.body_names)
  positions, bases = [], []
  for finger in ("thumb", "index", "middle", "ring", "pinky"):
    name = f"hand_r_{finger}_link{6 if finger == 'thumb' else 4}"
    indices = np.flatnonzero(names == name)
    if len(indices) != 35 or any(layout.grid_shape[int(i)] != (7, 5) for i in indices):
      raise ValueError(f"{name} 缺少规范的 7×5 指尖网格")
    pos = np.asarray(layout.local_pos[indices], dtype=np.float64)
    _, basis = _stable_pad_basis(pos, layout.local_normal[indices])
    positions.append(pos)
    bases.append(basis)
  return np.stack(positions), np.stack(bases)


def load_whiteboard_tactile_reference(
  *,
  task: str = "whiteboard-wipe",
  reference_hdf5: Path | None = None,
  demo_root: Path = DEFAULT_DEMO_ROOT,
) -> TactileTrace:
  contact = load_whiteboard_contact_reference(
    task=task,
    reference_hdf5=reference_hdf5,
    demo_root=demo_root,
  )
  with h5py.File(contact.metadata["source_hdf5"], "r") as file:
    group = file["tactile_contact_force"]
    indices = _right_indices([_text(name) for name in group["link_names"][:]])
    required = (
      "normal_taxel_force_n",
      "tangent_taxel_force_n",
      "taxel_positions_local_m",
      "tangent_basis_local",
    )
    missing = [key for key in required if key not in group]
    if missing:
      raise ValueError(f"专家缺少空间触觉字段：{missing}")
    normal = np.asarray(group["normal_taxel_force_n"][:], dtype=np.float64)[:, indices]
    tangent = np.asarray(group["tangent_taxel_force_n"][:], dtype=np.float64)[
      :, indices
    ]
    recorded_force = np.asarray(group["normal_force_n"][:], dtype=np.float64)[
      :, indices
    ]
    if not np.allclose(
      normal.sum(axis=(2, 3)), recorded_force, rtol=1e-6, atol=1e-8, equal_nan=True
    ):
      raise ValueError("专家逐格法向力之和与已记录的每指总力不一致")
    positions = np.asarray(group["taxel_positions_local_m"][:], dtype=np.float64)[
      indices
    ]
    basis = np.asarray(group["tangent_basis_local"][:], dtype=np.float64)[indices]
    tactile_metadata = json.loads(_text(group.attrs.get("metadata_json", "{}")))
    source = tactile_metadata.get("source", SOURCE)
    if source != SOURCE:
      raise ValueError(f"专家触觉来源不兼容：{source}")
  metadata = dict(contact.metadata)
  metadata.update(
    source=source,
    spatial_tactile_metadata=tactile_metadata,
    coordinate_source="专家 HDF5 原生指尖局部坐标和切向基",
    board_contact_time_s=contact.time_s[
      contact.channels["board_normal_n"] > 0.05
    ].tolist(),
  )
  return TactileTrace(contact.time_s, normal, tangent, positions, basis, metadata)


def _verify_frozen_coordinates(
  summary: dict, trial: Path, reference: TactileTrace
) -> dict:
  """Infer old NPZ coordinates only with frozen layout/code hash evidence."""
  manifest_value = summary.get("deployment_manifest")
  if not manifest_value:
    raise ValueError(
      "历史网格未保存坐标，且缺少冻结 deployment manifest；不能推断毫米坐标"
    )
  manifest_path = Path(manifest_value).expanduser()
  if not manifest_path.is_absolute():
    manifest_path = trial / manifest_path
  manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
  rows = {row["path"]: row for row in manifest.get("evaluation_source_files", [])}
  verified = []
  for relative in _COORDINATE_SOURCE_FILES:
    row = rows.get(relative)
    if row is None:
      raise ValueError(f"冻结清单缺少指尖坐标来源：{relative}")
    current = hashlib.sha256((_SOURCE_ROOT / relative).read_bytes()).hexdigest()
    if current != row.get("sha256"):
      raise ValueError(f"指尖坐标来源与历史版本不一致：{relative}；不能推断毫米坐标")
    verified.append({"path": relative, "sha256": current})
  positions, basis = _canonical_coordinates()
  if not np.allclose(positions, reference.taxel_positions_local_m, rtol=0, atol=1e-9):
    raise ValueError("专家指尖坐标与已验证的模型网格不一致")
  if not np.allclose(basis, reference.tangent_basis_local, rtol=0, atol=1e-7):
    raise ValueError("专家指尖切向坐标基与已验证的模型网格不一致")
  return {
    "status": "verified",
    "manifest": str(manifest_path.resolve()),
    "files": verified,
    "coordinate_tolerance_m": 1e-9,
  }


def load_historical_whiteboard_tactile_trial(
  trial_dir: Path, reference: TactileTrace
) -> TactileTrace:
  trial = Path(trial_dir).expanduser().resolve()
  summary = json.loads((trial / "summary.json").read_text(encoding="utf-8"))
  if summary.get("task") != reference.metadata.get("task"):
    raise ValueError("模型与专家必须对应同一白板任务")
  trace_path = next(
    (
      p
      for p in (
        trial / "evaluation_rollout_trace.npz",
        trial / "review" / "evaluation_rollout_trace.npz",
      )
      if p.is_file()
    ),
    None,
  )
  if trace_path is None:
    raise FileNotFoundError(f"缺少已有空间触觉轨迹：{trial}")
  source = summary.get("video", {}).get("tactile_source")
  comparison_path = trace_path.parent / "evaluation_comparison.json"
  if not source and comparison_path.is_file():
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    source = comparison.get("tactile", {}).get("source")
  if source != reference.metadata["source"]:
    raise ValueError(f"模型／专家触觉来源不一致或无法验证：{source!r}")
  proof = _verify_frozen_coordinates(summary, trial, reference)
  with np.load(trace_path, allow_pickle=False) as data:
    indices = _right_indices([_text(name) for name in data["fingertip_link_names"]])
    times = np.asarray(data["simulation_time_s"], dtype=np.float64)
    normal = np.asarray(data["normal_taxel_force_n"], dtype=np.float64)[:, indices]
    tangent = np.asarray(data["tangent_taxel_force_n"], dtype=np.float64)[:, indices]
    recorded_force = np.asarray(data["fingertip_normal_force_n"], dtype=np.float64)[
      :, indices
    ]
    if not np.allclose(
      normal.sum(axis=(2, 3)), recorded_force, rtol=1e-6, atol=1e-8, equal_nan=True
    ):
      raise ValueError("模型逐格法向力之和与已记录的每指总力不一致")
  contact = load_historical_whiteboard_contact_trial(trial)
  metadata = dict(contact.metadata)
  metadata.update(
    source=source,
    coordinate_source="冻结清单验证了规范网格与坐标基；使用与专家一致的毫米坐标",
    coordinate_verification=proof,
    original_artifacts_modified=False,
    spatial_sampling="已有控制采样网格，非逐物理步；不从视频颜色恢复力值",
    kernel_verification_note="专家记录的核宽度可核验；历史模型只保存算法来源，未单独记录实际核宽度",
  )
  return TactileTrace(
    times - times[0],
    normal,
    tangent,
    reference.taxel_positions_local_m.copy(),
    reference.tangent_basis_local.copy(),
    metadata,
  )


def choose_snapshot_times(
  rollout: TactileTrace, reference: TactileTrace
) -> list[float]:
  """Choose grasp/board/late snapshots on the unchanged expert physical clock."""
  end = min(float(rollout.time_s[-1]), float(reference.time_s[-1]))
  board = np.asarray(reference.metadata.get("board_contact_time_s", []), dtype=float)
  board = board[(board >= reference.time_s[0]) & (board <= end)]
  normal = reference.normal_taxel_force_n.sum(axis=(2, 3)).sum(axis=1)
  grasp = reference.time_s[(normal > 0.05) & (reference.time_s <= end)]
  if len(board):
    before = grasp[grasp < board[0]]
    first = float(before[len(before) // 2]) if len(before) else float(board[0])
    return [first, float(board[len(board) // 3]), float(board[-1])]
  if len(grasp):
    return [
      float(grasp[int((len(grasp) - 1) * fraction)]) for fraction in (0.2, 0.5, 0.9)
    ]
  return np.linspace(float(reference.time_s[0]), end, 3).tolist()
