"""Fit CPM-only training bounds and audit all observation normalization paths."""

# ruff: noqa: RUF001
# Chinese visualization labels intentionally use Chinese punctuation.
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch

from openpi.touchguide.normalization import CPMNormalization

ROOT = Path(
    "/path/to/datasets/touchguidence/augmentation/poker_fingertip_recovery_rules_v1_random10"
)
NAMES = [
    *(f"arm_joint{i}" for i in range(1, 8)),
    *(
        f"{finger}_joint{i}"
        for finger in ("thumb", "index", "middle", "ring", "pinky")
        for i in ((1, 2, 3, 5) if finger == "thumb" else (1, 2, 3, 4))
    ),
]


def encode(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value))


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, default=encode, ensure_ascii=False, indent=2))


def read(root, kind, eid):
    with h5py.File(root / kind / "episodes" / f"episode_{eid:06d}.h5") as f:
        state = f["state_raw"][:].astype(np.float64)
        physical = f["actions_absolute"][:].astype(np.float64)
        physical[..., :7] -= state[:, None, :7]
    if not np.isfinite(state).all() or not np.isfinite(physical).all():
        raise ValueError(f"Nonfinite state/actions: {kind}, {eid}")
    return state, physical


def fit_episode(task):
    root, eid, split = task
    result = {"eid": eid, "split": split}
    for kind in ("clean", "augmented"):
        state, physical = read(root, kind, eid)
        result[kind] = {
            key: {
                "min": value.reshape(-1, 27).min(0),
                "max": value.reshape(-1, 27).max(0),
                "pairs": len(state),
            }
            for key, value in [("state", state), ("actions", physical)]
        }
    return result


def aggregate_fit(records, modality):
    values = [
        r[kind][modality]
        for r in records
        if r["split"] == "train"
        for kind in ("clean", "augmented")
    ]
    return np.min([v["min"] for v in values], axis=0), np.max(
        [v["max"] for v in values], axis=0
    )


def fit_profile(records, original, root, margin):
    action_min, action_max = aggregate_fit(records, "actions")
    state_min, state_max = aggregate_fit(records, "state")
    hand_min = np.minimum(action_min[7:], state_min[7:])
    hand_max = np.maximum(action_max[7:], state_max[7:])
    fitted = {}
    for key, source_low, source_high in [
        ("actions", action_min, action_max),
        ("state", state_min, state_max),
    ]:
        low = np.r_[source_low[:7], hand_min]
        high = np.r_[source_high[:7], hand_max]
        center = (high + low) / 2
        width = np.maximum(high - low, 0.01)
        fitted[key] = {
            "low": center - width * (0.5 + margin),
            "high": center + width * (0.5 + margin),
            "train_min": low,
            "train_max": high,
        }
    return {
        "schema": "touchguide_cpm_normalization_v1",
        "data_root": str(root),
        "fit_policy": {
            "split": "train_only",
            "episodes": sum(r["split"] == "train" for r in records),
            "pool": ["all_clean", "all_10_recoveries_per_origin"],
            "per_side_margin_fraction_of_full_span": margin,
            "minimum_span_rad": 0.01,
            "shared_hand_state_action_bounds": True,
            "fixed_for_ratios": ["1:1", "1:5", "1:10"],
            "clipping": False,
        },
        **fitted,
        "original_pi05_stats": original,
        "original_statistics_sha256": hashlib.sha256(
            (root / "clean/meta/action_norm_stats.json").read_bytes()
        ).hexdigest(),
        "action_convention": "arm delta7 + hand absolute20",
        "state_convention": "arm absolute7 + hand absolute20",
        "noise_contract": "Generate noise in original pi05 coordinates, THEN apply the differentiable affine pi05_to_cpm transform; same order during training and inference.",
        "status": "validated_profile_not_yet_wired_into_legacy_trainers",
        "tactile": {
            "policy": "retain_current_train_RMS_signed_asinh_reference32_no_clip"
        },
        "vision": {"policy": "retain_RGB_uint8_to_minus1_plus1_then_ImageNet_once"},
    }


def numeric_summary(x):
    x = x.reshape(-1, 27).astype(np.float64)
    return {
        "count_per_dimension": len(x),
        "min": x.min(0),
        "max": x.max(0),
        "sum": x.sum(0),
        "sum_sq": np.square(x).sum(0),
        "outside": (np.abs(x) > 1).sum(0),
    }


def check_episode(task):
    root, eid, split, original, profile = task
    rng = np.random.default_rng(np.random.SeedSequence([42, eid, 205]))
    result = {"eid": eid, "split": split, "data": {}}
    sample = {}
    for kind in ("clean", "augmented"):
        state, physical = read(root, kind, eid)
        result["data"][kind] = {"pairs": len(state)}
        if kind == "clean":
            chosen = np.sort(rng.choice(len(state), min(32, len(state)), replace=False))
            ids = chosen
        else:
            ids = (chosen[:, None] * 10 + np.arange(10)).ravel()
        sample[kind] = {}
        with h5py.File(root / kind / "episodes" / f"episode_{eid:06d}.h5") as source:
            stored_old_actions = source["actions"][:]
        for modality, raw in [("state", state), ("actions", physical)]:
            qlow, qhigh = [np.asarray(original[modality][k]) for k in ("q01", "q99")]
            low, high = [np.asarray(profile[modality][k]) for k in ("low", "high")]
            # Preserve the actual legacy FP32 arithmetic when counting strict
            # threshold crossings; FP64 recomputation changes boundary rounding.
            old = (
                stored_old_actions
                if modality == "actions"
                else (raw.astype(np.float32) - qlow.astype(np.float32))
                / (qhigh.astype(np.float32) - qlow.astype(np.float32) + 1e-6)
                * 2
                - 1
            )
            new = 2 * (raw - low) / (high - low) - 1
            assert np.isfinite(old).all()
            assert np.isfinite(new).all()
            result["data"][kind][modality] = {
                "old": numeric_summary(old),
                "new": numeric_summary(new),
            }
            sample[kind][modality] = {
                "old": old[ids].astype(np.float32),
                "new": new[ids].astype(np.float32),
            }
    return result, sample


def aggregate(records, kind, modality, version, split):
    values = [
        r["data"][kind][modality][version]
        for r in records
        if split == "all" or r["split"] == split
    ]
    n = sum(v["count_per_dimension"] for v in values)
    mean = sum(v["sum"] for v in values) / n
    return {
        "count_per_dimension": n,
        "min": np.min([v["min"] for v in values], axis=0),
        "max": np.max([v["max"] for v in values], axis=0),
        "fraction_abs_gt_1": sum(v["outside"] for v in values) / n,
        "mean": mean,
        "std": np.sqrt(np.maximum(0, sum(v["sum_sq"] for v in values) / n - mean**2)),
    }


def check_camera(task):
    from openpi_client import image_tools  # noqa: PLC0415

    from openpi.touchguide.data import _decode_video_frame  # noqa: PLC0415

    root, eid = task
    out = {"episode": eid, "views": []}
    with h5py.File(root / "clean/episodes" / f"episode_{eid:06d}.h5") as f:
        anchors = f["frame_index"][:]
        # Start/middle/end of contact anchors, two input cameras, all episodes.
        for attr in ("head_video", "right_wrist_video"):
            path = f.attrs[attr]
            if isinstance(path, bytes):
                path = path.decode()
            for row in sorted({0, len(anchors) // 2, len(anchors) - 1}):
                image = _decode_video_frame(
                    str(path), int(anchors[row]), float(f.attrs["fps"])
                )
                assert image.dtype == np.uint8
                assert image.shape == (240, 320, 3)
                resized = image_tools.resize_with_pad(image, 224, 224)
                train = resized.transpose(2, 0, 1).astype(np.float32) / np.float32(
                    127.5
                ) - np.float32(1)
                inference = (
                    resized.astype(np.float32) / np.float32(127.5) - np.float32(1)
                ).transpose(2, 0, 1)
                expected = (
                    resized.transpose(2, 0, 1).astype(np.float32) / np.float32(255)
                    - np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]
                ) / np.array([0.229, 0.224, 0.225], np.float32)[:, None, None]
                mapped = (
                    (train + 1) * 0.5
                    - np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]
                ) / np.array([0.229, 0.224, 0.225], np.float32)[:, None, None]
                assert np.array_equal(train, inference)
                error = float(np.max(np.abs(mapped - expected)))
                assert error < 1e-6
                out["views"].append(
                    {
                        "view": attr,
                        "row": row,
                        "frame": int(anchors[row]),
                        "reader_min": float(train.min()),
                        "reader_max": float(train.max()),
                        "dino_min": float(mapped.min()),
                        "dino_max": float(mapped.max()),
                        "imagenet_mapping_error": error,
                    }
                )
    return out


def check_tensor_paths(root, profile):
    from openpi.touchguide.model import FrozenDinov2  # noqa: PLC0415
    from openpi.touchguide.tactile_dino import TactileDinoPreprocessor  # noqa: PLC0415

    torch.set_num_threads(4)
    normalizer = CPMNormalization(profile)
    report = {}
    with h5py.File(root / "augmented/episodes/episode_000007.h5") as f:
        ids = [0, len(f["state_raw"]) // 2, len(f["state_raw"]) - 1]
        state = torch.tensor(np.stack([f["state_raw"][i] for i in ids]))
        absolute = torch.tensor(np.stack([f["actions_absolute"][i] for i in ids]))
        old_action = torch.tensor(np.stack([f["actions"][i] for i in ids]))
        touch = torch.tensor(np.stack([f["tactile_map_raw"][i] for i in ids]))
    physical = absolute.clone()
    physical[..., :7] -= state[:, None, :7]
    direct = normalizer.normalize_physical_actions(physical)
    converted = normalizer.pi05_actions(old_action)
    torch.testing.assert_close(direct, converted, rtol=0, atol=2e-6)
    torch.testing.assert_close(
        normalizer.action_targets(converted, state), absolute, rtol=0, atol=1e-6
    )
    old_low = torch.tensor(profile["original_pi05_stats"]["state"]["q01"])
    old_high = torch.tensor(profile["original_pi05_stats"]["state"]["q99"])
    old_state = 2 * (state - old_low) / (old_high - old_low + 1e-6) - 1
    torch.testing.assert_close(
        normalizer.pi05_state(old_state),
        normalizer.normalize_state(state),
        rtol=0,
        atol=2e-6,
    )
    x = torch.cat(
        [old_action, torch.zeros(len(old_action), 30, 5)], -1
    ).requires_grad_()
    converted = normalizer.pi05_actions(x)
    grad = torch.autograd.grad(converted.sum(), x)[0]
    expected = normalizer.actions_original_width / normalizer.actions_width
    torch.testing.assert_close(
        grad[..., :27], expected.expand_as(grad[..., :27]), rtol=0, atol=1e-6
    )
    assert torch.count_nonzero(grad[..., 27:]) == 0
    # A_t in original coordinates is the sole noise source. Its converted
    # noise is an affine Gaussian, not a new N(0,I) in CPM coordinates.
    eps = torch.randn(old_action.shape, generator=torch.Generator().manual_seed(42))
    for t in [0.0, 0.1, 0.5, 0.9, 1.0]:
        left = normalizer.pi05_actions((1 - t) * old_action + t * eps)
        right = (1 - t) * normalizer.pi05_actions(
            old_action
        ) + t * normalizer.pi05_actions(eps)
        torch.testing.assert_close(left, right, rtol=0, atol=2e-6)
    scale = json.loads((root / "clean/meta/tactile_scale.json").read_text())["rms"]
    tactile = TactileDinoPreprocessor(scale)
    separate = torch.cat([tactile(touch[i : i + 1]) for i in range(len(touch))])
    torch.testing.assert_close(tactile(touch), separate, rtol=0, atol=0)
    assert torch.isfinite(separate).all()
    zero = tactile.encode_image(torch.zeros_like(touch[:1]))
    assert torch.all(zero == 0.5)
    rgb = (
        torch.rand(2, 3, 224, 224, generator=torch.Generator().manual_seed(42)) * 2 - 1
    )
    buffers = SimpleNamespace(
        rgb_mean=tactile.imagenet_mean, rgb_std=tactile.imagenet_std
    )
    actual = FrozenDinov2.preprocess(buffers, rgb)
    expected_rgb = ((rgb + 1) * 0.5 - tactile.imagenet_mean) / tactile.imagenet_std
    torch.testing.assert_close(actual, expected_rgb, rtol=0, atol=0)
    report.update(
        status="passed",
        action_conversion_max_error=float(
            (direct - normalizer.pi05_actions(old_action)).abs().max()
        ),
        roundtrip_max_error=float(
            (normalizer.action_targets(direct, state) - absolute).abs().max()
        ),
        gradient_min=float(grad[..., :27].min()),
        gradient_max=float(grad[..., :27].max()),
        padded_gradient_exact_zero=True,
        noise_conversion_commutes_with_interpolation=True,
        current_tactile_train_inference_function_identical=True,
        tactile_batch_independence_exact=True,
        visual_model_preprocessing_verified=True,
        gpu_bf16_tested=False,
    )
    return report


def visualize(out, report):
    import matplotlib as mpl  # noqa: PLC0415

    mpl.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, axes = plt.subplots(1, 2, figsize=(16, 10))
    y = np.arange(27)
    colors = {"clean": "#2563eb", "augmented": "#ea580c"}
    for ax, modality in zip(axes, ("actions", "state"), strict=True):
        ax.axvspan(-1, 1, color="#dcfce7", alpha=0.6)
        for k, offset in [("clean", -0.2), ("augmented", 0.2)]:
            old = report["exact"][k][modality]["old"]["all"]
            new = report["exact"][k][modality]["new"]["all"]
            ax.hlines(
                y + offset,
                old["min"],
                old["max"],
                colors[k],
                alpha=0.25,
                linewidth=3,
                label=f"{k}: old range",
            )
            ax.hlines(
                y + offset,
                new["min"],
                new["max"],
                colors[k],
                linewidth=3,
                label=f"{k}: new range",
            )
        ax.set_yticks(y, NAMES, fontsize=9)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.2)
        for v in [-1, 1]:
            ax.axvline(v, color="#16a34a", linestyle="--", linewidth=1)
        ax.set_xlabel("Normalized value; faint = old, solid = new")
        ax.set_title(modality + ": exact ranges of all 200 episodes")
        ax.legend(loc="lower left", fontsize=8)
    fig.suptitle(
        "CPM-only ranges fitted on TRAIN ONLY; shared hand bounds; 10% margin per side"
    )
    fig.tight_layout()
    fig.savefig(out / "normalization_comparison.png", dpi=150)
    plt.close(fig)
    html = PAGE.replace(
        "__DATA__",
        json.dumps(report, default=encode, ensure_ascii=False).replace("</", "<\\/"),
    )
    (out / "index.html").write_text(html)


PAGE = r"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>CPM 四模态归一化检查</title><meta name="viewport" content="width=device-width, initial-scale=1"><style>body{font:15px system-ui;background:#f1f5f9;color:#172b45;margin:0}main{max-width:1300px;margin:24px auto;padding:0 20px}section{background:white;border:1px solid #dbe3ed;border-radius:12px;margin-bottom:18px;padding:20px}p{line-height:1.7}h1{font-size:25px}h2{font-size:19px}table{width:100%;border-collapse:collapse}th,td{padding:9px;border-bottom:1px solid #e2e8f0;text-align:right}th:first-child,td:first-child{text-align:left}.muted{color:#64748b;font-size:13px}select{padding:8px;margin:6px 16px 12px 0}svg,img{width:100%;height:auto}a{color:#2563eb}.scroll{overflow:auto}</style><main><h1>CPM 四模态归一化检查</h1><section><h2>新动作 / state 范围</h2><p>只用训练集的全部干净与恢复数据确定边界，每侧增加原始跨度的 10% 余量。手指 state 与动作共用物理范围；手臂 state（绝对位置）与动作（增量）分别确定范围。三种训练比例固定使用同一组参数，不裁剪。</p><p><b>新参数已生成并独立验证，但没有覆盖旧 H5、旧模型或旧训练器配置。</b>旧模型仍使用原来的参数。新 CPM 接入此配置后，才能实际使用新坐标。</p><div id="summary"></div><p class="muted">范围和比例是全量精确统计。比例按动作 / state 的数值计算，不是样本比例。验证集只用于检查，没有参与范围拟合。未加 flow 噪声的动作落在参考范围内，不代表带噪动作也必须在范围内。</p></section><section><h2>逐关节：新旧范围和超范围比例</h2><label>模态 <select id="modality"><option value="actions">动作</option><option value="state">state</option></select></label><label>数据划分 <select id="split"><option value="all">全量</option><option value="train">训练集</option><option value="val">验证集</option></select></label><div class="scroll" id="details"></div><p>蓝色：干净；橙色：恢复 / 扰动。淡色线：旧范围；实色线：新范围。绿色：[-1, 1]。</p><img src="normalization_comparison.png" alt="动作与state新旧归一化范围全量对比"><a href="normalization_comparison.png" target="_blank">打开图片</a></section><section><h2>视觉</h2><p id="vision"></p><p>训练与推理均使用 RGB、相同的 224×224 缩放与填充、uint8→[-1,1]，随后还原至 [0,1] 并做一次 ImageNet 标准化。ImageNet 标准化后的负值或大于 1 的值是正常的。这里验证的是转换一致性，不能证明所有相机图像分布完全一致。</p><h2>触觉</h2><p id="touch"></p><p>保持固定训练 RMS、带符号 asinh、reference=32、不截断、不逐帧归一化。零力对应 0.5，切向保留符号；之后统一做一次 ImageNet 标准化。极少数力值映射后超 [0,1]，程序没有截断这些值。</p><h2>与 π0.5 的衔接</h2><p>π0.5 的权重和归一化不变。训练与推理均在 π0.5 坐标生成 / 接收带噪动作后，使用可微仿射变换转到 CPM 坐标。动作梯度按链式法则返回 π0.5 坐标；不能在新坐标中重新使用独立 N(0,I) 噪声替代这一转换。</p><p id="checks" class="muted"></p></section></main><script>
const D=__DATA__,name={clean:'干净',augmented:'增强'},fmt=x=>Number(x).toFixed(4),pct=x=>(100*x).toFixed(4)+'%';
let s='<table><tr><th>模态 / 数据</th><th>旧范围</th><th>新范围</th><th>旧手指超区间</th><th>新手指超区间</th></tr>';
for(const m of ['actions','state'])for(const k of ['clean','augmented']){let a=D.exact[k][m].old.all,b=D.exact[k][m].new.all,avg=x=>x.slice(7).reduce((p,q)=>p+q,0)/20;s+=`<tr><td>${m} / ${name[k]}</td><td>[${fmt(Math.min(...a.min))}, ${fmt(Math.max(...a.max))}]</td><td>[${fmt(Math.min(...b.min))}, ${fmt(Math.max(...b.max))}]</td><td>${pct(avg(a.fraction_abs_gt_1))}</td><td>${pct(avg(b.fraction_abs_gt_1))}</td></tr>`}document.getElementById('summary').innerHTML=s+'</table>';
function update(){const m=document.getElementById('modality').value,split=document.getElementById('split').value;let t='<table><tr><th>关节 / 数据</th><th>旧最小</th><th>旧最大</th><th>旧超区间</th><th>新最小</th><th>新最大</th><th>新超区间</th><th>新物理下界 rad</th><th>新物理上界 rad</th></tr>';for(let j=0;j<27;j++)for(const k of ['clean','augmented']){let a=D.exact[k][m].old[split],b=D.exact[k][m].new[split];t+=`<tr><td>${D.joint_names[j]} / ${name[k]}</td><td>${fmt(a.min[j])}</td><td>${fmt(a.max[j])}</td><td>${pct(a.fraction_abs_gt_1[j])}</td><td>${fmt(b.min[j])}</td><td>${fmt(b.max[j])}</td><td>${pct(b.fraction_abs_gt_1[j])}</td><td>${fmt(D.profile[m].low[j])}</td><td>${fmt(D.profile[m].high[j])}</td></tr>`}document.getElementById('details').innerHTML=t+'</table>'}
for(const id of ['modality','split'])document.getElementById(id).addEventListener('change',update);update();
document.getElementById('vision').textContent=`实测 ${D.vision.decoded_frames} 帧，覆盖 ${D.vision.episodes} 条 episode、两路输入相机。训练 / 推理数值转换一致，ImageNet 标准化最大误差 ${D.vision.max_mapping_error.toExponential(2)}。`;
const f=D.tactile.previous_full_exact.augmented.touch.mapped_pixel_outside_0_1_fraction;document.getElementById('touch').textContent=`增强触觉映射后超 [0,1] 的比例：法向 ${pct(f[0])}，切向0 ${pct(f[1])}，切向1 ${pct(f[2])}。同一函数用于训练与推理，实测批次组合不改变表示。`;
document.getElementById('checks').textContent=`坐标转换、反归一化、梯度与噪声映射一致性检查：${D.tensor_paths.status}。本轮未做 GPU BF16 或真实执行效果验证。`;
</script></html>"""


def run(args):
    root = args.data_root.resolve()
    out = args.output_dir or root / "validation/modality_normalization_v1"
    out.mkdir(parents=True, exist_ok=True)
    original = json.loads((root / "clean/meta/action_norm_stats.json").read_text())[
        "norm_stats"
    ]
    manifest = json.loads((root / "clean/meta/manifest.json").read_text())
    tasks = [
        (root, eid, split)
        for split in ("train", "val")
        for eid in manifest["splits"][split]["episode_indices"]
    ]
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(fit_episode, task) for task in tasks]):
            records.append(future.result())
            if len(records) % 20 == 0:
                print(f"Fit scan {len(records)}/{len(tasks)} episodes", flush=True)
    profile = fit_profile(records, original, root, args.margin)
    dump(out / "cpm_normalization_v1.json", profile)
    checked, sampled = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed(
            [pool.submit(check_episode, (*task, original, profile)) for task in tasks]
        ):
            record, sample = future.result()
            checked.append(record)
            sampled.append(sample)
            if len(checked) % 20 == 0:
                print(
                    f"Validation scan {len(checked)}/{len(tasks)} episodes", flush=True
                )
    report = {
        "status": "passed",
        "data_root": str(root),
        "joint_names": NAMES,
        "profile": profile,
        "exact": {
            kind: {
                modality: {
                    version: {
                        split: aggregate(checked, kind, modality, version, split)
                        for split in ("train", "val", "all")
                    }
                    for version in ("old", "new")
                }
                for modality in ("actions", "state")
            }
            for kind in ("clean", "augmented")
        },
    }
    report["sampled_quantiles"] = {
        kind: {
            modality: {
                version: np.percentile(
                    np.concatenate(
                        [s[kind][modality][version].reshape(-1, 27) for s in sampled]
                    ),
                    [0, 1, 50, 95, 99, 99.9, 100],
                    axis=0,
                )
                for version in ("old", "new")
            }
            for modality in ("state", "actions")
        }
        for kind in ("clean", "augmented")
    }
    report["tensor_paths"] = check_tensor_paths(root, profile)
    cameras = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed(
            [pool.submit(check_camera, (root, t[1])) for t in tasks]
        ):
            cameras.append(future.result())
            if len(cameras) % 20 == 0:
                print(f"Visual audit {len(cameras)}/{len(tasks)} episodes", flush=True)
    report["vision"] = {
        "status": "passed",
        "episodes": len(cameras),
        "decoded_frames": sum(len(c["views"]) for c in cameras),
        "sampling": "Contact anchor start/middle/end for both views in every episode; not all video frames",
        "max_mapping_error": max(
            v["imagenet_mapping_error"] for c in cameras for v in c["views"]
        ),
        "reader_range": [
            min(v["reader_min"] for c in cameras for v in c["views"]),
            max(v["reader_max"] for c in cameras for v in c["views"]),
        ],
        "dino_range": [
            min(v["dino_min"] for c in cameras for v in c["views"]),
            max(v["dino_max"] for c in cameras for v in c["views"]),
        ],
        "per_episode": cameras,
    }
    previous = json.loads(
        (root / "validation/asinh_action_audit/report.json").read_text()
    )
    report["tactile"] = {
        "status": "retain_current_parameters",
        "previous_full_exact": {
            kind: previous["exact"][kind]["all"] for kind in ("clean", "augmented")
        },
        "source_report_sha256": hashlib.sha256(
            (root / "validation/asinh_action_audit/report.json").read_bytes()
        ).hexdigest(),
        "scope": "Reuse previous full 200-episode tactile scan; current preprocessing exercised on real tensors and both caller paths inspected; no GPU BF16 test",
    }
    report["new_train_and_val_outside_counts"] = {
        kind: {
            m: {
                s: round(
                    float(
                        np.sum(report["exact"][kind][m]["new"][s]["fraction_abs_gt_1"])
                    )
                    * report["exact"][kind][m]["new"][s]["count_per_dimension"]
                )
                for s in ("train", "val")
            }
            for m in ("actions", "state")
        }
        for kind in ("clean", "augmented")
    }
    dump(out / "report.json", report)
    dump(out / "vision_review.json", report["vision"])
    visualize(out, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "output_dir": str(out),
                "new_outside_counts": report["new_train_and_val_outside_counts"],
                "tensor_paths": report["tensor_paths"],
                "visual_frames": report["vision"]["decoded_frames"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--margin", type=float, default=0.1)
    args = parser.parse_args()
    if not math.isfinite(args.margin) or args.margin < 0:
        raise ValueError("margin must be finite and nonnegative")
    run(args)
