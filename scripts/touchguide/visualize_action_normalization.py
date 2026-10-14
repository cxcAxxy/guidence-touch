"""Visualize existing exact normalization audit and three explicitly selected pairs."""

# ruff: noqa: RUF001
# Chinese UI text intentionally uses Chinese punctuation.
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

DEFAULT_ROOT = Path(
    "/path/to/datasets/touchguidence/augmentation/poker_fingertip_recovery_rules_v1_random10"
)
NAMES = [
    *(f"arm_joint{i}" for i in range(1, 8)),
    *(
        f"{f}_joint{i}"
        for f in ("thumb", "index", "middle", "ring", "pinky")
        for i in ((1, 2, 3, 5) if f == "thumb" else (1, 2, 3, 4))
    ),
]


def run(root, output, episode):
    import matplotlib as mpl  # noqa: PLC0415

    mpl.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    audit_path = root / "validation/asinh_action_audit/report.json"
    audit = json.loads(audit_path.read_text())
    assert audit["status"] == "passed"
    assert Path(audit["data_root"]).resolve() == root.resolve()
    stats = json.loads((root / "clean/meta/action_norm_stats.json").read_text())[
        "norm_stats"
    ]["actions"]
    low, high = [np.asarray(stats[k], dtype=np.float32) for k in ("q01", "q99")]
    exact = {k: audit["exact"][k]["all"] for k in ("clean", "augmented")}
    cases = []
    with (
        h5py.File(root / f"augmented/episodes/episode_{episode:06d}.h5", "r") as aug,
        h5py.File(root / f"clean/episodes/episode_{episode:06d}.h5", "r") as clean,
    ):
        actions = aug["actions"][:]
        # Deliberately choose two tail examples and one median-excursion example.
        peak = np.max(np.abs(actions[:, :, 7:]), axis=(1, 2))
        chosen = [
            int(np.argmax(np.max(actions[:, :, 19], axis=1))),
            int(np.argmin(np.min(actions[:, :, 15], axis=1))),
            int(np.argsort(peak)[len(peak) // 2]),
        ]
        for row, title in zip(
            chosen,
            [
                "无名指 joint1 正向尾部示例",
                "中指 joint1 负向尾部示例",
                "本 episode 最大幅值居中示例",
            ],
            strict=True,
        ):
            source = int(aug["source_clean_row"][row])
            record = {
                "title": title,
                "episode": episode,
                "augmented_row": row,
                "source_clean_row": source,
                "frame_index": int(aug["frame_index"][row]),
                "variant_index": int(aug["variant_index"][row]),
            }
            for kind, f, r in [("clean", clean, source), ("augmented", aug, row)]:
                state = f["state_raw"][r]
                absolute = f["actions_absolute"][r]
                physical = absolute.copy()
                physical[:, :7] -= state[None, :7]
                normalized = f["actions"][r]
                expected = (physical - low) / (high - low + 1e-6) * 2 - 1
                assert np.max(np.abs(expected - normalized)) < 1e-5
                record[kind] = {
                    "normalized": normalized.tolist(),
                    "physical": physical.tolist(),
                    "state": state.tolist(),
                }
            assert np.array_equal(
                aug["actions_absolute"][row, -1], clean["state_raw"][source]
            )
            cases.append(record)
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "data_root": str(root),
        "audit_path": str(audit_path),
        "names": NAMES,
        "exact": exact,
        "q01": low.tolist(),
        "q99": high.tolist(),
        "cases": cases,
        "sampled_quantiles": {
            k: audit["sampled"][k]["normalized_action_per_dimension"] for k in exact
        },
        "scope": "Exact ranges and fractions from all 200 episodes; three selected examples from one episode; no flow noise and no clipping.",
    }
    (output / "data.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    colors = {"clean": "#2563eb", "augmented": "#ea580c"}
    fig, axes = plt.subplots(
        1, 2, figsize=(15, 10), gridspec_kw={"width_ratios": [1.3, 1]}
    )
    y = np.arange(27)
    axes[0].axvspan(
        -1, 1, color="#dcfce7", alpha=0.65, label="Reference interval [-1, 1]"
    )
    for kind, offset, label in [
        ("clean", -0.17, "Clean task actions"),
        ("augmented", 0.17, "Recovery actions"),
    ]:
        d = exact[kind]["actions"]
        axes[0].hlines(
            y + offset, d["min"], d["max"], color=colors[kind], linewidth=3, label=label
        )
        axes[0].scatter(d["min"], y + offset, color=colors[kind], s=10)
        axes[0].scatter(d["max"], y + offset, color=colors[kind], s=10)
        axes[1].barh(
            y + offset,
            np.array(d["fraction_abs_gt_1"]) * 100,
            height=0.32,
            color=colors[kind],
            label=label,
        )
    for ax in axes:
        ax.set_yticks(y, NAMES, fontsize=9)
        ax.invert_yaxis()
        ax.axhline(6.5, color="#64748b", linestyle=":", linewidth=1)
        ax.grid(axis="x", alpha=0.2)
    axes[0].axvline(-1, color="#16a34a", linestyle="--", linewidth=1)
    axes[0].axvline(1, color="#16a34a", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Normalized action value (dimensionless)")
    axes[0].set_title("Exact min/max: all action values, all 200 episodes")
    axes[0].legend(loc="lower left", fontsize=8)
    axes[1].set_xlabel("Action values outside [-1, 1] (%)")
    axes[1].set_title("Exact fraction outside reference interval")
    axes[1].legend(loc="lower right", fontsize=8)
    fig.suptitle(
        "Action normalization audit: shared q01/q99 transform, no clipping", fontsize=14
    )
    fig.tight_layout()
    fig.savefig(output / "action_normalization.png", dpi=160)
    plt.close(fig)
    html = HTML.replace(
        "__PAYLOAD__", json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    )
    (output / "index.html").write_text(html)
    review = {
        "status": "passed",
        "exact_audit_episodes": audit["episodes"],
        "example_episode": episode,
        "selected_examples": len(cases),
        "checks": [
            "Recomputed normalization equals stored actions within 1e-5",
            "Recovery final target equals source clean current state exactly",
            "All summary values copied from existing full exact audit",
        ],
        "artifacts": ["index.html", "action_normalization.png", "data.json"],
    }
    (output / "review.json").write_text(
        json.dumps(review, ensure_ascii=False, indent=2)
    )
    print(
        json.dumps({"output_dir": str(output), **review}, ensure_ascii=False, indent=2)
    )


HTML = r"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>动作归一化验证</title>
<style>body{margin:0;background:#f1f5f9;color:#172b45;font:15px system-ui,sans-serif}main{max-width:1250px;margin:24px auto;padding:0 18px}.panel{background:white;border:1px solid #dbe3ed;border-radius:12px;padding:20px;margin-bottom:18px}h1{font-size:25px}h2{font-size:19px;margin-top:0}p{line-height:1.7}.blue{color:#2563eb}.orange{color:#ea580c}.muted{color:#64748b;font-size:13px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:9px;border-bottom:1px solid #e2e8f0;text-align:right}td:first-child,th:first-child{text-align:left}select{padding:8px;max-width:100%;margin:5px 14px 8px 0}svg{width:100%;height:auto}.scroll{overflow:auto}.legend{display:flex;gap:22px;margin:10px 0}.tag{display:inline-block;padding:3px 7px;background:#fff1e6;color:#c2410c;border-radius:5px}a{color:#2563eb}</style>
<main><h1>干净动作与恢复动作：归一化验证</h1><section class="panel"><p><b>两种动作都会超出 [-1, 1]。</b> 当前使用同一套 π0.5 的 q01 / q99 线性映射，不截断。绿色区间表示原始动作分位数参考区间，<b>不表示机械关节限位</b>。</p><p>归一化公式：<code>a_norm = 2 × (a − q01) / (q99 − q01 + 10⁻⁶) − 1</code>。手臂 7 维的 a 是相对当前 state 的关节增量；手指 20 维的 a 是绝对关节目标。a 超过原始 q01 / q99 时，归一化结果可以超过 [-1, 1]。</p><div id="summary"></div><p class="muted">比例按动作数值计算：每条数据 30 步 × 对应组的关节维度；不是“有多少条样本超范围”。范围与比例来自全量 200 条 episode。统计对象是未加 flow 噪声的 action，不是 A_t。</p></section>
<section class="panel"><h2>27 个关节的全量范围</h2><div class="legend"><span class="blue">━━ 干净任务动作</span><span class="orange">━━ 恢复动作</span><span>绿色背景：[-1, 1]</span></div><div id="ranges"></div><p class="muted">横线两端是全量最小值、最大值，不能代表大部分数据的位置。下方表格给出该维度的超范围比例与抽样中位数 / 99% 分位数。</p><a href="action_normalization.png" target="_blank">打开可导出的全量统计图（PNG）</a></section>
<section class="panel"><h2>选择关节查看详细统计与 30 步动作</h2><label>关节 <select id="joint"></select></label><div id="detail"></div><p class="muted">分位数来自抽样（6,400 条干净配对、64,000 条增强配对）；范围与超范围比例仍为全量精确统计。</p><label>具体示例 <select id="example"></select></label><label>数值 <select id="units"><option value="normalized">归一化值</option><option value="physical">物理关节量（rad）</option></select></label><div id="metadata" class="muted"></div><div id="trajectory"></div><p class="muted">这三条是 episode 7 中特意选出的两个尾部示例和一个最大幅值居中示例，不代表全数据分布。蓝线是同一起点的原始 30 步任务动作；橙线是从扰动 state 回到干净起点 state 的恢复动作，因此两条曲线的终点不要求相同。</p><p class="muted">归一化区间越界不等于违反关节限位，也不等于归一化公式有误。恢复数据的手指扰动上限为 0.15 rad；是否能改善真实执行，需要另外验证。</p></section></main>
<script>
const D=__PAYLOAD__;
const C={clean:'#2563eb',augmented:'#ea580c'}, L={clean:'干净动作',augmented:'恢复动作'};
const fmt=x=>Number(x).toFixed(4), pct=x=>(100*x).toFixed(4)+'%';
function el(tag, attrs, text=''){return `<${tag} ${attrs}>${text}</${tag}>`}
function line(x1,y1,x2,y2,color,width=2,dash=''){return el('line',`x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="${width}" ${dash?'stroke-dasharray="'+dash+'"':''}`)}
function text(x,y,t,fill='#475569',anchor='start'){return el('text',`x="${x}" y="${y}" fill="${fill}" text-anchor="${anchor}" font-size="12"`,t)}
function svg(parts,h){return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1100 ${h}">${parts.join('')}</svg>`}
function summary(){let s='<table><tr><th>动作</th><th>配对数</th><th>范围（全 27 维）</th><th>手臂超 [-1,1]</th><th>手指超 [-1,1]</th><th>手指超 [-3,3]</th></tr>';for(const k of ['clean','augmented']){const d=D.exact[k],a=d.actions,avg=(f,start,end)=>a[f].slice(start,end).reduce((x,y)=>x+y,0)/(end-start);s+=`<tr><td style="color:${C[k]}">${L[k]}</td><td>${d.pairs.toLocaleString()}</td><td>[${fmt(Math.min(...a.min))}, ${fmt(Math.max(...a.max))}]</td><td>${pct(avg('fraction_abs_gt_1',0,7))}</td><td>${pct(avg('fraction_abs_gt_1',7,27))}</td><td>${pct(avg('fraction_abs_gt_3',7,27))}</td></tr>`}document.getElementById('summary').innerHTML=s+'</table>'}
function ranges(){const p=[],x=v=>200+(v+5)/10*860;p.push(el('rect',`x="${x(-1)}" y="22" width="${x(1)-x(-1)}" height="846" fill="#dcfce7"`));for(let v=-5;v<=5;v++){p.push(line(x(v),20,x(v),867,'#e2e8f0',1),text(x(v),888,v,'#475569','middle'))}for(let j=0;j<27;j++){const y=38+j*31;p.push(text(12,y+4,`${j}: ${D.names[j]}`));for(const k of ['clean','augmented']){const a=D.exact[k].actions,yy=y+(k==='clean'?-5:5);p.push(line(x(a.min[j]),yy,x(a.max[j]),yy,C[k],3))}}p.push(line(x(-1),20,x(-1),867,'#16a34a',1,'5 4'),line(x(1),20,x(1),867,'#16a34a',1,'5 4'));document.getElementById('ranges').innerHTML=svg(p,910)}
function update(){const j=Number(document.getElementById('joint').value),c=D.cases[Number(document.getElementById('example').value)],unit=document.getElementById('units').value;
 let t='<table><tr><th>动作</th><th>最小</th><th>最大</th><th>超 [-1,1]</th><th>中位数（抽样）</th><th>99% 分位数（抽样）</th></tr>';for(const k of ['clean','augmented']){const a=D.exact[k].actions,q=D.sampled_quantiles[k].values;t+=`<tr><td style="color:${C[k]}">${L[k]}</td><td>${fmt(a.min[j])}</td><td>${fmt(a.max[j])}</td><td>${pct(a.fraction_abs_gt_1[j])}</td><td>${fmt(q[2][j])}</td><td>${fmt(q[4][j])}</td></tr>`}document.getElementById('detail').innerHTML=t+'</table>';
 document.getElementById('metadata').textContent=`episode ${c.episode} · 原始帧 ${c.frame_index} · 干净起点行 ${c.source_clean_row} · 增强行 ${c.augmented_row} · variant ${c.variant_index} · q01=${fmt(D.q01[j])} rad，q99=${fmt(D.q99[j])} rad`;
 const ys={},all=[];for(const k of ['clean','augmented']){ys[k]=c[k][unit].map(row=>row[j]);all.push(...ys[k])}if(unit==='normalized')all.push(-1,1);let lo=Math.min(...all),hi=Math.max(...all),pad=Math.max((hi-lo)*.12,.005);lo-=pad;hi+=pad;
 const x=s=>85+s/29*970,y=v=>335-(v-lo)/(hi-lo)*290,p=[];
 if(unit==='normalized')p.push(el('rect',`x="85" y="${y(1)}" width="970" height="${y(-1)-y(1)}" fill="#dcfce7" opacity=".6"`));
 for(let i=0;i<=5;i++){const v=lo+(hi-lo)*i/5;p.push(line(85,y(v),1055,y(v),'#e2e8f0',1),text(75,y(v)+4,fmt(v),'#475569','end'))}for(const s of [0,4,9,14,19,24,29])p.push(text(x(s),355,s+1,'#475569','middle'));
 if(unit==='normalized'){for(const v of [-1,1])p.push(line(85,y(v),1055,y(v),'#16a34a',1,'6 4'))}
 for(const k of ['clean','augmented'])p.push(el('polyline',`points="${ys[k].map((v,s)=>`${x(s)},${y(v)}`).join(' ')}" fill="none" stroke="${C[k]}" stroke-width="3"`));
 p.push(text(85,20,unit==='normalized'?'归一化动作（无单位）':j<7?'手臂目标相对各自当前 state 的增量（rad）':'手指绝对关节目标（rad）'),text(1055,382,'动作步数（1–30）','#475569','end'));
 document.getElementById('trajectory').innerHTML=svg(p,400);
}
D.names.forEach((n,j)=>document.getElementById('joint').add(new Option(`${j}: ${n}`,j)));D.cases.forEach((c,i)=>document.getElementById('example').add(new Option(c.title,i)));document.getElementById('joint').value=19;for(const id of ['joint','example','units'])document.getElementById(id).addEventListener('change',update);summary();ranges();update();
</script></html>"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--episode", type=int, default=7)
    args = parser.parse_args()
    run(
        args.data_root,
        args.output_dir or args.data_root / "visualization/action_normalization",
        args.episode,
    )
