# tac_forward_guidance v1

复用原始视觉—触觉 CPM 和 20k π0.5 checkpoint，不重新训练、不改动现有 TouchGuide。
方法标签：`pi05+tac_forward_guidance`；对照：`pi05`。

## 算法与边界

在引导步骤计算 `v = pi05.denoise_step(x,t)`、`clean = x - t*v`，随后调用
`CPM(clean[..., :27], flow_time=0)`。完整梯度为
`g = d CPM(clean(x)) / d x`，保留当前 action expert 的 Jacobian。
只对当前一步反传，每步更新后 detach latent；所有模型参数冻结，且不创建优化器。

沿用项目的 flow 权重 `t/(1-t)`，将 `-dt*scale*t/(1-t)*g` 作为单独 FP32 修正，
先执行原始 Euler 更新再加入修正，保留 BF16 原更新的乘法舍入顺序。
这是 Universal-forward-style 的 flow 适配，CPM cosine 是匹配分数，非校准概率或 Q 值。
不包含 backward 内循环、self-recurrence 或 simulator 梯度。

默认：10 步采样、最后 3 步引导、scale=0.3、完整梯度、每个样本单步物理 latent
修正 RMS 上限 0.01。`MAX_UPDATE_RMS=0` 关闭限幅。限幅不是关节安全保证。
前 27 维按原 checkpoint 坐标处理（手臂 delta 7 + 手指绝对目标 20），
guidance 的后 5 维为零；最终 padding 仍可能因 action expert 耦合而间接变化。
不对 clean action 进行分位数截断。

CPM 的视觉、触觉输入与原方法一致：右手五指 `[5,3,7,5]`，单位 N，当前观测，
采样内观测固定；没有新增历史、腕力或本体状态分支。
`guidance_common/scorer.py` 是共享评分适配，保留动作计算图。
原 CPM 网络实现、权重加载、归一化和 policy 输入/输出转换直接复用已有代码。
state-conditioned CPM checkpoint 在 v1 中明确拒绝，避免未验证的状态适配。

## 快速运行

在 `/path/to/guidence-touch` 下执行：

```bash
# 1. CPU 数值测试，无须 checkpoint/GPU。
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD:$PWD/src:$PWD/packages/openpi-client/src" \
  JAX_PLATFORMS=cpu /path/to/touchguidance/bin/python \
  -m unittest tac_forward_guidance.sampler_test tac_forward_guidance.entrypoint_test -v

# 2. GPU 节点：真实权重及固定输入离线验证（本次已完成，换权重后重跑）。
bash tac_forward_guidance/run_verify.sh

# 3. 只打印评估命令，不启动模型/仿真，不创建结果目录。
DRY_RUN=1 bash tac_forward_guidance/run_eval.sh

# 4. GPU 节点：各 3 个配对 seed，检查运行及失败阶段。
NUM_TRIALS=3 bash tac_forward_guidance/run_eval.sh
```

默认权重：

- π0.5：`/path/to/datasets/sim/poker-draw/pi05_trex/pi05_pytorch_20k`
- CPM：`/path/to/datasets/touchguidence/cpm_dino_b128_g8_20260928_173559/best.pt`
- CPM 数据配置：`/path/to/datasets/sim/poker-draw/pi05_trex/touchguide/cpm_datasets/poker_0920_200_touchguide_v1`

可通过 `BASE_CHECKPOINT`、`CPM_CHECKPOINT`、`DATA_ROOT`、`TOKENIZER_MODEL` 覆盖。
默认输出在本目录 `artifacts/` 下，使用时间戳及进程号建立新目录，禁止覆盖。
模型端和仿真端分别使用原 `touchguidance` / `vtla` 环境；默认新端口 18787。

## 验证和效果评估

CPU 测试验证：解析链式梯度、与 stop-velocity 的区别、逐样本梯度及限幅、
CPM 时间固定为零、关闭引导/零梯度的原采样一致性、padding、冻结参数和方法标签。

GPU `verify.py` 默认逐输入处理，减少反传激活显存。验证报告保存 stock 对照、
早期轨迹一致性、full/stop 梯度差异、实际动作差异、显存/耗时、限幅次数。
离线验证启用额外局部 score probe：在 `x+shift` 上重新计算 π0.5 速度和 clean 分数。
分数不必每次都增加，BF16 舍入及非线性都可能影响有限步更新；报告不等同闭环成功。
生产采样默认不启用这个额外 probe，避免多一次 action-expert forward。

评估保持原控制器、30 Hz、预测/执行均为 30 步、相同随机卡牌位姿及噪声 seed 规则。
模型推理期间仿真暂停，因此另行报告 server latency，不代表实时可部署。
runner 和 launcher 是原 TouchGuide 入口的独立适配版本，仿真实现、视频工具和
成功判定仍使用已有模块。没有修改旧脚本或将新实验写入旧 TouchGuide 结果。

默认保留旧诊断协议 `FULL_DURATION_EVALUATION=1`，会关闭部分失败保护，
结果明确标记 `diagnostic_only=true`，不能作为正式成功率。
数值稳定性保护始终开启，与此开关无关。2026-10-09 起，共享 poker middle preset
固定使用 `implicit` 而非 `implicitfast`，避免 MuJoCo 3.8.1 中轻薄卡牌脱离接触后
角阻尼被显式积分而发散；不改变 2 ms 步长、质量、阻尼、摩擦或伺服增益。
每个物理步检查 BADQPOS/BADQVEL/BADQACC/BADCTRL、非有限状态和时间推进。
发现异常立即终止、标记 `invalid_physics=true`；仅保留正常前缀的未完成 Raw，
不追加重置后的终帧，不将重置后追赶时钟的片段计入同一 trial。
积分器、MuJoCo 版本和保护版本写入 `summary.json` 的 `contact_model`；
launcher 记录保护模块及相关模型/物理源文件 hash，拒绝与旧协议混合续跑。
此修复会改变物理协议，修复前 baseline/TouchGuide/forward 结果不能直接作为配对对照。
正式评估使用 `FULL_DURATION_EVALUATION=0`，对所有方法使用相同协议；
成功定义仍是现有 simulator outcome monitor，需额外核对目标阶段是否满足。

```bash
# 稳定后逐步扫描；各配置使用独立输出目录。
GUIDANCE_SCALE=1 NUM_TRIALS=10 bash tac_forward_guidance/run_eval.sh
# 显式近似消融，不是完整 forward。
GRADIENT_MODE=stop_velocity bash tac_forward_guidance/run_eval.sh
# 同一协议续跑；代码、权重文件大小/mtime、路径或采样参数变化会拒绝复用旧 trial。
RESUME=1 RUN_NAME=已有运行目录名 NUM_TRIALS=10 bash tac_forward_guidance/run_eval.sh
```

当前触觉是 MuJoCo 分布式 taxel 力估计，并按 `card_core_geom` 筛选，继承原数据语义。
对真实传感器迁移和桌面接触的结论需要单独验证。

## 验证状态

实现会话已通过 CPU 数值/入口测试、TouchGuide 原采样回归、lint 和仿真 dry-run。
2026-10-09 在 H20 上使用真实 20k π0.5、epoch 400 CPM 及三个固定观测完成 GPU 验证，
详见 [离线报告](artifacts/verification_20261009_104955_3984315/report.json)：

- 关闭引导与 stock 的动作最大差异为 0，前七个无引导步骤一致，模型参数保持冻结。
- full 与 stop-velocity 梯度不同，证明本次真实模型保留了 action expert Jacobian。
- 归一化动作差异 RMS 为 0.001068，实际关节目标最大差异约 0.00334 rad；
  直接 padding 引导为 0。默认配置较保守，仍需闭环评估是否足够有效。
- 九次局部分数 probe 中三次提高，其余不变或小幅下降；不是单调上升或成功率保证。
- 验证进程 CUDA allocator 峰值约 7.98 GiB；耗时包含验证专用的额外 probe，
  不应作为线上延迟结论。

已运行 scale=0.3 的三次旧协议闭环；其中 seed 001/002 的数值重置使原轨迹
不能作为有效配对效果证据。修复积分器和异常检测后，另行重跑 seed 002 满 50 秒：
无数值报警或时间回退，Raw 5001 个状态采样通过验证，顶部视频 501 帧；任务仍未完成。
修复验证存于原评估目录的 `tac_forward_guidance/scale_0.3_physics_fixed_20261009_144300_guidance_only/`，
详见 `summary.json` 和 `run_checks.json`。原结果未覆盖。
281 项相关回归和 21 项 forward 测试通过；10 项既有失败单独排除，未更改旧参数来使其通过。
下一步评估 guidance 效果需让 baseline 与 forward 使用同一修复后的物理协议。
脚本默认使用旧诊断协议；正式成功率应显式设置 `FULL_DURATION_EVALUATION=0`。
注意：`COMPARE_BASE=0` 仅关闭每次推理的额外动作对比；`run_eval.sh` 仍默认评估两种方法。
只跑新方法时用 `evaluate.py --variants pi05+tac_forward_guidance --no-compare-base` 显式指定。
