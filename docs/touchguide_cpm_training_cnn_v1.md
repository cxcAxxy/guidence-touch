# CPM 训练与启动

训练入口 `src/openpi/touchguide/train.py`，一键脚本 `scripts/train_poker_touchguide_cpm.sh`。
本阶段训练观测与候选轨迹的匹配模型 CPM，尚未接入 pi0.5 去噪 guidance。

## 默认配置

| 配置 | 默认值及含义 |
| --- | --- |
| Python | `/path/to/simulator-env/bin/python`，无需先激活 conda |
| 训练数据 | `pi05_trex/touchguide/cpm_datasets/poker_0920_200_touchguide_v1` |
| 输入 | 当前顶部/右腕图像、右手五指三通道触觉点阵、带噪候选动作；不输入 F6 或历史力 |
| 参数 | 共 27,702,561；冻结 DINOv2 的 22,056,576；训练其他 5,645,985 |
| 训练范围 | 视觉投影、触觉 CNN/幅值分支、融合 Transformer、动作/时间编码器、温度等 CPM 参数 |
| batch | 64 个观测和 64 条动作组成 64×64 匹配矩阵；每行 1 个正配对和 63 个负配对 |
| epoch | 30；39,333 个训练样本，每轮 614 个 batch，使用 39,296 样本，随机舍弃尾部 37 样本 |
| 总更新 | 18,420；每轮重新无放回打乱，不把数据预先固定分成训练 batch |
| 优化器 | AdamW，LR=1e-4，weight decay=1e-4，梯度范数裁剪=1 |
| LR 调度 | 前 5% 更新 warmup，再 cosine 降至 1e-5 |
| 精度 | CUDA BF16 autocast；模型参数 FP32，最终余弦相似度/温度/损失 FP32 |
| worker | 4；不够稳定或 NAS 解码慢时可改为 0；每个视频解码器只用 1 个线程 |
| 动作加噪 | t∈{0,.1,…,.9}，batch 共享 t，截断归一化几何分布 p=.3；每条动作独立 ε |
| 日志 | 每 20 次更新的样本加权平均值，以及完整 epoch 平均值 |
| 验证 | 每个 epoch 结束；20 个验证 episode，4,372 样本，固定 69 组，噪声按样本 ID 固定 |
| 权重 | 每 1,000 步保留 `step_XXXXXX.pt`；每轮更新 `last.pt`，验证改善时更新 `best.pt` |

以上训练超参数是本地第一版设置，不代表论文公布了这些精确值。触觉 CNN 使用 CPM 自己的初始化；不加载此前 T-Rex 触觉重建编码器的 checkpoint。动作使用准备数据时固化的 pi0.5 20,000 步 checkpoint 归一化文件，CPM 训练不加载 pi0.5 模型。

## 启动

直接修改 bash 文件顶部配置，再执行：

```bash
bash /path/to/guidence-touch/scripts/train_poker_touchguide_cpm.sh
```

若希望关闭终端后继续运行，可先 `tmux new -s poker_cpm`，在其中执行上面的 bash。
`Ctrl+B` 然后 `D` 分离，`tmux attach -t poker_cpm` 重新进入。

W&B：entity=`xianchichen8-hunan-university-library`，project=`poker-draw-touchguide-cpm`。
使用现有登录凭证；如当前 Python 环境尚未登录，可执行：

```bash
/path/to/simulator-env/bin/python -m wandb login
```

每次新训练生成带时间戳的独立运行名与目录，禁止误覆盖已有输出。

```text
/path/to/datasets/sim/poker-draw/pi05_trex/touchguide/cpm_runs/<RUN_NAME>/
  run_config.json
  training.log
  metrics.jsonl
  step_001000.pt
  ...
  best.pt
  last.pt
  wandb/
```

checkpoint 包含完整模型（含冻结 DINO）、AdamW、学习率调度器、样本进度、随机状态、配置与 W&B run ID；约 150 MiB/份。

## W&B 图怎么读

- `train/loss`：对称 InfoNCE，总损失。分别记录 `loss_obs_to_action` 和 `loss_action_to_obs`；训练每个点是最近一段更新的平均值。
- `val/loss`：固定验证集、各噪声时刻损失按训练采样概率加权的平均值；**选 best.pt 的标准是最低 val/loss**。
- `accuracy_obs_to_action`：为一个观测在本组候选中选对示范动作的比例。
- `accuracy_action_to_obs`：为一条动作在本组观测中选对其对应观测的比例。
- `positive_score` / `negative_score` / `score_gap`：正、负配对的原始余弦相似度和差值；差值增大说明更容易区分配对。
- `temperature`：学习到的对比温度。
- `val_t00/*`…`val_t09/*`：分别对应 t=0,.1,…,.9。`val/late_loss` 单独统计 .1～.4 的平均值，供未来末段去噪 guidance 参考；不是当前选 best 的标准。
- `train_epoch/*`：完整训练轮次的平均值；`train/lr`、`train/grad_norm` 为学习率和裁剪前梯度范数。
- `val/uninformed_loss` 和 `val/chance_accuracy` 给出当前固定分组的随机匹配基线。

64 候选完全分不清时，loss 约 ln(64)=4.159，准确率约 1/64=1.56%。验证尾组只有 20 条，指标按实际样本数加权，基线同时按这个分组计算。
负标签来自观测/动作身份不匹配，不能据此断言该动作在物理上必然失败；CPM 的 loss 下降也不能直接替代仿真任务成功率。

## 中断续训

将 bash 顶部 `RESUME_CHECKPOINT` 填为原运行的 `last.pt` 或 `step_XXXXXX.pt`，再运行同一脚本。
脚本自动使用 checkpoint 所在目录；在线 W&B 自动继续原 run。离线 W&B 不保证合并已有离线日志。

续训校验数据/归一化/模型配置、batch、seed、LR、总 epoch 等，避免静默换训练条件。
本版本总 epoch 必须保持原值，不支持直接把 30 改成 60 来延长 cosine 调度。
保存频率、日志频率、worker 数可以更改。中途恢复会重放当前 epoch 已消费的样本前缀来恢复私有噪声 RNG，因此启动可能有短暂等待。
中断发生在一次更新中间时，不保存半更新状态，使用此前完整保存的 checkpoint。

## 检查范围

已完成 H20 上真实 batch 64、BF16 的前向/反向和一组固定验证数据的十个噪声时刻检查。
显存峰值 allocated=1.889 GiB，reserved=2.217 GiB；梯度有限，冻结 DINO 无梯度。
该诊断没有 optimizer.step，没有创建 W&B run，也没有产生训练权重。
AdamW 状态未在该 GPU 诊断中分配，额外状态预计约 43 MiB；训练时预取/分配行为可能使显存峰值略变。

诊断报告保存于数据目录 `meta/cpm_training_gpu_validation.json`；可以通过训练入口的 `--check-only --report ...` 再次检查。
CPU 合成回归覆盖指标加权、验证固定噪声与状态恢复、checkpoint 往返以及训练调度/续训逻辑；合成参数更新不代表实际 CPM 已训练。
