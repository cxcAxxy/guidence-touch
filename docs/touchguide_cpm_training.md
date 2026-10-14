# CPM训练与启动：共享冻结DINOv2-Base与DDP

单卡训练入口为`src/openpi/touchguide/train.py`，DDP入口为`src/openpi/touchguide/train_distributed.py`；一键脚本统一为`scripts/train_poker_touchguide_cpm.sh`。当前训练观测与候选轨迹的匹配模型CPM，尚未接入pi0.5去噪guidance。旧8卡、batch128、400 epoch运行已完成20轮并保存checkpoint；本次计划新开单机16卡、batch128、验证batch128、200 epoch运行，不续训旧权重。模型环境配置见 [touchguidance 环境说明](touchguidance_environment.md)。旧版CNN训练说明已归档为 [touchguide_cpm_training_cnn_v1.md](touchguide_cpm_training_cnn_v1.md)。

## 配置与论文依据

论文CPM训练的Table XIV使用batch64、LR=1e-5、warmup ratio=0.05和200 epoch；本次保留LR、warmup比例与轮数，global batch选128以适配本地16卡。其他优化器/调度细节、仿真taxel处理、噪声采样粒度和网络具体尺寸为本地配置。[TouchGuide Appendix L2 / Table XIV](https://arxiv.org/html/2601.20239v6)。

| 配置 | 当前值及含义 |
| --- | --- |
| Python | `/path/to/touchguidance/bin/python`，无需先激活conda |
| 数据入口 | `pi05_trex/touchguide/cpm_datasets/poker_0920_200_touchguide_v1` |
| 输入 | 当前顶部/右腕RGB、右手五指三通道taxel、带噪候选动作与噪声时刻；无F6或历史力输入 |
| backbone | 一份共享冻结DINOv2-Base，视觉与触觉都使用预训练权重；始终eval/no_grad |
| 网络 | 每图257→65 token，融合序列456；8层、d1024、16 heads、FF4096、PreLN、dropout=.1 |
| 参数 | 总208,843,009；冻结86,580,480；可训练122,262,529 |
| 训练范围 | CPM视觉/触觉投影、identity/CLS、融合Transformer、观测输出投影、动作/时间编码器与温度 |
| GPU/进程 | 本次`--num-gpus 16`，单机16卡DDP；`--num-gpus N`和脚本顶部`NUM_GPUS`均可调整卡数，命令行参数优先 |
| 训练batch | 本次global128，每卡8条，gather后构成128×128矩阵，每行127个身份负配对 |
| 验证batch | 本次显式`--val-batch-size 128`；旧运行的默认验证分组64保持不变 |
| epoch | 200；39,333个训练样本每轮307个global batch，使用39,296样本，丢弃尾部37样本 |
| 总更新 | 61,400；其他global batch按`200 × floor(39,333 / global_batch)`计算 |
| 优化器 | AdamW；LR=1e-5；本地weight decay=1e-4、梯度范数裁剪=1 |
| LR调度 | 前5%更新warmup，本次为3,070次，再cosine降至初始LR的0.1，即1e-6；cosine形式/min ratio为本地选择 |
| 精度 | CUDA BF16 autocast；模型参数FP32，最终余弦/温度/损失FP32；原始小taxel映射使用FP64后输出FP32 |
| 触觉处理 | 固定train RMS、有符号asinh映射、保持7:5几何后pad；无硬clip或逐帧归一化 |
| worker | 每rank4个，16卡共64个；可改为0；DDP worker使用`spawn`，每个视频解码器只使用1个线程 |
| 动作噪声 | t∈{0,.1,…,.9}，global batch共享t，截断归一化几何分布p=.3；每条动作独立epsilon，同一列跨所有观测复用 |
| 日志 | 每20次更新记录按样本数加权的平均值，以及完整epoch平均值 |
| 验证 | 每epoch结束；20个episode、4,372样本、34组128条加尾组20条，共35组；rank分配组，all_reduce样本加权统计，rank0每epoch记录指标 |
| checkpoint | 仅rank0写NAS；默认每20个epoch及最后epoch覆盖更新`last.pt`，并在这些时刻按验证损失更新`best.pt`；默认`--save-interval 0`关闭独立step快照 |

网络详情见 [touchguide_cpm_network.md](touchguide_cpm_network.md)。论文报告118M可训练CPM参数；本地122.26M配置不是公开作者网络的逐项复现。这里不加载旧T-Rex触觉重建权重，也不训练或加载pi0.5模型；动作仍使用数据准备时固化的pi0.5 step20k归一化统计。

## Global batch、全局对比与验证

`--batch-size`表示所有rank合计的训练batch。它必须整除world size，并保证每rank至少2条样本；本次16卡最低global32，选global128时每卡8条。每rank>=2是现有collate/noise工具的要求，训练负样本池来自所有rank。卡数可通过`--num-gpus N`或脚本顶部`NUM_GPUS`调整；改变卡数需新建运行。

各rank编码本地样本后，以保留梯度的gather汇集观测与动作特征，参与同一个`Bglobal×Bglobal`双向InfoNCE目标。本次global128、world16时每卡编码8条，每个观测匹配128条全局候选。batch改变会改变负样本数量，以及训练随机基线`log(Bglobal)`和`1/Bglobal`。

| Global batch | 16卡每rank样本数 | 每epoch更新数 | 200 epoch总更新数 | 每epoch尾部丢弃 |
| --- | ---: | ---: | ---: | ---: |
| 64 | 4 | 614 | 122,800 | 37 |
| **128** | **8** | **307** | **61,400** | **37** |
| 256 | 16 | 153 | 30,600 | 165 |
| 512 | 32 | 76 | 15,200 | 421 |

global sampler每轮重新无放回打乱，并在global batch边界drop_last。改变batch应新建运行；其他卡数和batch组合的吞吐/显存需按相应配置检查。

验证batch由独立的`--val-batch-size`控制。本次显式设128，保留34组128样本和1组20样本，共35组；旧8卡运行的验证分组仍是64。每个rank处理分配给它的完整组；最后将各指标的weighted_sum和sample_count进行all_reduce，而不是平均rank均值。只有rank0记录每epoch最终统计；每20个epoch及最后一个epoch才从保存时刻选择best并保存权重。不同验证分组的loss/准确率基线不同，旧run与新run的绝对数值不可直接比较。

## 启动与输出

可以修改bash顶部配置，也可以用命令行参数覆盖。云端交接命令与挂载要求见 [云端训练说明](touchguide_cpm_cloud.md)。

启动时先校验数据、DINO和实际NAS输出挂载；`--check-paths`是只读路径检查。云端单机16卡先执行真实GPU检查：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 bash /path/to/guidence-touch/scripts/train_poker_touchguide_cpm.sh \
  --num-gpus 16 \
  --batch-size 128 \
  --val-batch-size 128 \
  --epochs 200 \
  --save-every-epochs 20 \
  --save-interval 0 \
  --python-bin /path/to/touchguidance/bin/python \
  --output-root /path/to/datasets/touchguidence \
  --check-only
```

`--check-only`运行真实前向/反向及最多每rank一个完整固定验证组，16卡最多16组，每组全部10个噪声时刻；不更新optimizer、不创建W&B run、不保存checkpoint。检查通过后，执行 [云端说明](touchguide_cpm_cloud.md) 中不含`--check-only`的完整正式命令。

启动器通过指定Python环境的torchrun入口创建每卡一个rank。省略`--num-gpus`时脚本沿用顶部的`NUM_GPUS`配置，当前默认16；单卡可显式传`--num-gpus 1`。云端环境可覆盖`--python-bin`；NAS输出位置可覆盖`--output-root`，但`paths.py`要求其位于实际可写的`/nas` NFS/SMB挂载。

如需关闭终端后继续运行，可在`tmux new -s poker_cpm`中执行。`Ctrl+B`然后`D`分离，`tmux attach -t poker_cpm`重新进入。

W&B继续使用entity=`xianchichen8-hunan-university-library`、project=`poker-draw-touchguide-cpm`与现有登录凭证。DDP仅rank0创建和记录一个run；其他rank不重复连接。首次登录可执行：

```bash
/path/to/touchguidance/bin/python -m wandb login
```

运行名按实际global batch与GPU数生成`cpm_dino_b<B>_g<G>_<YYYYMMDD_HHMMSS>`；本次16卡global128为`cpm_dino_b128_g16_<YYYYMMDD_HHMMSS>`。新运行要求输出目录为空，防止覆盖已有产物。

```text
/path/to/datasets/touchguidence/
  cpm_dino_b128_g16_<YYYYMMDD_HHMMSS>/
    run_config.json
    training.log
    metrics.jsonl
    best.pt
    last.pt
    wandb/
```

所有rank共享同一个NAS运行目录，仅rank0写日志和checkpoint。每份保存一份完整FP32模型（包括冻结DINO）、AdamW、scheduler、零起始epoch、已消费global batch数、global step、best validation、严格训练契约和rank0 W&B id/name/entity/project，并收集各rank对应的随机状态；16卡即16份rank RNG，不保存16份模型/optimizer。模型与AdamW状态合计原估计约1.7GiB/份，具体大小以实际保存为准。`--save-every-epochs 20 --save-interval 0`使200轮中在第20、40、…、200轮保存，只覆盖`best.pt`和`last.pt`，不再长期保留按step生成的独立权重。每个epoch仍验证、记录W&B与metrics；`best.pt`只比较保存轮次的`val/loss`。需要更短的恢复间隔时，可将`--save-interval`设置为正数，额外保留`step_XXXXXX.pt`。每20轮保存意味着中断后可能需要重训接近20轮。保存采用同目录临时文件与原子替换，并拒绝误覆盖其他类型artifact。

## 指标解释

- `train/loss`：双向InfoNCE平均损失；两个方向分别记录。每个训练日志点汇总最近一个日志间隔。
- `val/loss`：固定分组、固定epsilon下，各噪声时刻损失按训练采样概率加权；每轮都记录，**仅每20轮及最后一轮的val/loss参与best.pt选择**。
- `accuracy_obs_to_action`：每个观测在同组候选中匹配到对应示范动作的比例。
- `accuracy_action_to_obs`：每条动作在同组观测中匹配到对应观测的比例。
- `positive_score`、`negative_score`、`score_gap`：正负身份配对的原始余弦均值及其差值。
- `temperature`：可学习对比温度，与动作噪声时刻t不同。
- `val_t00/*`…`val_t09/*`：分别对应t=0,.1,…,.9；`val/late_loss`为.1～.4平均损失，供未来guidance分析，不用于选择best。
- `train_epoch/*`：完整epoch平均值；`train/lr`与`train/grad_norm`为实际更新LR和裁剪前梯度范数。
- `val/uninformed_loss`、`val/chance_accuracy`：与固定验证分组一致的随机匹配基线。

新验证分组128的均匀分数基线为loss≈ln(128)=4.852、准确率≈1/128=0.78%；最后一组20条另有自己的基线。验证包含34组128样本及最后一组20样本，按实际样本数加权，包括相应随机基线。负标签来自身份不匹配，不是物理失败标签；匹配loss和准确率不能直接代替仿真成功率。

## 中断与续训

将bash顶部`RESUME_CHECKPOINT`设为原运行的`last.pt`，再执行同一脚本；若曾启用正数`--save-interval`，也可从`step_XXXXXX.pt`恢复。脚本使用checkpoint所在原目录；在线W&B继续原run，离线W&B不保证合并已有离线日志。

严格契约校验数据索引、归一化与模型配置hash、global batch、验证分组、world size、seed、LR、总epoch等。续训必须保持world size、训练/验证batch和总epoch不变：旧8卡/400 epoch运行可以按原配置恢复，但不能作为新16卡/200 epoch运行的`--resume`输入；新运行也应以16卡、batch128、验证batch128和200 epoch恢复。保存/日志频率及worker数可修改。旧命令如果显式设置了`--save-interval 1000`，必须改成0才能关闭step快照。旧CNN checkpoint与本版网络不兼容，会在恢复时拒绝。

中途恢复按epoch重新构造各rank采样与噪声生成器，重播已消费batch前缀后恢复该rank对应的checkpoint RNG；不能将rank0 RNG复制给所有rank。epoch结束后的checkpoint则在下一轮iterator创建前恢复对应状态。若显式启用step快照并在最后一次更新之后、最终验证之前保存checkpoint，恢复只完成待执行验证与rank0按保存周期执行的best/last保存，不额外更新参数。已完成所有epoch验证的运行不再次续训。

在一次forward或更新中间中断时，不保存半更新状态，应从此前完整checkpoint恢复。CPU合成回归在相同配置下验证了mid-epoch、epoch边界与最终更新后恢复一致；不承诺GPU bitwise复现。只加载本项目生成且可信的checkpoint，格式marker不改变pickle的可信来源要求。

## 当前验证状态

触觉预处理CPU回归、独立InfoNCE公式/梯度回归，以及先前单卡实现的checkpoint往返和微型假CPM训练/续训调度回归已通过。新增2/8进程CPU Gloo全局InfoNCE检查也已通过，梯度与单进程全局计算的最大绝对差为1.19e-7；8进程报告为`/tmp/touchguide_distributed_validation_8.json`。

此前已完成H20单卡、真实batch64、BF16的前向/反向及一组固定验证数据的10个噪声时刻检查。这是单卡结果，不是八卡DDP结果。显存峰值allocated=11.248 GiB、reserved=11.602 GiB；所有可训练参数梯度有限，冻结DINO无梯度。候选动作对原始分数的梯度非零，去掉非零触觉会改变观测编码。该诊断不执行optimizer.step、不创建W&B run、不保存训练权重；未分配AdamW状态，后续训练额外状态估计约0.91 GiB，并应保留额外显存余量。报告位于数据目录`meta/cpm_training_gpu_validation.json`。这些接口检查不代表模型已学会任务。

2进程tiny trainer验证连续6次更新与中断后2+4、3+3、6+0恢复：最终模型、Adam、scheduler、进度、各rank RNG及指标精确一致。不均匀验证组（4、4、2，两个rank分2/1组）的样本加权all_reduce与单进程统计一致，只有rank0保存checkpoint与指标。报告为`/tmp/cpm_ddp_training_validation.json`。这些CPU合成更新没有训练实际CPM，没有加载预训练权重、连接W&B或写NAS。

新DDP入口也在H20上以world_size=1、global/local batch16、BF16通过真实前后向及一个完整64样本验证组的10个噪声时刻检查；梯度有限、冻结DINO无梯度，无optimizer更新、W&B或checkpoint。报告为`/tmp/cpm_ddp_entry_spawn_gpu_check.json`（num_workers=1，spawn worker正常退出）。该单卡结果不用于估算八卡训练显存。

先前实际数据的CPU读取检查也通过：spawn单worker与零worker的首个训练batch完全一致；当时使用默认64条验证分组，共69组，尾组20样本读取正常。报告为`/tmp/cpm_spawn_workers_validation.json`，未加载模型或写NAS。本次显式改为128条验证分组，需要重新做16卡`--check-only`。

旧8卡/400 epoch运行已训练至第20轮并保存checkpoint；其GPU日志不代替新单机16卡、验证分组128的检查。云端用前面的`--check-only`检查本次GPU数、训练和验证batch：真实前后向加最多每rank一个完整验证组，全部10个噪声时刻，不更新optimizer、不创建W&B、不产生训练checkpoint。GPU报告应明确world size、global/local batch与检查范围。原单卡batch64报告仍保留在`meta/cpm_training_gpu_validation.json`，不作为新配置的验收结果。新运行尚未启动。
