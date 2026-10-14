# 云端单机十六卡DDP CPM训练交接

此流程训练TouchGuide CPM，加载共享冻结DINOv2-Base，不加载或训练pi0.5模型。新实验在**单节点16张A100**上通过`torchrun`启动16个进程，使用global batch 128、验证分组128、200 epoch。代码位于`/path/to/guidence-touch`。

旧8卡、global batch 128、400 epoch运行已完成20轮并保存checkpoint；它与本次16卡、200 epoch配置不兼容，**新实验不要传`--resume`，也不要删除旧权重**。旧运行需要时仍可按原8卡/400 epoch配置续训。云端先执行下面的`--check-only`，确认16卡A100和验证分组能正常运行。

## 云端先检查，再正式训练

```bash
cd /path/to/guidence-touch

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 bash scripts/train_poker_touchguide_cpm.sh \
  --num-gpus 16 \
  --python-bin /path/to/touchguidance/bin/python \
  --data-root /path/to/datasets/sim/poker-draw/pi05_trex/touchguide/cpm_datasets/poker_0920_200_touchguide_v1 \
  --output-root /path/to/datasets/touchguidence \
  --batch-size 128 \
  --val-batch-size 128 \
  --epochs 200 \
  --lr 1e-5 \
  --num-workers 4 \
  --save-every-epochs 20 \
  --save-interval 0 \
  --wandb-enabled \
  --check-only
```

以上命令先执行真实CPM的16卡前向/反向检查，并检查最多每rank一个完整固定验证组：16卡最多16组，每组覆盖全部10个噪声时刻。检查不执行optimizer更新、不创建W&B run、不保存checkpoint。成功退出后执行下面的正式训练命令；`--wandb-enabled`只在正式训练时创建rank0 run。

```bash
cd /path/to/guidence-touch

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 bash scripts/train_poker_touchguide_cpm.sh \
  --num-gpus 16 \
  --python-bin /path/to/touchguidance/bin/python \
  --data-root /path/to/datasets/sim/poker-draw/pi05_trex/touchguide/cpm_datasets/poker_0920_200_touchguide_v1 \
  --output-root /path/to/datasets/touchguidence \
  --batch-size 128 \
  --val-batch-size 128 \
  --epochs 200 \
  --lr 1e-5 \
  --num-workers 4 \
  --save-every-epochs 20 \
  --save-interval 0 \
  --wandb-enabled
```

`--num-gpus N`可选择单节点实际使用的卡数，脚本顶部`NUM_GPUS`也可修改；命令行参数优先，当前默认值为16。本次传16，通过所选Python环境的`torchrun`创建16个rank；若要单卡运行，显式传`--num-gpus 1`。改卡数应作为新运行，训练global batch仍需整除卡数。`--python-bin`可改为云端实际环境的Python。

`--batch-size 128`是**所有rank合计的训练batch**，16卡每卡8条。`--val-batch-size 128`把验证候选组设为128条，最后不足128条的尾组仍保留。两项均显式指定；旧运行保留它原来的64条验证分组。训练global batch必须整除GPU数，且每rank至少2条样本；16卡最低global32。

每个rank编码自己的观测与动作，随后gather特征，形成128×128匹配矩阵。每个观测对应127个batch内身份负样本，训练目标不是16个相互独立的batch8 CPM。改变global batch也会改变训练负样本数量与InfoNCE随机基线。

本次训练集有39,333条样本，global128每轮`floor(39,333 / 128)=307`次更新，丢弃尾部37条；200轮合计61,400次更新，5% warmup为3,070次。验证集4,372条，按显式`--val-batch-size 128`划为34组128条和1组20条，共35组。它改变验证损失和准确率的随机基线，因此不能与旧64条验证分组的`val/loss`直接比较。

运行名按实际global batch与GPU数生成`cpm_dino_b<B>_g<G>_<时间戳>`；这次为`cpm_dino_b128_g16_<时间戳>`，可用`--run-name NAME`指定。`--num-workers 4`为每rank4个worker，16卡总共64个；DDP数据worker使用`spawn`，可根据数据读取条件调整。

## 启动前的路径检查

```bash
cd /path/to/guidence-touch
bash scripts/train_poker_touchguide_cpm.sh \
  --num-gpus 16 \
  --python-bin /path/to/touchguidance/bin/python \
  --output-root /path/to/datasets/touchguidence \
  --check-paths
```

`--check-paths`是只读路径检查，只验证数据、依赖与NAS挂载，不加载DINO、不运行训练或W&B、不生成权重；正式启动也执行同样的路径检查。它不替代16卡训练/通信的GPU检查。

## 必须可访问的路径

云端把同一NAS挂载到`/nas`时，以下路径可沿用。

| 内容 | 路径 |
| --- | --- |
| CPM入口、索引、配置与归一化 | `/path/to/datasets/sim/poker-draw/pi05_trex/touchguide/cpm_datasets/poker_0920_200_touchguide_v1` |
| 实际H5，当前episodes链接目标 | `/path/to/datasets/sim/poker-draw/pi05_trex/touchguide/cpm_datasets/poker_0920_200_v1/episodes` |
| 顶部和右腕视频，合计400个MP4 | `/path/to/datasets/sim/poker-draw/pi05/0920_200/videos/chunk-000/observation.images.head`和`observation.images.right_wrist` |
| DINOv2-Base目录，含manifest、权重与config | `/path/to/datasets/sim/poker-draw/pi05_trex/touchguide/backbones/dinov2_base_f9e44c8` |
| 权重与日志输出 | `/path/to/datasets/touchguidence` |

不能仅复制当前CPM入口：episodes含绝对符号链接，H5还记录视频绝对路径。如果更改NAS挂载位置，需要一起迁移链接、视频路径及JSON中的reader/backbone路径，仅覆盖`--data-root`不够。建议保持`/nas`挂载位置一致。

配置中的pi0.5源checkpoint路径仅记录来源；CPM训练直接使用已有数据和统计文件，不加载该checkpoint，也不改写既存CPM数据metadata。

## 环境、验证与日志

当前模型环境为`/path/to/touchguidance/bin/python`，从已验证的`vtla`离线复制，配置与激活见 [环境说明](touchguidance_environment.md)。云端如果同样挂载CPFS且此环境可用，不必激活conda；其他环境需有CUDA版PyTorch及CPM依赖。本机记录版本为Torch2.7.1、torchvision0.22.1、Transformers4.53.2、h5py3.16.0、PyAV17.1.0、NumPy1.26.4、W&B0.27.2、safetensors0.8.0、Pillow12.2.0。模型从本地DINO目录离线加载，无需重做baseline转换或数据构建。

新运行验证为35个固定组：34组128样本与1组20样本，共4,372条；旧8卡run的64样本分组保持不变。各rank分配完整、互不重复的验证组；统计按样本数加权并`all_reduce`汇总。只有rank0记录最终指标；每20个epoch及最后一个epoch根据当次验证结果更新best并写checkpoint。

W&B仍使用entity=`xianchichen8-hunan-university-library`、project=`poker-draw-touchguide-cpm`，**只有rank0创建和记录一个run**。云端需用有该entity权限的账号登录：

```bash
/path/to/touchguidance/bin/python -m wandb login
```

## Checkpoint与续训

所有rank使用同一个NAS运行路径，但只有rank0写入文件：

```text
/path/to/datasets/touchguidence/
  cpm_dino_b128_g16_<时间戳>/
    best.pt
    last.pt
    training.log
    metrics.jsonl
    run_config.json
    wandb/
```

`paths.py`要求输出目录在`/nas`下且实际位于可写NFS/SMB挂载；NAS未挂载或输出落在本地盘时会报错。`--resume PATH`必须指向原NAS运行目录的checkpoint，继续写原目录。

每份checkpoint包含一份完整FP32模型、AdamW和scheduler状态、进度/契约、各rank的随机状态及rank0 W&B信息；16卡保存16份rank RNG，不复制16份完整模型/optimizer。每20个epoch（20、40、…、200）更新同名`last.pt`；`best.pt`仅从这些保存时刻挑选验证损失最低的权重并覆盖更新。每个epoch仍验证并记录W&B指标。`--save-interval 0`关闭按step独立保存，因此不会产生`step_*.pt`；若需要更短的中断恢复间隔，可显式设`--save-interval 1000`等正数。长期保留的权重通常只有best/last两份，但中断后默认最多需要重训接近20个epoch。完整模型与AdamW状态原先估计约1.7GiB/份，实际以文件大小为准；原位替换时还需为临时文件预留空间。

续训必须保持**world size、global batch、验证分组、总epoch**及其他严格契约一致。因此新16卡run应以相同16卡、batch128、验证batch128、200 epoch恢复；旧8卡/400 epoch的`last.pt`不能作为此次新运行的`--resume`输入，但可以按旧配置恢复。更改卡数、batch或总epoch应新建运行。保存/日志频率及worker数可调整。旧命令若仍显式带`--save-interval 1000`，会继续产生按step保存的权重，必须改为`--save-interval 0`。

## 已完成检查与单机16卡检查范围

- 2/8进程CPU Gloo合成检查验证全局InfoNCE及可导gather的梯度缩放，与单进程全局计算的最大梯度绝对差为1.19e-7；同时检查global sampler、共享噪声时刻及各rank RNG恢复。8进程报告为`/tmp/touchguide_distributed_validation_8.json`。
- 2进程tiny trainer完成连续6次更新，以及中断后2+4、3+3、6+0三种恢复。最终模型、Adam、scheduler、进度、各rank RNG和指标精确一致。验证组大小为4、4、2，两个rank分2/1组；按样本数加权all_reduce与单进程汇总一致，仅rank0写checkpoint和指标。报告为`/tmp/cpm_ddp_training_validation.json`。
- 新DDP入口在H20上以world_size=1、global/local batch16、BF16完成真实前向/反向和一个64样本验证组的全部10个噪声时刻检查。所有可训练梯度有限，冻结DINO无梯度；未更新optimizer、未创建W&B或保存checkpoint。报告为`/tmp/cpm_ddp_entry_spawn_gpu_check.json`（num_workers=1，spawn worker正常退出）。

CPU合成检查没有加载真实CPM或预训练权重、没有使用GPU或写NAS。新入口单卡CUDA检查与旧8卡运行都不能替代**本次16卡A100、验证组128**的检查；在云端先用`--check-only`确认通信、真实数据、当前batch的梯度与显存，再开始正式训练。

原H20单卡global64 BF16检查为allocated=11.248GiB、reserved=11.602GiB，不含optimizer更新或AdamW状态分配，仅保留为单卡网络前后向记录。新16卡每卡8条训练样本，验证按128样本完整组处理；这些历史数字不用于估算新运行的显存峰值。
