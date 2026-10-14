# 脚本用途

当前摸牌TouchGuide CPM的单机多卡DDP入口；新实验用16张A100、global batch128、验证batch128、200 epoch。云端先做真实GPU检查：

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

模型环境与激活方式见 [touchguidance 环境说明](../docs/touchguidance_environment.md)。数据和DINO依赖已准备好，支持命令行参数与环境变量覆盖；`--check-paths`只检查路径和NAS挂载。云端可直接转发 [16卡交接说明](../docs/touchguide_cpm_cloud.md) 中的完整`--check-only`和正式命令，训练及续训规则见 [训练说明](../docs/touchguide_cpm_training.md)。2/8进程CPU Gloo全局梯度检查和2进程tiny trainer中断/续训检查已通过；旧8卡/400 epoch运行保存了第20轮checkpoint，新16卡实验尚未启动。

`--check-only`执行真实前后向和最多每rank一个完整固定验证组，16卡最多16组、每组全部10个噪声时刻，不更新optimizer、不连接W&B、不保存checkpoint。成功后执行云端说明中不带`--check-only`的正式命令。`--check-paths`则仅做只读路径检查，不加载模型。运行名包含实际batch与GPU数，本次如`cpm_dino_b128_g16_<时间戳>`。

`--num-gpus N`通过torchrun在单节点启动N个进程；也可以修改脚本顶部`NUM_GPUS`，命令行参数优先，当前默认16；单卡请显式传`--num-gpus 1`。本次传16，global batch128对应每卡8条。`--batch-size`必须整除GPU数并保证每rank至少2条；每rank>=2是现有collate/noise接口限制。改变GPU数应新建运行。

各rank的特征gather后形成128×128匹配矩阵，每个观测有127个全局身份负样本。显式`--val-batch-size 128`使4,372条验证样本组成34组128条和尾组20条，共35组，按rank分配并all_reduce样本加权统计；旧8卡运行保留64条验证分组。仅rank0连接一个W&B run并写NAS checkpoint。checkpoint保存完整模型/AdamW/scheduler和各rank RNG；每20个epoch覆盖更新best/last，每epoch仍验证和记录W&B。`--save-interval 0`不生成独立step快照。续训要求GPU数、训练/验证batch与总epoch保持不变；本次新运行不能`--resume`旧8卡/400 epoch权重，旧权重保留可按原配置恢复。

CPU验证中，全局InfoNCE梯度最大误差为1.19e-7；2进程合成训练的2+4、3+3、6+0恢复精确一致，不均匀验证组加权汇总与单进程一致，rank0单独写checkpoint。报告为`/tmp/touchguide_distributed_validation_8.json`及`/tmp/cpm_ddp_training_validation.json`；合成检查未训练实际CPM。

200 epoch更新数按`200 × floor(39,333 / global_batch)`计算：本次global128每轮307次、共61,400次，5% warmup为3,070次。可用`--python-bin`指定云端环境、`--output-root`指定NAS输出；输出必须通过`paths.py`的实际可写NAS挂载检查。

| 根目录入口 | 用途 |
| --- | --- |
| `train_poker_touchguide_cpm.sh` | 共享冻结DINOv2-Base CPM的单卡/DDP训练启动器；global batch，NAS rank0保存 |
| `prepare_poker_touchguide_cpm.sh` | 准备/校验DINOv2-Base及CPM模型配置，重复运行会检查已有配置 |
| `prepare_poker_touchguide_baseline.sh` | pi0.5 20k权重的JAX→PyTorch转换、固定输入及两后端一致性校验，后续guidance需要 |
| `train_poker_f6_vqvae.sh` | T-Rex六维力编码器训练 |
| `train_poker_tactile_map.sh` | T-Rex触觉图像重建编码器训练 |
| `train_poker_tactile_expert.sh` | T-Rex触觉专家训练，保留原方案的重训入口 |
| `train.py` / `train_pytorch.py` | OpenPI通用JAX / PyTorch训练入口 |
| `compute_norm_stats.py` | 计算数据归一化统计量 |
| `serve_policy.py` | OpenPI策略服务 |
| `train_test.py` | 自动发现的训练与续训检查 |
| `docker/` | Docker及NVIDIA容器环境安装工具 |

`touchguide/` 保留实际实现：数据构建、正负配对、噪声配置、Base下载、模型配置及验证工具。
旧Small下载器、旧CNN模型校验器和已完成的自动续训监控已清理。

## 重建数据所需工具

以下步骤已完成。仅在重新构建数据时使用；构建器会拒绝覆盖已有数据。
此前四个分步骤shell包装已删除，保留相应Python程序和参数：

| 顺序 | Python程序 | 产物 |
| --- | --- | --- |
| 1 | `touchguide/build_cpm_dataset.py` | 从pi05与触觉数据构建配对来源；默认旧数据入口`poker_0920_200_v1` |
| 2 | `touchguide/prepare_paper_dataset.py` | 从上述来源建立当前正样本入口`poker_0920_200_touchguide_v1`；训练时在线配负样本 |
| 3 | `touchguide/prepare_batch_pairing.py` | batch配对配置及固定验证分组 |
| 4 | `touchguide/prepare_action_noise.py` | 在线动作噪声配置及小规模示例 |
| 5 | `prepare_poker_touchguide_cpm.sh` | Base依赖及CPM模型配置 |

从仓库根目录运行Python工具前设置：

```bash
export PYTHONPATH="$PWD/src:$PWD/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
/path/to/touchguidance/bin/python scripts/touchguide/prepare_batch_pairing.py --help
```

配对和噪声的具体设置见 [配对说明](../docs/touchguide_batch_pairing.md) 和 [噪声说明](../docs/touchguide_action_noise.md)。
NAS上的来源元数据记录历史创建时的脚本路径及hash；清理包装入口不改变既有数据与历史记录。
