# touchguidance 模型环境

环境目录：`/path/to/miniconda/envs/touchguidance`。

该 Conda 环境用于 CPM 训练及 pi0.5 + TouchGuidance 模型推理，以已验证的 `vtla` 为基础离线复制，保留 CUDA、JAX/Flax 和 pi0.5 所需 Transformers 补丁。环境依赖采用独立文件副本；当前模型源码注册到同一个项目目录。

## 使用

```bash
source /path/to/miniconda/etc/profile.d/conda.sh
conda activate touchguidance
cd /path/to/guidence-touch
```

TouchGuide 的训练与准备脚本默认使用新环境的绝对 Python 路径，无需先激活：

```text
/path/to/touchguidance/bin/python
```

云端训练命令见 [八卡交接说明](touchguide_cpm_cloud.md)。云端共享同一 CPFS 与 NAS 时，路径可以直接沿用。`--python-bin` 或 `PYTHON_BIN` 可以覆盖脚本默认值。

## 配置

- Python 3.11.15，PyTorch 2.7.1，torchvision 0.22.1，Transformers 4.53.2。
- JAX/jaxlib/CUDA 插件 0.5.3，Flax 0.10.2，Orbax Checkpoint 0.11.13。
- NumPy 1.26.4、h5py 3.16.0、PyAV 17.1.0、W&B 0.27.2、safetensors 0.8.0。
- 新环境已移除无关的 N0-VTLA editable 入口，原 `vtla` 未修改。
- 通过新环境专用的 `touchguidance_project.pth` 注册 `openpi-main/src` 和 `packages/openpi-client/src`。这提供本地源码导入路径，没有执行 `pip install -e` 或重新解析项目依赖。
- 本环境定位为模型端；仿真程序继续按仿真项目的环境与依赖配置启动。

本机创建命令如下，已有同名环境时无需重跑：

```bash
/path/to/miniconda/bin/conda create \
  --prefix /path/to/miniconda/envs/touchguidance \
  --clone /path/to/miniconda/envs/simulator-env \
  --offline --copy --yes
```

本机 `vtla` 的 `pip` 实际为 26.2，但 Conda 记录仍为 26.1.2。直接克隆会混合两版文件；本次已把原环境的 `pip/` 与 `pip-26.2.dist-info/` 完整复制到新环境，再仅在新环境执行 `pip uninstall n0vtla`，让 `pip check` 通过。原 `vtla` 文件没有被修改。

重建环境时，复制完成后先执行相同的 pip 修复，再移除新环境的 `n0vtla` editable 入口，并把上面的两条项目源码路径写入新环境的 `site-packages/touchguidance_project.pth`。若在另一台服务器从零创建，也须保留 pi0.5 的 Transformers 补丁；仅安装原版 Transformers 不完整。

## 验证

- `pip check`：`No broken requirements found`。
- 直接导入 CPM、八卡 DDP 入口、pi0.5 PyTorch 模型、policy 配置与 `openpi-client` 成功。
- 五份 Transformers 替换文件与项目源码 SHA256 完全一致，Gemma RMSNorm 保留 `cond` 参数。
- 本机 H20 上的 PyTorch BF16 前向/反向及 JAX 0.5.3 CUDA 小计算通过。八卡 NCCL 尚待云端 `--check-only` 检查。
- 本次检查报告：`/tmp/touchguidance_environment_validation.json` 与 `/tmp/touchguidance_jax_gpu_validation.json`。

仅执行依赖、导入与小规模 CUDA 检查，没有加载模型权重或启动正式训练。
