# π0.5 + OA Guidance for tactile poker manipulation

本仓库整理了 KaiHand 扑克牌仿真任务的 π0.5 推理适配、Observation–Action（OA）打分器、指尖扰动/触觉合成增强、训练与评价脚本，以及对应的 MuJoCo 仿真环境源码。

**数据、模型权重、tokenizer 和 DINO 预训练权重不在仓库中。** 它们应放在本机或集群外部路径。仓库没有打包原始数据集、增强样本、训练 checkpoint、视频、实验日志或访问凭据。

## 内容

- `src/openpi/touchguide/`：OA 打分器、动作/状态编码、触觉和图像预处理、训练、推理适配和采样引导。
- `scripts/touchguide/`：数据审计、接触几何、指尖运动学扰动、触觉合成、归一化、配对策略、训练和仿真评价工具。
- `scripts/build_poker_oa_dataset.sh`：构建干净配对、10 倍增强配对和 1:1、1:5、1:10 VDS 训练入口。
- `scripts/train_poker_recovery_state_cpm.sh`：分别训练不同增强比例；`--ratio all` 顺序运行三个比例。
- `scripts/run_poker_oa_guidance.sh`：π0.5 + OA guidance 仿真入口。默认预测 30 步，每次执行前 5 步，然后读取新观测重新预测。
- `scripts/touchguide/export_oa_inference_metadata.py`：从训练数据入口导出部署需要的配置，不复制样本或权重。
- `simulation/`：所需 KaiHand MuJoCo 源码、任务模型和资源文件；不含采集数据和运行结果。
- `docs/`：网络结构、数据构造、归一化、配对损失、训练和部署说明。

## 环境

建议使用 Python 3.11，并把模型、仿真和数据构造放在相互隔离的环境中：OpenPI/模型依赖 NumPy `<2`，当前仿真包依赖 NumPy `>=2`。示例服务器环境分别是 `touchguidance` 和 `vtla`。其他环境请按 `pyproject.toml` 和 `simulation/pyproject.toml` 安装依赖。

```bash
uv sync --no-dev
```

在 OpenPI Python 环境里安装 PyTorch 所需的 Transformers 补丁（上游 OpenPI 的 PyTorch 推理要求）：

```bash
python -m pip install 'transformers==4.53.2'
python - <<'PY'
import site
print(site.getsitepackages()[0])
PY
```

将上一步打印路径下 `transformers/` 目录与本仓库 `src/openpi/models_pytorch/transformers_replace/` 中的文件合并。不要覆盖其他 transformers 文件。可参考 [上游 OpenPI 安装说明](README_OPENPI.md)。

安装仿真环境时，使用 `simulation/pyproject.toml` 的依赖配置，并设置 `SIM_ROOT` 指向仓库中的 `simulation/`。

π0.5 基础权重、OA 打分器 checkpoint、tokenizer 和 DINOv2 权重需要在受控位置单独准备。仓库不提供这些文件，也不会把权重上传到 GitHub。

## 构建数据

以下路径要按你的环境修改。`SOURCE_ROOT` 是已准备好的 π0.5 observation/action 数据入口，`TACTILE_ROOT` 是时间同步的触觉 sidecar，输出目录应在仓库之外：

```bash
SOURCE_ROOT=/path/to/poker_pi05_dataset \
TACTILE_ROOT=/path/to/synchronized_tactile_sidecars \
OUTPUT_DIR=/path/to/output/poker_oa_dataset \
SIM_ROOT="$PWD/simulation" \
PYTHON_BIN=/path/to/data-env/bin/python \
bash scripts/build_poker_oa_dataset.sh
```

该脚本只准备数据，不启动训练。会产生全部干净起点和 10 倍增强样本，再创建嵌套 VDS 入口。磁盘空间需按数据规模预留。细节见 [`docs/touchguide_recovery_state_cpm.md`](docs/touchguide_recovery_state_cpm.md) 与 [`docs/oa_scorer_architecture.md`](docs/oa_scorer_architecture.md)。

## 训练

```bash
AUGMENTATION_ROOT=/path/to/output/poker_oa_dataset \
OUTPUT_ROOT=/path/to/checkpoints \
NUM_GPUS=8 BATCH_SIZE=128 VAL_BATCH_SIZE=128 EPOCHS=200 \
bash scripts/train_poker_recovery_state_cpm.sh --ratio all
```

`--ratio all` 会按 1:1、1:5、1:10 启动三个独立训练，每个比例使用独立的新模型和 W&B run。训练权重只写到 `OUTPUT_ROOT`，不会写进源码仓库。

## 部署和仿真评价

先将配置文件复制到仓库之外并填写真实路径：

```bash
cp configs/oa_guidance.env.example /path/to/oa_guidance.env
# 编辑 OA_CHECKPOINT、OA_DATA_ROOT、BASE_CHECKPOINT、MODEL_PYTHON、SIM_PYTHON、TOKENIZER_MODEL、OUTPUT_ROOT
bash scripts/run_poker_oa_guidance.sh /path/to/oa_guidance.env
```

`OA_DATA_ROOT` 可以是训练入口，也可以是配置导出目录。部署只需要 OA 模型 checkpoint 和它匹配的配置；导出时不复制样本：

```bash
python scripts/touchguide/export_oa_inference_metadata.py \
  --data-root /path/to/output/poker_oa_dataset/ratio_1_1 \
  --output /path/to/oa_inference_metadata \
  --instruction 'Pick up the playing card and hold its face toward the robot.'
```

launcher 默认 `EXECUTE_STEPS=5`：每轮模型仍预测 30 步，仿真执行前 5 步后读取新的图像、state 与触觉并重规划。10 个 flow denoising steps 是单次推理内部的生成步数，与 30 步动作 horizon、执行前缀长度不同。该模式未启用异步推理或 RTC。

快速仿真参数在 [`configs/oa_guidance.env.example`](configs/oa_guidance.env.example)。真实评测应保存视频/遥测，并按照约定的三阶段标准人工/脚本复核；一个短 rollout smoke test 只能验证接线、输入输出和重规划节奏，不能证明打分器提升成功率。

## 许可和引用

OpenPI 原项目许可见 `LICENSE`、`LICENSE_GEMMA.txt`；捆绑仿真包使用 `simulation/LICENSE`。DINOv2、Transformers、PyTorch、MuJoCo 等依赖遵守其各自许可。π0.5 基础模型和预训练权重不随代码分发。
