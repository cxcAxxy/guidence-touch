三个规模的全帧接触配对数据

运行：bash /path/to/guidence-touch/scripts/prepare_poker_contact_scales.sh
默认根目录：/path/to/datasets/touchguidence/augmentation/poker_fingertip_recovery_v2_random10

clean/ 保存所有有接触且具备完整 30 步动作的原始配对；沿用原始 180/20 条示范的训练/验证拆分。
augmented/ 每个起点保存 10 条不重复的有效扰动观测—30 步平滑恢复动作配对。
geometry/ 保存接触方向与源数据的关联依据。
ratio_1_1/、ratio_1_5/、ratio_1_10/ 是三个独立的数据集入口，分别采用每个起点的前 1/5/10 个随机变体，彼此嵌套。
三个入口使用 HDF5 Virtual Dataset 引用共享的 clean/ 与 augmented/ 文件。不要单独搬运这些入口；当前引用为绝对路径，搬迁根目录后应重新生成视图。视觉仍引用原始 NAS 视频，不复制视频。

查看状态：cat 根目录/progress.json
完成状态：progress.json 的 status 为 complete；summary.json 给出实际条数、H5 字节数和三个比例的拆分统计。
不完整的起点记录在 augmented/completed/episode_*.json 的 shortfalls 中；不足 10 个变体时不会声称比例入口已完成。
生成器支持同一配置下按已完成 episode 续跑，使用 run.lock 防止重复启动；不续跑此前已终止的全组合版本。

扰动只作用于可靠接触的手指；手臂不变。指尖沿接触法向与两个切向扰动，优先 0.5/1/2 mm，离散有效候选不足时在 0.5–2 mm 内连续重抽。多个手指可以使用不同方向。
IK 验证关节范围、指尖误差和触觉垫旋转。恢复动作采用 30 步最小跃度轨迹，从扰动关节位置返回原始当前状态；并非经过仿真执行验证的恢复成功轨迹。
触觉由局部接触近似合成，不是物理真值；保留原始视觉是局部近似。
state 为绝对关节位置；动作中手臂为相对当前 state 的增量、手为绝对目标。沿用 pi05 20000 步的归一化统计，不截断；动作噪声增强留给训练读取阶段。

CPMDataset 可以读取三个入口。训练接入仍需配置：同一全局 batch 内，每个原始起点 episode_index+frame_index 只出现一次，避免干净样本和同源变体互为假负样本。不要直接使用旧数据的固定验证索引或负样本索引。
生成三个数据规模不等于每个训练 batch 强制为这些比例；训练采样比例、统一验证方案和公平的训练预算应另行配置。
此脚本只生成数据，不启动训练。
