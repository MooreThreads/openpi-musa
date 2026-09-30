# OpenPI：MUSA 适配与使用说明

本目录是在 OpenPI 基础上整理的 PyTorch/MUSA 训练适配版本，面向 Moore Threads S5000 和 `torch_musa` 环境。当前重点是 π₀/π₀.₅ 的 PyTorch 训练、LeRobot 数据读取、MUSA 单机/双机 DDP，以及 LIBERO 数据集训练。

本说明以 `openpi-train` 容器为主要运行环境。仓库只保存代码和 Git 元数据；LIBERO/ALOHA 子模块工作目录可以按需重新拉取，训练数据和模型权重不随代码仓库提供。

## 1. 运行环境

已核对的 `worker36088` 运行环境如下：

```text
容器：openpi-train
镜像：openpi-train:migrated-20260903
代码目录：/workspace/openpi
Python：3.10.12
PyTorch：2.7.1
torch_musa：2.7.1+84708cc
MUSA SDK：4.3.5
MCCL：2.11.4
驱动：5.2.0-server
GPU：Moore Threads S5000，8 卡
```

MUSA 环境检查：

```bash
docker exec -it openpi-train bash
cd /workspace/openpi

python - <<'PY'
import torch
import torch_musa

print("torch:", torch.__version__)
print("musa_available:", torch.musa.is_available())
print("musa_device_count:", torch.musa.device_count())
PY
```

代码使用 `torch.musa.is_available()` 判断 MUSA，不依赖可能不存在的 `torch.backends.musa` 属性。

## 2. 代码和依赖

如果需要 LIBERO/ALOHA 的仿真代码或资源，先初始化子模块：

```bash
git submodule update --init --recursive
```

本地只需要 OpenPI 主代码时，可以不初始化子模块。主仓库仍会保留 `.gitmodules` 和子模块提交引用。

使用 `uv` 安装 Python 依赖：

```bash
uv sync
uv pip install -e .
```

在容器中运行时，优先使用容器已有的 Python、PyTorch、`torch_musa` 和 MCCL 环境，不要在宿主机直接运行 MUSA 训练脚本。

## 3. 数据、权重和输出目录

双机启动脚本 [`tools/launch_musa_2node.sh`](tools/launch_musa_2node.sh) 的默认路径如下：

| 项目 | 默认路径 |
|---|---|
| OpenPI 代码 | `/workspace/openpi` |
| LIBERO LeRobot 数据 | `/data01e_biaozhun/fastwam/datasets/libero_mujoco3.3.2/libero_10_no_noops_lerobot` |
| PyTorch 基础权重 | `/workspace/openpi/checkpoints/pi05_base_pytorch` |
| 权重文件 | `/workspace/openpi/checkpoints/pi05_base_pytorch/model.safetensors` |
| 归一化统计 | `/workspace/openpi/assets/pi05_libero/libero_10_no_noops_lerobot/norm_stats.json` |
| 双机输出 | `/workspace/openpi/runs/2node` |

训练前必须在运行训练的每个节点确认数据、权重和归一化统计存在：

```bash
test -f /workspace/openpi/checkpoints/pi05_base_pytorch/model.safetensors
test -f /workspace/openpi/assets/pi05_libero/libero_10_no_noops_lerobot/norm_stats.json
test -d /data01e_biaozhun/fastwam/datasets/libero_mujoco3.3.2/libero_10_no_noops_lerobot
```

`pi05_libero` 配置仍保留上游数据集和权重配置；使用本地 MUSA 数据时，应通过命令行覆盖 `--data.repo-id`、`--pytorch-weight-path` 和 assets 路径。

## 4. 单机 MUSA 训练

单机 8 卡示例：

```bash
cd /workspace/openpi

MUSA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone \
  --nnodes=1 \
  --nproc_per_node=8 \
  scripts/train_pytorch.py pi05_libero \
  --exp-name=pi05_libero_musa_smoke \
  --pytorch-weight-path=/workspace/openpi/checkpoints/pi05_base_pytorch \
  --data.repo-id=/data01e_biaozhun/fastwam/datasets/libero_mujoco3.3.2/libero_10_no_noops_lerobot \
  --data.assets.assets-dir=/workspace/openpi/assets/pi05_libero \
  --data.assets.asset-id=libero_10_no_noops_lerobot \
  --batch-size=32 \
  --num-workers=0 \
  --num-train-steps=1000 \
  --log-interval=10 \
  --save-interval=100 \
  --checkpoint-base-dir=/workspace/openpi/runs/musa_single \
  --no-wandb-enabled
```

`--batch-size` 是所有 rank 的总 batch。8 卡时每个进程的 local batch 为 `batch_size / 8`，因此总 batch 必须能被进程数整除。

训练脚本会自动选择：

| 设备 | DDP backend |
|---|---|
| CUDA | `nccl` |
| MUSA | `mccl` |
| CPU | `gloo` |

当前 PyTorch 训练入口执行全参数微调：模型参数全部交给 AdamW，并在启动时检查所有参数均为 `requires_grad=True`。它不是 LoRA 或部分参数微调入口。

## 5. 双机 MUSA 训练

脚本 [`tools/launch_musa_2node.sh`](tools/launch_musa_2node.sh) 默认配置：

```text
NNODES=2
NPROC_PER_NODE=8
MASTER_ADDR=10.12.11.143
MASTER_PORT=29555
NODE_RANK=0：主节点 143
NODE_RANK=1：节点 168
MUSA DDP backend：`mccl`
NUM_TRAIN_STEPS=1000
BATCH_SIZE=32
```

两台机器的容器都需要已经启动，并且代码、数据、权重路径可访问。分别在两个容器中执行：

```bash
# 节点 143
cd /workspace/openpi
NODE_RANK=0 ./tools/launch_musa_2node.sh
```

```bash
# 节点 168
cd /workspace/openpi
NODE_RANK=1 ./tools/launch_musa_2node.sh
```

常用覆盖参数：

```bash
NODE_RANK=0 \
MASTER_ADDR=10.12.11.143 \
MASTER_PORT=29555 \
EXP_NAME=pi05_libero_2node \
BATCH_SIZE=32 \
NUM_TRAIN_STEPS=1000 \
SAVE_INTERVAL=100 \
./tools/launch_musa_2node.sh
```

脚本启动前会检查 MUSA 设备数、基础权重和 `norm_stats.json`。它只负责提交 `torchrun`，不能用进程启动成功代替训练成功验证。

## 6. LeRobot 数据适配

当前 [`src/openpi/training/data_loader.py`](src/openpi/training/data_loader.py) 做了以下兼容：

- 同时兼容旧版 `lerobot.common.datasets` 和新版 `lerobot.datasets` 导入路径。
- `--data.repo-id` 如果是本地目录，会作为 LeRobot 的 `root` 使用，目录名作为 dataset ID。
- 视频解码固定使用 `PyAV`，避免当前 MUSA 容器中的 `torchcodec` 兼容性问题。
- `actions` 会映射为新版 LeRobot 使用的 `action` 字段。

LIBERO 字段映射在 [`src/openpi/training/config.py`](src/openpi/training/config.py) 中定义：

```text
observation/image       -> observation.images.image
observation/wrist_image -> observation.images.wrist_image
observation/state       -> observation.state
actions                 -> action
prompt                  -> prompt
```

如果出现字段不存在错误，先检查数据集的 `meta/info.json`、episode 字段和实际视频/状态字段，再调整 `LeRobotLiberoDataConfig`，不要只修改命令行路径。

## 7. 权重加载和 checkpoint

MUSA 下 safetensors 不接受 `musa:*` 设备字符串，代码采用两阶段加载：

```text
safetensors -> CPU state_dict -> MUSA 模型
```

如果加载时出现 missing/unexpected keys，当前实现会记录 warning，而不是立即退出；正式训练前必须确认这些 key 不影响模型结构。

训练 checkpoint 默认包含：

```text
<checkpoint_base_dir>/<config_name>/<exp_name>/<step>/
├── model.safetensors
├── optimizer.pt
├── metadata.pt
└── assets/<asset_id>/norm_stats.json   # 若数据配置提供
```

使用 `--resume` 时，训练入口会加载该实验目录下最新的数字 step。最终 step 也会保存 checkpoint；保存动作由 rank 0 执行。

## 8. MUSA Flash SDPA

Gemma PyTorch attention 支持通过环境变量选择 MUSA Flash SDPA：

```bash
export OPENPI_USE_MUSA_FLASH=1
```

当前实现只有在以下条件同时满足时才启用：

- 当前设备是 MUSA；
- `head_dim` 为 64 或 128；
- `OPENPI_USE_MUSA_FLASH=1`。

其它情况退回 eager attention。默认值为关闭：

```bash
unset OPENPI_USE_MUSA_FLASH
```

Flash 路径与 muDNN/torch_musa 版本相关，开启前应单独做有限输入、有限 mask 和有限步数验证；不要仅凭进程启动或端口监听判断 Flash 训练成功。

## 9. 训练检查

### 9.1 环境和进程

```bash
docker top openpi-train -eo pid,args
mthreads-gmi -q -d MEMORY
```

### 9.2 日志和最终 step

```bash
# 如果后台启动时将 stdout/stderr 重定向到日志文件：
tail -f /workspace/openpi/runs/2node/*.log

find /workspace/openpi/runs/2node \
  -maxdepth 6 -type f \( -name 'model.safetensors' -o -name 'metadata.pt' \) -ls
```

前台启动时直接观察两个容器的终端输出；`launch_musa_2node.sh` 本身不会自动创建日志文件。

至少确认：

1. 每个节点的 rank 数量与 `NPROC_PER_NODE` 一致；
2. 日志出现 `world_size`、loss、learning rate 和最终 step；
3. `model.safetensors`、`optimizer.pt`、`metadata.pt` 均存在；
4. 最终 checkpoint 能在目标 MUSA 设备上重新加载；
5. loss、grad norm 和模型输出均为有限值。

## 10. 常见问题

| 问题 | 检查或处理方式 |
|---|---|
| `torch.musa.is_available()` 为 False | 在 `openpi-train` 容器内检查 MUSA 设备挂载、`torch_musa` 版本和 `MUSA_VISIBLE_DEVICES` |
| `mccl` 初始化失败 | 确认两台机器的 `MASTER_ADDR`、`MASTER_PORT`、`NODE_RANK`、`NNODES` 和网卡连通性一致 |
| `torchcodec` 导入或解码失败 | 使用本适配中的 `video_backend="pyav"`，不要强制切回 torchcodec |
| `LeRobotDataset` 找不到本地数据 | `--data.repo-id` 应指向包含 `meta/`、`data/`、`videos/` 的数据集根目录 |
| safetensors 报 MUSA 设备错误 | 使用当前代码的 CPU state_dict 加载分支，不要直接传 `device="musa:0"` |
| MUSA Flash 输出 NaN 或异常 | 先取消 `OPENPI_USE_MUSA_FLASH`，使用 eager attention，再单独定位运行库和 mask 问题 |
| checkpoint 缺少最终 step | 检查 rank 0 日志、`save_interval`、输出目录权限和 `model.safetensors` 是否生成 |
| 子模块目录为空 | 执行 `git submodule update --init --recursive`；子模块不是主仓库普通目录 |

## 11. 当前边界

- 本适配主要覆盖 PyTorch π₀/π₀.₅ 训练；JAX 训练路径仍按上游实现使用。
- 双机脚本中的 IP、数据路径、权重路径和输出路径是当前环境默认值，换服务器必须显式覆盖。
- 代码仓库不包含训练权重、训练结果或 LIBERO/ALOHA 子模块工作目录。
- PyTorch 入口当前按全参数训练设计，不提供 LoRA 或参数冻结配置。
- MUSA Flash SDPA 是显式可选路径，默认关闭；完整长时训练和真实数据验证需要单独记录日志、loss、checkpoint 和有限值结果。

## License

本项目沿用 OpenPI 上游许可和各子模块的原有许可。主项目许可见 [LICENSE](LICENSE)，Gemma 相关许可见 [LICENSE_GEMMA.txt](LICENSE_GEMMA.txt)。
