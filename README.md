# OpenPI：MUSA 适配与训练说明

本仓库是在 OpenPI 基础上整理的 PyTorch/MUSA 训练适配版本，主要覆盖 π₀/π₀.₅、LeRobot 数据读取、单机/多机 MUSA DDP，以及 LIBERO 数据训练。

仓库只保存代码和配置，不包含训练数据、基础权重和训练结果。本文所有命令都从仓库根目录执行，并使用相对路径，便于迁移到不同机器和目录。

## 1. 兼容环境

代码已在以下两组 MUSA PyTorch 运行时完成训练短测：

| Python | PyTorch | torch_musa | 状态 |
|---|---|---|---|
| 3.10.12 | 2.7.1 | 2.7.1 | 已验证 |
| 3.10.12 | 2.11.0.post1 | 2.11.0.post1 | 已验证 |

两组环境共同验证过的主要上层依赖为：

```text
transformers 4.53.2
LeRobot 0.4.4
PyAV 15.1.0
```

`torch` 和 `torch_musa` 必须使用彼此匹配的版本。不要只升级其中一个，也不要用普通 PyTorch wheel 覆盖设备环境提供的 MUSA wheel。

运行前检查：

```bash
python - <<'PY'
import importlib.metadata as metadata
import torch
import torch_musa

for package in ("torch", "torch_musa", "transformers", "lerobot", "av"):
    print(f"{package}: {metadata.version(package)}")

print("musa_available:", torch.musa.is_available())
print("musa_device_count:", torch.musa.device_count())
assert torch.musa.is_available()
assert torch.musa.device_count() > 0
PY
```

代码通过 `torch.musa.is_available()` 检测 MUSA，不依赖可能不存在的 `torch.backends.musa`。

### 1.1 使用源码时保留设备运行时

`pyproject.toml` 和 `uv.lock` 中的 `torch==2.7.1` 是依赖基线，不代表源码的版本上限。在已经配置好的 torch 2.11 MUSA 环境中，不要直接用 lockfile 把 `torch/torch_musa` 降级或替换。

仓库元数据声明 Python ≥3.11，但已验证的设备运行时为 Python 3.10.12。对已经准备好的 MUSA 环境，建议不重新解析项目 lockfile，直接通过 `PYTHONPATH` 使用源码：

```bash
export PYTHONPATH="$PWD/src:$PWD/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
```

设置后重新执行上面的版本、import 和 MUSA 可用性检查。全新环境应先按照设备运行时要求安装一组匹配的 `torch/torch_musa`，再安装其余依赖。

如果需要 LIBERO/ALOHA 的仿真代码或资源：

```bash
git submodule update --init --recursive
```

只运行 OpenPI 主训练代码时，可以不初始化未使用的子模块。

## 2. 目录约定

本文假设仓库结构如下：

```text
openpi/
├── libero_10_no_noops_lerobot/       # LeRobot 数据集根目录，也可以是相对软链接
│   ├── meta/
│   ├── data/
│   └── videos/
├── checkpoints/
│   └── pi05_base_pytorch/
│       └── model.safetensors
├── assets/
│   └── pi05_libero/
│       └── libero_10_no_noops_lerobot/
│           └── norm_stats.json
└── runs/
```

以下变量在后续示例中复用：

```bash
DATASET=libero_10_no_noops_lerobot
WEIGHT_PATH=checkpoints/pi05_base_pytorch
ASSETS_DIR=assets/pi05_libero
ASSET_ID=libero_10_no_noops_lerobot
OUTPUT_DIR=runs
```

如果实际数据保存在仓库外，可以在仓库根目录创建相对软链接，使 `DATASET` 仍保持稳定：

```bash
ln -s ../datasets/libero_10_no_noops_lerobot "$DATASET"
```

训练前检查：

```bash
test -f "$DATASET/meta/info.json"
test -d "$DATASET/data"
test -d "$DATASET/videos"
test -f "$WEIGHT_PATH/model.safetensors"
```

## 3. LIBERO 数据字段

当前 `LeRobotLiberoDataConfig` 期望数据集中包含：

```text
observation.images.image
observation.images.wrist_image
observation.state
action
```

训练前可检查 `meta/info.json`：

```bash
python - <<'PY'
import json

path = "libero_10_no_noops_lerobot/meta/info.json"
info = json.load(open(path))
features = set(info["features"])

required = {
    "observation.images.image",
    "observation.images.wrist_image",
    "observation.state",
    "action",
}

missing = required - features
if missing:
    raise RuntimeError(f"Missing dataset fields: {sorted(missing)}")

print("Dataset fields validated")
print("frames:", info.get("total_frames"))
print("episodes:", info.get("total_episodes"))
print("tasks:", info.get("total_tasks"))
print("fps:", info.get("fps"))
PY
```

OpenPI 会通过 Repack 将 LeRobot 字段适配成模型字段：

```text
observation.images.image       -> observation/image
observation.images.wrist_image -> observation/wrist_image
observation.state              -> observation/state -> state
action                         -> actions
task                           -> prompt
```

`RepackTransform` 只负责字段适配，不应改变 state/action 数值。

## 4. 使用官方脚本生成 norm stats

不要直接复制 LeRobot 的 `meta/stats.json`，也不要手工创建 `norm_stats.json`。应使用仓库官方脚本：

[`scripts/compute_norm_stats.py`](scripts/compute_norm_stats.py)

官方脚本执行的处理顺序为：

```text
LeRobot sample
→ RepackTransform
→ LiberoInputs
→ 统计 state/actions
→ 写入 norm_stats.json
```

### 4.1 配置目标数据集

运行脚本前，确认 `pi05_libero` 对应的 `LeRobotLiberoDataConfig.repo_id` 是：

```python
repo_id="libero_10_no_noops_lerobot"
```

这个值与本文假设的数据集相对目录一致。官方脚本会使用配置中的 `repo_id` 读取数据，并按下面的路径保存结果：

```text
assets/<config_name>/<repo_id>/norm_stats.json
```

因此本例输出为：

```text
assets/pi05_libero/libero_10_no_noops_lerobot/norm_stats.json
```

### 4.2 先做小规模检查

```bash
python scripts/compute_norm_stats.py \
  --config-name pi05_libero \
  --max-frames 1024
```

`--max-frames` 仅用于确认数据、视频和 transform 能正常读取。小规模统计不能作为正式训练 asset。

### 4.3 生成完整统计量

删除小规模输出后，对完整数据集重新计算：

```bash
rm -f "$ASSETS_DIR/$ASSET_ID/norm_stats.json"

python scripts/compute_norm_stats.py \
  --config-name pi05_libero
```

脚本完成后应生成：

```text
assets/pi05_libero/libero_10_no_noops_lerobot/norm_stats.json
```

如果使用独立的、与 lockfile 匹配的 Python ≥3.11 项目环境，也可以使用：

```bash
uv run scripts/compute_norm_stats.py \
  --config-name pi05_libero
```

不要为了运行统计脚本而覆盖现有 MUSA 环境中的 `torch/torch_musa`。

### 4.4 校验生成结果

```bash
python - <<'PY'
import json
import math

path = (
    "assets/pi05_libero/"
    "libero_10_no_noops_lerobot/"
    "norm_stats.json"
)

stats = json.load(open(path))["norm_stats"]
expected = {"state", "actions"}

if set(stats) != expected:
    raise RuntimeError(
        f"Invalid norm stats keys: {sorted(stats)}; "
        f"expected={sorted(expected)}"
    )

assert len(stats["state"]["mean"]) == 8
assert len(stats["actions"]["mean"]) == 7

for key in expected:
    for field in ("mean", "std", "q01", "q99"):
        values = stats[key][field]
        assert values
        assert all(math.isfinite(value) for value in values)

print("norm stats validated:", sorted(stats))
PY
```

正确 key 必须是：

```text
state
actions
```

下面这种 raw dataset key 不适用于当前 Normalize 阶段：

```text
observation.state
action
```

由于 Normalize 默认允许缺失 key，错误 key 可能导致归一化被静默跳过，所以正式训练前必须执行上述校验。

## 5. 单机 MUSA 训练

先做短测：

```bash
MTHREADS_VISIBLE_DEVICES=0,1 \
MUSA_VISIBLE_DEVICES=0,1 \
python -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=2 \
  scripts/train_pytorch.py pi05_libero \
  --exp-name pi05_libero_musa_smoke \
  --pytorch-weight-path "$WEIGHT_PATH" \
  --data.repo-id "$DATASET" \
  --data.assets.assets-dir "$ASSETS_DIR" \
  --data.assets.asset-id "$ASSET_ID" \
  --batch-size 64 \
  --num-workers 0 \
  --num-train-steps 10 \
  --log-interval 1 \
  --save-interval 10 \
  --checkpoint-base-dir "$OUTPUT_DIR/musa_single" \
  --no-wandb-enabled
```

短测通过后再扩展到8卡：

```bash
MTHREADS_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MUSA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=8 \
  scripts/train_pytorch.py pi05_libero \
  --exp-name pi05_libero_musa \
  --pytorch-weight-path "$WEIGHT_PATH" \
  --data.repo-id "$DATASET" \
  --data.assets.assets-dir "$ASSETS_DIR" \
  --data.assets.asset-id "$ASSET_ID" \
  --batch-size 256 \
  --num-workers 16 \
  --num-train-steps 30000 \
  --log-interval 100 \
  --save-interval 2000 \
  --checkpoint-base-dir "$OUTPUT_DIR/musa_single"
```

`--batch-size` 是所有 rank 的 global batch。它必须能被总进程数整除。8卡 global batch 256 对应每卡 local batch 32。

训练入口按设备选择通信后端：

| 设备 | DDP backend |
|---|---|
| CUDA | `nccl` |
| MUSA | `mccl` |
| CPU | `gloo` |

当前 PyTorch 训练入口执行全参数微调，不是 LoRA 或部分参数微调。

## 6. 双机 MUSA 训练

两台机器需要具有相同的相对目录结构，并能访问各自的：

```text
libero_10_no_noops_lerobot/
checkpoints/pi05_base_pytorch/
assets/pi05_libero/libero_10_no_noops_lerobot/
```

两台机器设置相同的主节点地址和端口，只改变 `NODE_RANK`：

```bash
NNODES=2
NPROC_PER_NODE=8
MASTER_ADDR=<主节点可达地址>
MASTER_PORT=29555
```

主节点：

```bash
NODE_RANK=0 \
python -m torch.distributed.run \
  --nnodes "$NNODES" \
  --nproc-per-node "$NPROC_PER_NODE" \
  --node-rank "$NODE_RANK" \
  --master-addr "$MASTER_ADDR" \
  --master-port "$MASTER_PORT" \
  scripts/train_pytorch.py pi05_libero \
  --exp-name pi05_libero_2node \
  --pytorch-weight-path "$WEIGHT_PATH" \
  --data.repo-id "$DATASET" \
  --data.assets.assets-dir "$ASSETS_DIR" \
  --data.assets.asset-id "$ASSET_ID" \
  --batch-size 512 \
  --num-workers 8 \
  --num-train-steps 1000 \
  --log-interval 10 \
  --save-interval 100 \
  --checkpoint-base-dir "$OUTPUT_DIR/musa_2node" \
  --no-wandb-enabled
```

第二个节点运行相同命令，只设置：

```bash
NODE_RANK=1
```

进程启动不等于训练成功。必须确认所有 rank 加入同一个 world、loss/grad norm 为有限值，并且最终 checkpoint 完整写出。

## 7. 权重加载与 checkpoint

MUSA 下 safetensors 不能直接接收 `musa:*` 设备字符串。当前代码使用：

```text
safetensors → CPU state_dict → MUSA 模型
```

checkpoint 目录结构：

```text
<checkpoint_base_dir>/<config_name>/<exp_name>/<step>/
├── model.safetensors
├── optimizer.pt
├── metadata.pt
└── assets/<asset_id>/norm_stats.json
```

使用 `--resume` 时会加载实验目录下最新的数字 step。恢复训练前必须确认 checkpoint 中的 norm stats 与原训练保持一致；不能用修复过的归一化配置直接恢复旧的未归一化 checkpoint。

## 8. MUSA Flash SDPA

Gemma attention 可以通过环境变量选择 MUSA Flash SDPA：

```bash
export OPENPI_USE_MUSA_FLASH=1
```

只有当前设备为 MUSA 且 `head_dim` 为64或128时才尝试 Flash 路径。默认关闭：

```bash
unset OPENPI_USE_MUSA_FLASH
```

Flash 路径对 `torch_musa`、muDNN、mask 和 dtype 更敏感。正式启用前至少验证：

- 输入、attention 输出、loss 和 gradient 都是有限值；
- 全 mask 行不会产生 NaN；
- 固定 batch 的 eager/Flash 输出误差在可接受范围内；
- 短测能够正常保存并重新加载 checkpoint。

## 9. 训练验收

训练时至少确认：

1. rank 数量、world size 和每卡 local batch 正确；
2. `torch.musa.is_available()` 为 True；
3. 日志持续输出有限的 loss、learning rate 和 grad norm；
4. DataLoader 实际读取图像、state、action 和 prompt；
5. `norm_stats.json` key 为 `state/actions`；
6. state/actions 在 Normalize 后确实发生数值变化；
7. `model.safetensors`、`optimizer.pt` 和 `metadata.pt` 均存在；
8. 最终 checkpoint 能在目标 MUSA 环境中重新加载；
9. 训练进程正常退出，没有 NaN、OOM 或 traceback。

查看输出：

```bash
find "$OUTPUT_DIR" \
  -type f \
  \( -name model.safetensors -o -name optimizer.pt -o -name metadata.pt \) \
  -print
```

## 10. 常见问题

| 问题 | 检查或处理方式 |
|---|---|
| `torch.musa.is_available()` 为 False | 检查设备运行时、`torch/torch_musa` 配对和可见设备变量 |
| 安装依赖后 MUSA 不可用 | 检查是否被普通 PyTorch wheel 覆盖；恢复匹配的 MUSA wheel |
| `mccl` 初始化失败 | 核对 `MASTER_ADDR`、`MASTER_PORT`、`NODE_RANK`、`NNODES` 和网络连通性 |
| 找不到本地数据 | `--data.repo-id` 应指向含 `meta/`、`data/`、`videos/` 的数据集根目录 |
| 图像字段不存在 | 检查 `meta/info.json` 与 `LeRobotLiberoDataConfig` 的 Repack mapping |
| norm stats 存在但 loss 异常偏低 | 检查 key 是否为 `state/actions`，并验证 Normalize 前后数值 |
| safetensors 报 MUSA 设备错误 | 使用当前 CPU state_dict 中转加载逻辑，不要直接传 `device="musa:0"` |
| Flash attention 出现 NaN | 取消 `OPENPI_USE_MUSA_FLASH`，先回退 eager attention |
| checkpoint 缺少最终 step | 检查 rank 0 日志、输出权限和进程退出状态 |

## 11. 当前边界

- 本适配主要覆盖 PyTorch π₀/π₀.₅ 训练，JAX 训练路径仍按上游实现使用。
- PyTorch入口当前按全参数训练设计，不提供 LoRA 或参数冻结配置。
- MUSA Flash SDPA 是显式可选路径，默认关闭。
- 版本兼容结论来自已完成的训练短测；更换 MUSA SDK、驱动、muDNN 或 MCCL 后仍需重新验证。
- 仓库不包含数据、基础权重、训练结果和未初始化的子模块内容。

## License

本项目沿用 OpenPI 上游许可和各子模块原有许可。主项目许可见 [LICENSE](LICENSE)，Gemma相关许可见 [LICENSE_GEMMA.txt](LICENSE_GEMMA.txt)。
