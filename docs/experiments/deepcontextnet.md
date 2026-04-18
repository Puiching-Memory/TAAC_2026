---
icon: lucide/flask-conical
---

# DeepContextNet

**上下文感知深度建模**

## 概述

DeepContextNet 当前在仓库里的实现是标准 Transformer 风格的上下文序列建模实验包，不是框架级 HSTU 实现。它复用了默认数据管道、默认 ranking loss、TorchRec sparse embedding 路径，以及框架默认的混合优化器路由。

## 模型架构

- 4 层 Transformer，**8 头**注意力
- Embedding 维度 128
- Recent sequence length 32（最长）
- Batch size 32（最小，匹配更大模型的显存需求）
- 默认走框架 `FeatureSchema` + `TorchRecEmbeddingBagAdapter`
- 默认数据管道与默认 loss builder 已交给框架层处理

## 默认配置

| 参数              | 值   |
| ----------------- | ---- |
| `embedding_dim`   | 128  |
| `num_layers`      | 4    |
| `num_heads`       | 8    |
| `epochs`          | 10   |
| `batch_size`      | 32   |
| `learning_rate`   | 2e-4 |
| `pairwise_weight` | 0.0  |

## 快速运行

```bash
uv run taac-train --experiment config/gen/deepcontextnet
uv run taac-evaluate single --experiment config/gen/deepcontextnet
```

## 当前自定义部分

- `model.py`：保留 DeepContextNet 自己的序列建模块
- `__init__.py`：`build_data_pipeline=None`、`build_loss_stack=None`、`build_optimizer_component=None`，训练侧完全复用框架默认 builder
- `utils.py`：仅保留兼容性 helper，不再承载独立优化器实现

## 输出目录

```
outputs/gen/deepcontextnet/
```

## 来源

[suyanli220/TAAC-2026-Baseline-Tencent-Advertisement-Contest](https://github.com/suyanli220/TAAC-2026-Baseline-Tencent-Advertisement-Contest)
*** Add File: /desay120T/ct/dev/uid01954/TAAC_2026/docs/guide/devcontainer.md
---
icon: lucide/container
---

# 开发容器 (WSL2 + Docker)

## 适用场景

仓库的主支持路径是 Linux。Windows 用户应当通过 WSL2 + Docker Desktop + VS Code Dev Containers 进入 Linux 容器，而不是在 Windows 原生环境直接执行 `uv sync`。

当前仓库已经提供：

- `.devcontainer/devcontainer.json`
- `.devcontainer/Dockerfile`
- `.devcontainer/post-create.sh`
- `scripts/verify_gpu_env.py`

## 宿主机前置要求

### Windows

1. 安装 WSL2。
2. 安装 Docker Desktop，并启用 WSL integration。
3. 安装 NVIDIA Windows 驱动，确保 WSL2 可见 GPU。
4. 安装 VS Code 与 Dev Containers 扩展。

### Linux

1. 安装 Docker Engine。
2. 安装 NVIDIA Container Toolkit。
3. 确认 `docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi` 可以看到 GPU。

## 打开仓库

```bash
git clone https://github.com/Puiching-Memory/TAAC_2026.git
cd TAAC_2026
code .
```

在 VS Code 中执行 Reopen in Container。容器启动后会基于 CUDA 12.8 Ubuntu 24.04 镜像构建，并自动运行：

```bash
bash .devcontainer/post-create.sh
```

## 容器内初始化流程

`post-create.sh` 会依次执行：

1. `uv python install 3.13`
2. `uv sync --locked --python 3.13`
3. `uv run python scripts/verify_gpu_env.py --json`

第三步会验证：

- `torch` 导入
- `torchrec` 导入
- `fbgemm_gpu` 导入
- `triton` 导入
- CUDA 可见性
- 一个最小 TorchRec embedding probe

## 常用命令

```bash
# 完整测试
uv run pytest -q

# GPU 测试
uv run python scripts/run_gpu_tests.py

# 训练 baseline
uv run taac-train --experiment config/gen/baseline

# 重新检查环境链路
uv run python scripts/verify_gpu_env.py --json
```

## 常见问题

### 容器里看不到 GPU

- 确认 Docker 已启用 GPU 支持。
- 确认宿主机 `nvidia-smi` 正常。
- Windows 下确认 Docker Desktop 已打开 WSL2 integration。

### `uv sync --locked` 失败

- 不要额外传国内镜像参数。
- 确认网络可以访问 PyPI 和 PyTorch CUDA 128 index。
- 如果你改动了 `pyproject.toml`，记得同步更新 `uv.lock`。

### 只想看文档，不想装完整 CUDA 训练栈

可以在宿主机或容器内执行文档轻量安装路径：

```bash
uv sync --locked --no-install-package torch --no-install-package torchrec --no-install-package fbgemm-gpu
uv run --no-project --isolated --with zensical zensical build --clean
```
