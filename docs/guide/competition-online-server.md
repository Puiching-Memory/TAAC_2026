---
icon: lucide/server
---

# 线上运行环境速查

本页整理已经从真实线上任务日志里验证过的环境事实，用来指导 bundle 设计和线上排障。它不是平台 SLA；如果平台镜像或资源策略变化，应以新的任务日志为准。

采样日志来自 2026-04-25、2026-04-26 和 2026-04-27 的线上任务启动记录。

## 最重要的结论

- 线上默认 Python 是 `/opt/conda/envs/competition/bin/python3`，版本 `3.10.20`。
- 平台预装 CUDA 12.6 相关 PyTorch 栈，包括 `torch==2.7.1+cu126`、`torchrec==1.2.0+cu126` 和 `fbgemm_gpu==1.2.0+cu126`。
- 任务启动时没有可靠的 `uv`，训练和推理 bundle 应使用 `TAAC_RUNNER=python`。
- 公共 PyPI、Astral、GitHub、PyTorch 官方索引和 conda-forge 不应作为任务启动依赖。
- Tencent PyPI / Anaconda 镜像在继承平台代理时可达。
- 可见 GPU 是约 `0.2x NVIDIA H20`，可用显存约 19.6 GiB；不要按整卡 H20 设计 batch size。
- `gcc`、`g++`、`make`、`nvcc` 可用；`cmake`、`ninja`、`pkg-config` 缺失。

## 对训练配置的影响

按小显存 GPU 设计默认值：

- 优先启用 AMP，再逐步调 batch size。
- OOM 时先降 batch size、序列长度、模型宽度和 `num_workers`。
- 不要在任务启动阶段覆盖安装核心 GPU 包。
- 需要原生构建的依赖尽量提前解决，不要指望线上临时编译复杂 CUDA / C++ 包。

一个更接近线上使用的启动方式：

```bash
export TAAC_RUNNER=python
export TAAC_PYTHON=/opt/conda/envs/competition/bin/python3
export TRAIN_DATA_PATH=/path/to/train.parquet_or_dataset_dir
export TAAC_SCHEMA_PATH=/path/to/schema.json
export TRAIN_CKPT_PATH=/path/to/output

bash run.sh --device cuda
```

## 网络和依赖

日志中平台注入的是小写代理变量：

```text
http_proxy=http://21.100.120.217:3128
https_proxy=http://21.100.120.217:3128
```

实践上要记住两点：

- 不要无条件 `unset http_proxy https_proxy`，否则会一起关掉 Tencent 镜像通路。
- 不要在线安装 `uv` 或重建 CUDA / PyTorch 栈。

缺少纯 Python 包时，可以优先尝试平台可达镜像：

```bash
python -m pip install -i https://mirrors.cloud.tencent.com/pypi/simple/ <package>
```

## 平台资源概览

| 项目          | 观测值                                             |
| ------------- | -------------------------------------------------- |
| 操作系统      | Ubuntu 22.04.5 LTS                                 |
| Python        | `/opt/conda/envs/competition/bin/python3`，3.10.20 |
| CUDA Runtime  | 12.6                                               |
| NVIDIA Driver | 535.247.01                                         |
| nvcc          | 12.6.77                                            |
| GPU           | `0.2x NVIDIA H20`                                  |
| 可见显存      | 约 19574 MiB                                       |
| 内存          | 日志中约 2.2 TiB，无 swap                          |

CPU、总内存和磁盘看起来宽裕，但作业实际可用资源仍可能受平台调度影响。训练排障时以当前任务日志为准。

## 排障速查

| 现象                     | 优先检查                                      |
| ------------------------ | --------------------------------------------- |
| `uv: command not found`  | bundle 是否仍在用 `TAAC_RUNNER=uv`            |
| 依赖安装失败             | 是否访问了公共 PyPI / Astral / PyTorch 索引   |
| 训练 OOM                 | batch size、序列长度、模型宽度、`num_workers` |
| Torch / CUDA 版本冲突    | 是否在线覆盖了平台预装 GPU 栈                 |
| 找不到 `cmake` / `ninja` | 是否把复杂原生构建留到了线上启动阶段          |
| 推理找不到缓存目录       | 显式设置 `TAAC_BUNDLE_WORKDIR`                |

## 采集新证据

需要更新这页时，先在线上跑维护实验：

```bash
uv run taac-package-train \
  --experiment experiments/host_device_info \
  --output-dir outputs/bundles/host_device_info
```

上传后执行训练 bundle，把 stdout 作为原始材料，再把稳定结论整理回本页。

采集日志至少要保留这些信息：

- 任务日期和平台任务 ID。
- Python 可执行文件与版本。
- `pip list` 里核心 GPU 包版本。
- `nvidia-smi` 摘要。
- 代理变量是否存在，以及 Tencent 镜像探测结果。
- `gcc`、`g++`、`make`、`nvcc`、`cmake`、`ninja`、`pkg-config` 是否可用。

维护实验入口和采集逻辑分别在 `experiments/host_device_info/__init__.py` 与 `experiments/host_device_info/runner.py`。bundle 运行时选择 runner 的逻辑在 `src/taac2026/infrastructure/platform/env.py`，依赖安装逻辑在 `src/taac2026/infrastructure/platform/deps.py`。
