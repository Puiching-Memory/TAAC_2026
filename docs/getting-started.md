---
icon: lucide/rocket
---

# 快速开始

## 前置要求

- Linux 环境，推荐 Ubuntu 24.04
- Python 3.13（仓库支持范围为 3.12-3.13）
- [uv](https://docs.astral.sh/uv/) 包管理器
- NVIDIA GPU + CUDA 12.8 驱动栈用于完整训练与 GPU 测试

!!! important "平台支持"
    仓库只支持 Linux 运行时。Windows 原生环境不再是支持目标；如果你在 Windows 上开发，请使用 WSL2 + Docker 并通过 devcontainer 进入 Linux 容器。完整步骤见 [开发容器 (WSL2 + Docker)](guide/devcontainer.md)。

## 安装

```bash
# 克隆仓库并同步依赖
git clone https://github.com/Puiching-Memory/TAAC_2026.git
cd TAAC_2026

# 安装并固定 Python 3.13
uv python install 3.13
uv sync --locked
```

!!! warning "关于 PyPI 镜像"
    仓库在 `pyproject.toml` 里固定了 canonical PyPI 作为默认索引，以保证 `uv.lock` 在本机和 CI 间一致。不要额外传 `--default-index` 或 `--index-url` 指向国内镜像，否则 `uv` 会判定 `uv.lock` 需要更新。  
    如果只是想加速下载，优先使用系统代理或透明代理。

### 使用 devcontainer

如果你通过 WSL2 + Docker 开发，可以直接在仓库根目录执行 VS Code 的 Reopen in Container。容器会使用 `.devcontainer/` 下的 CUDA 基础镜像，并在首次创建后自动运行：

```bash
bash .devcontainer/post-create.sh
```

该脚本会安装 Python 3.13、执行 `uv sync --locked`，并运行 GPU / TorchRec 自检脚本。

## 下载样例数据

实验包默认使用 HuggingFace 上的样例数据集 [`TAAC2026/data_sample_1000`](https://huggingface.co/datasets/TAAC2026/data_sample_1000)。框架会自动从本地 HuggingFace 缓存中解析数据路径。

如果 `data/datasets--TAAC2026--data_sample_1000/` 还不存在，第一次运行训练/评估命令时也会自动回退到该 HF 数据集名，并把下载内容写入父目录 `data/` 作为缓存根。

```bash
# 下载数据到仓库根下的 data/ 目录（作为 HF 缓存根）
uv run python -c "
from datasets import load_dataset
load_dataset('TAAC2026/data_sample_1000', cache_dir='data')
"
```

下载完成后，实验包会自动在 `data/datasets--TAAC2026--data_sample_1000/` 下解析 parquet 文件，优先使用 `refs/main` 指向的快照，无需手动指定版本哈希。

## 训练第一个模型

```bash
uv run taac-train --experiment config/baseline
```

训练完成后，产物会写入 `outputs/config/baseline/`：

```
outputs/config/baseline/
├── best.pt                 # 最佳 checkpoint（按 val AUC 选择）
├── summary.json            # 训练摘要（指标、超参数、耗时）
├── training_curves.json    # 逐 epoch 训练曲线
└── profiling/              # 模型参数量、FLOPs、推理延迟
```

如果你希望提前打开编译与 AMP，可以直接使用同一套 CLI：

```bash
uv run taac-train --experiment config/baseline --compile --amp --amp-dtype bfloat16
```

## 评估

```bash
# 评估默认输出目录中的 best.pt
uv run taac-evaluate single --experiment config/baseline

# 启用 CPU int8 动态量化推理
uv run taac-evaluate single --experiment config/baseline --quantize int8

# 导出当前评估图为 torch.export artifact
uv run taac-evaluate single --experiment config/baseline --export-mode torch-export

# 显式复用运行时优化配置
uv run taac-evaluate single --experiment config/baseline --compile --amp --amp-dtype bfloat16
```

评估报告包含：AUC、PR-AUC、Brier Score、LogLoss、GAUC，以及推理延迟（ms/sample）。

`--quantize int8` 当前会把推理切到 CPU，并对 `nn.Linear` 与 TorchRec `EmbeddingBagCollection` 应用动态 int8 量化；这条路径适合本地或 CI 的推理回归验证，不会修改训练 checkpoint 本身。

`--export-mode torch-export` 会基于当前评估 batch 生成 `.pt2` artifact，默认输出到评估输出目录，可用于后续 AOTI / runtime backend 集成验证。

## 超参数搜索

```bash
# 使用 Optuna 搜索 20 个 trial，自动按 GPU 显存并发调度
uv run taac-search --experiment config/baseline --trials 20
```

搜索默认约束：

- 模型参数量 ≤ 3 GiB
- 验证集端到端推理总时长 ≤ 180 秒

→ 详细搜索配置见 [超参数搜索](guide/search.md)

## 运行其他实验包

所有实验包使用相同的 CLI 接口，只需替换 `--experiment` 路径：

```bash
# 训练 InterFormer
uv run taac-train --experiment config/interformer

# 训练 OneTrans
uv run taac-train --experiment config/onetrans

# 训练 HyFormer
uv run taac-train --experiment config/hyformer
```

→ 完整实验包列表见 [实验包总览](experiments/index.md)

## 跑测试

```bash
# 完整回归
uv run pytest tests -q

# 快速单元测试
uv run pytest -m unit -q

# 本地 GPU 测试集合
uv run python scripts/run_gpu_tests.py
```

→ 详细测试指南见 [测试](guide/testing.md)

## 本地文档站

文档构建前需要先生成 gitignored 图表产物：

```bash
uv sync --locked --no-install-package torch --no-install-package torchrec --no-install-package fbgemm-gpu --no-install-package triton
uv run taac-dataset-eda
uv run taac-tech-timeline
uv run taac-bench-report
uv run --no-project --isolated --with zensical zensical build --clean
```

→ 详细步骤见 [本地生成站点](guide/local-site.md)

## CLI 命令速查

| 命令                          | 用途               |
| ----------------------------- | ------------------ |
| `taac-train`                  | 训练实验包         |
| `taac-evaluate`               | 评估 checkpoint    |
| `taac-search`                 | Optuna 超参数搜索  |
| `taac-bench-report`           | 生成 benchmark 图表 |
| `taac-plot-model-performance` | 生成性能对比图     |
| `taac-clean-pycache`          | 清理 `__pycache__` |
