---
icon: lucide/rocket
---

# 快速开始

## 前置要求

- Linux 环境，推荐 Ubuntu 24.04
- Python 3.13（仓库支持范围为 3.10-3.13）
- [uv](https://docs.astral.sh/uv/) 包管理器
- NVIDIA GPU + CUDA 12.6 / 12.8 / 13.0 驱动栈之一用于完整训练；GPU 测试与 GPU benchmark 通过本地 CUDA CLI 复现

!!! important "平台支持"
    仓库只支持 Linux 运行时。Windows 与 WSL 不在支持范围内。

## 安装

```bash
# 克隆仓库并同步依赖
git clone https://github.com/Puiching-Memory/TAAC_2026.git
cd TAAC_2026
git lfs install
git lfs pull

# 安装并固定 Python 3.13
uv python install 3.13

# CPU-only profile：文档、unit、CPU benchmark
uv sync --locked --extra cpu

# GPU profile：训练，以及本地 integration / GPU 测试 / GPU benchmark
# 手动选择与你本机 CUDA 对应的 profile；如需切换可改成 cuda126 / cuda128 / cuda130
uv sync --locked --extra cuda128
```

文档站提交的 EDA 与技术时间线图表 JSON 由 Git LFS 管理；如果你看到 `docs/assets/figures/**/*.echarts.json` 只有 pointer 文本，先重新执行 `git lfs pull`。

!!! warning "关于 PyPI 镜像"
    仓库在 `pyproject.toml` 里固定了 canonical PyPI 作为默认索引，以保证 `uv.lock` 在本机和 CI 间一致。不要额外传 `--default-index` 或 `--index-url` 指向国内镜像，否则 `uv` 会判定 `uv.lock` 需要更新。  
    如果只是想加速下载，优先使用系统代理或透明代理。

## 训练第一个模型

```bash
uv run taac-train --experiment config/baseline
```

训练完成后，产物会写入 `outputs/config/baseline/`：

```
outputs/config/baseline/
├── best.pt                 # 最佳 checkpoint（按 val AUC 选择）
├── summary.json            # 训练摘要（指标、超参数、耗时）
├── validation_predictions.jsonl  # 最佳验证轮次的样本级预测
├── training_curves.json    # 逐 epoch 训练曲线
└── profiling/              # 模型参数量、FLOPs、推理延迟
```

`validation_predictions.jsonl` 每行包含：`sample_index`、`user_id`、`item_id`、`timestamp`、`raw_label`、`target`、`score`。

如果你希望提前打开编译与 AMP，可以直接使用同一套 CLI：

```bash
uv run taac-train --experiment config/baseline --compile --amp --amp-dtype bfloat16
```

如果你需要把单个实验包上传到线上训练平台，可以先生成单 zip：

```bash
uv run taac-package-train --experiment config/baseline
```

默认产物会写到 `outputs/training_bundles/baseline-train-bundle.zip`。

## 数据集选择

实验包默认使用 HuggingFace 上的样例数据集 [`TAAC2026/data_sample_1000`](https://huggingface.co/datasets/TAAC2026/data_sample_1000)。

默认行为如下：

- 不传 `--dataset-path`：直接按数据集名 `TAAC2026/data_sample_1000` 加载
- 传本地路径：支持 parquet 文件路径或包含 parquet 的目录路径
- 传自定义 Hub 名称：按你给的 `<owner>/<repo>` 直接加载

当目标 Hub 数据不在本地缓存时，`datasets` 会自动下载并写入缓存。

你也可以显式指定数据路径覆盖默认值：

```bash
# 本地 parquet
uv run taac-train --experiment config/baseline --dataset-path /path/to/train.parquet

# 本地目录（包含 parquet）
uv run taac-train --experiment config/baseline --dataset-path /path/to/dataset_dir

# 自定义 Hub 数据集名
uv run taac-train --experiment config/baseline --dataset-path some_owner/some_dataset
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

# 显式指定评估报告输出路径
uv run taac-evaluate single --experiment config/baseline --output-path outputs/evaluation.json
```

评估报告包含：AUC、PR-AUC、Brier Score、LogLoss、GAUC，以及推理延迟（ms/sample）。

评估还会额外导出样本级验证集预测：

- 不传 `--output-path`：写到运行目录下的 `validation_predictions.jsonl`
- 传 `--output-path outputs/evaluation.json`：写到同目录下的 `outputs/evaluation.validation_predictions.jsonl`

该 jsonl 与训练产物使用同一字段结构，适合做错误样本排查、切片分析和后续提交格式转换。

`--quantize int8` 当前会把推理切到 CPU，并通过 torchao 对 `nn.Linear` 应用动态 int8 量化；如果模型包含 TorchRec `EmbeddingBagCollection`，评估会直接拒绝这条 int8 路径并要求回退到 `--quantize none`。这条路径适合本地或 CI 的推理回归验证，不会修改训练 checkpoint 本身。

`--export-mode torch-export` 会基于当前评估 batch 生成 `.pt2` artifact，默认输出到评估输出目录，可用于后续 AOTI / runtime backend 集成验证。

## 超参数搜索

```bash
# 使用 Optuna 搜索 20 个 trial，自动按 GPU 显存并发调度
uv run taac-search --experiment config/baseline --trials 20
```

搜索默认约束：

- 模型参数量 ≤ 3 GiB
- 默认不启用模型单样本前向计算量上限

如需按模型单样本前向计算量过滤 trial，可追加 `--max-model-tflops-per-sample <value>`。

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

## 线上训练打包

`taac-package-train` 会把指定实验包裁剪成训练所需的最小运行时，并输出单个 zip。
zip 顶层包含：

- `run.sh`：自动解压 payload、按 `TAAC_CUDA_PROFILE` 选择 CUDA profile、调用训练 CLI
- `runtime_payload.tar.gz`：最小训练源码与目标实验包
- `bundle_manifest.json`：打包元数据
- `README.md`：压缩包内的使用说明

其中 `runtime_payload.tar.gz` 解压后包含 `pyproject.toml`、`uv.lock` 和最小训练源码。

```bash
# 生成默认命名的 zip
uv run taac-package-train --experiment config/baseline

# 自定义输出文件名
uv run taac-package-train --experiment config/interformer --output /tmp/interformer-online.zip --force
```

解压后，线上环境最少只需要：

```bash
export TAAC_DATASET_PATH=/path/to/train.parquet
bash run.sh --compile --amp --amp-dtype bfloat16
```

其中：

- `TAAC_DATASET_PATH` 必填，指向 parquet 文件或数据缓存目录
- `TAAC_OUTPUT_DIR` 可选，覆盖训练输出目录
- `TAAC_ENABLE_TE=1` 可选，为 bundle 安装 transformer-engine 额外依赖
- `TAAC_FORCE_EXTRACT=1` 可选，强制重新解压 payload

如果你不走 bundle，也可以直接对训练 CLI 传运行时数据路径：

```bash
uv run taac-train --experiment config/baseline --dataset-path /path/to/train.parquet
```

→ 完整参数和目录结构见 [线上训练打包](guide/online-training-bundle.md)

## 跑测试

```bash
# 完整回归
uv run pytest tests -q

# 快速单元测试
uv run pytest -m unit -q

# 本地 GPU 测试集合（CLI only）
uv run pytest tests/gpu/test_gpu_environment.py tests/gpu -q
```

→ 详细测试指南见 [测试](guide/testing.md)

## 本地文档站

文档构建前需要先刷新 EDA 与技术时间线图表，并把生成的 JSON 一起提交到仓库：

```bash
uv sync --locked --no-install-package torch --no-install-package torchrec --no-install-package fbgemm-gpu --no-install-package triton
uv run taac-dataset-eda
uv run taac-tech-timeline
uv run taac-bench-report
uv run --no-project --isolated --with zensical zensical build --clean
```

→ 详细步骤见 [本地生成站点](guide/local-site.md)

## CLI 命令速查

| 命令                          | 用途                                                          |
| ----------------------------- | ------------------------------------------------------------- |
| `taac-train`                  | 训练实验包                                                    |
| `taac-evaluate`               | 评估 checkpoint                                               |
| `taac-search`                 | Optuna 超参数搜索                                             |
| `taac-package-train`          | 打包线上训练 zip                                              |
| `taac-bench-report`           | 生成 benchmark 图表                                           |
| `taac-plot-model-performance` | 生成性能对比图                                                |
| `taac-clean-pycache`          | 清理 `__pycache__`，见 [仓库缓存清理](guide/cache-cleanup.md) |
