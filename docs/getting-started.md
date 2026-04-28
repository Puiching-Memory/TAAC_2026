---
icon: lucide/rocket
---

# 快速开始

## 前置要求

- Linux 运行时；Windows 与 WSL 不在支持范围内。
- Python `>=3.10,<3.14`，本地推荐使用仓库固定的 Python 3.10.20。
- 本地开发使用 [uv](https://docs.astral.sh/uv/)。
- pytest、hypothesis 和 benchmark 工具已经并入默认依赖；CUDA 运行时仍通过 `cuda126` extra 启用。
- 线上 bundle 使用平台已激活的 Python/Conda 环境，不要求线上安装 `uv`。

## 安装

```bash
git clone https://github.com/Puiching-Memory/TAAC_2026.git
cd TAAC_2026

git lfs install
git lfs pull

uv python install 3.10.20
uv sync --locked --extra cuda126
```

!!! warning "不要随意替换 uv 索引"
    仓库的 `pyproject.toml` 与 `uv.lock` 是依赖事实源。额外传 `--default-index` 或 `--index-url` 可能让 `uv` 判定 lockfile 需要更新。

## 准备数据

训练与评估命令都需要官方 parquet 数据。`--dataset-path` 可以指向 parquet 文件，也可以指向包含 parquet 的目录。`--schema-path` 指向官方 `schema.json`；如果 schema 与 parquet 在同一目录，可以省略。

```bash
DATASET_PATH=data/sample_1000_raw/demo_1000.parquet
SCHEMA_PATH=data/sample_1000_raw/schema.json
```

## 训练第一个模型

先用很小的 epoch 和 batch 做通路检查：

```bash
bash run.sh train --experiment config/baseline \
    --dataset-path "$DATASET_PATH" \
    --schema-path "$SCHEMA_PATH" \
    --num_epochs 1 \
    --batch_size 8 \
    --device cpu
```

本地真正训练时通常切到 CUDA：

```bash
uv sync --locked --extra cuda126

bash run.sh train --experiment config/baseline \
    --dataset-path "$DATASET_PATH" \
    --schema-path "$SCHEMA_PATH" \
    --device cuda
```

默认训练产物会写入 `outputs/config/baseline/`，常见文件包括：

```text
outputs/config/baseline/
├── best.pt
├── summary.json
├── validation_predictions.jsonl
├── training_curves.json
├── ns_groups.json
├── logs/
└── tensorboard/
```

`summary.json` 记录训练摘要和指标；`best.pt` 是最佳验证 checkpoint；`validation_predictions.jsonl` 是样本级验证预测。

## 评估

`run.sh val` 会调用评估 CLI 的 `single` 模式，并默认读取运行目录中的 `best.pt`。评估同样需要数据路径：

```bash
bash run.sh val --experiment config/baseline \
    --dataset-path "$DATASET_PATH" \
    --schema-path "$SCHEMA_PATH" \
    --run-dir outputs/config/baseline \
    --device cpu
```

显式指定输出文件：

```bash
bash run.sh val --experiment config/baseline \
    --dataset-path "$DATASET_PATH" \
    --schema-path "$SCHEMA_PATH" \
    --run-dir outputs/config/baseline \
    --output outputs/evaluation.json \
    --device cpu
```

当前 PCVR 评估报告会在 `auc` 旁边自动输出 `auc_ci`、`logloss`、`brier`、`score_diagnostics` 和 `sample_count`，并在顶层写出 `data_diagnostics` 说明 Row Group 切分是否适合 L1 比较。评估同时写出 `evaluation.json` 与 `validation_predictions.jsonl`。运行参数以当前 `taac-evaluate single` parser 为准。

## 运行其他实验包

所有实验包都走同一条入口，只替换 `--experiment`：

```bash
bash run.sh train --experiment config/interformer \
    --dataset-path "$DATASET_PATH" \
    --schema-path "$SCHEMA_PATH" \
    --num_epochs 1 \
    --batch_size 8 \
    --device cpu

bash run.sh train --experiment config/onetrans \
    --dataset-path "$DATASET_PATH" \
    --schema-path "$SCHEMA_PATH" \
    --num_epochs 1 \
    --batch_size 8 \
    --device cpu
```

→ 完整列表见 [实验包总览](experiments/index.md)

## 线上训练打包

线上平台上传目录应只包含两个顶层文件：

```text
<training_bundle>/
├── run.sh
└── code_package.zip
```

生成 baseline bundle：

```bash
uv run taac-package-train --experiment config/baseline --force
```

自定义输出目录：

```bash
uv run taac-package-train --experiment config/interformer \
    --output-dir outputs/training_bundles/interformer_training_bundle \
    --force
```

本地仓库模式默认 runner 是 `uv`；检测到同级 `code_package.zip` 的 bundle 模式默认 runner 是 `python`。线上平台通常没有 `uv`，因此上传包会复用平台已激活的 Python/Conda 环境。

线上运行示例：

```bash
export TAAC_DATASET_PATH=/path/to/train.parquet_or_dataset_dir
export TAAC_SCHEMA_PATH=/path/to/schema.json
export TAAC_OUTPUT_DIR=/path/to/output
export TAAC_RUNNER=python

bash run.sh --device cuda
```

做线上 smoke 时可以临时缩小训练量：

```bash
bash run.sh --num_epochs 1 --batch_size 8 --device cpu
```

→ 完整格式和环境变量见 [线上训练打包](guide/online-training-bundle.md)

## 测试

当前可执行回归集中在 `tests/unit/`：

```bash
uv run pytest tests/unit -q
uv run pytest tests/unit/test_experiment_packages.py -q
uv run pytest tests/unit/test_package_training.py -q
```

→ 详细说明见 [测试](guide/testing.md)

## 本地文档站

```bash
uv run --no-project --isolated --with zensical zensical build --clean
```

如果要刷新 EDA 或技术时间线图表，再按 [本地生成站点](guide/local-site.md) 执行对应报告命令。当前 benchmark 报告入口只是占位 JSON 输出。

## 统一入口速查

| 命令                        | 用途                                        |
| --------------------------- | ------------------------------------------- |
| `bash run.sh train`         | 训练实验包                                  |
| `bash run.sh val`           | 评估单个实验/单个 checkpoint                |
| `bash run.sh infer`         | 生成推理结果                                |
| `uv run pytest`            | 运行 pytest                                 |
| `uv run taac-search`        | 记录搜索请求 JSON                           |
| `uv run taac-package-train` | 生成 `run.sh` + `code_package.zip` 上传目录 |
| `uv run taac-package-infer` | 生成 `infer.py` + `code_package.zip` 上传目录 |