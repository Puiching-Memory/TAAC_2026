---
icon: lucide/package
---

# 训练与推理 Bundle

Bundle 是平台上传用的代码包。训练入口是 `run.sh`，推理入口是 `infer.py`，两者均与同目录的 `code_package.zip` 配合。模型权重由训练产物提供，不封装在推理代码包中。

## 先核对运行环境

当前 `pyproject.toml` 要求 Python `>=3.12,<3.15`，依赖采用当前项目版本；打包器会把这份文件原样放入 zip。归档平台日志记录的是 Python 3.10.20、Torch 2.7.1+cu126，不能据此证明当前包可直接在旧平台运行。

平台需提供符合要求的解释器、Torch/CUDA 和依赖安装条件。默认 pip 安装会校验 Python 下限；`TAAC_SKIP_PIP_INSTALL=1` 只跳过安装，不会修复语法、ABI 或包版本不兼容。生成 zip 成功也不等于上传执行成功。

## 生成上传物

在已同步 CUDA 环境中运行下面命令。仅有 CPU/dev 环境时，在 `uv run` 后加 `--with torch==2.13.0`，以满足实验导入。

```bash
uv run taac-package-train \
  --experiment experiments/baseline \
  --output-dir outputs/bundles/baseline_training
uv run taac-package-infer \
  --experiment experiments/baseline \
  --output-dir outputs/bundles/baseline_inference
```

```text
baseline_training/             baseline_inference/
├── run.sh                     ├── infer.py
└── code_package.zip           └── code_package.zip
```

输出目录已有文件时按 CLI 提示处理；仅在确实要覆盖该目标时使用 `--force`。不要用 `bash run.sh package` 或 `package-infer`，shell 不支持这些子命令。

## 检查 zip 和 manifest

```bash
uv run python -m zipfile -l outputs/bundles/baseline_training/code_package.zip
unzip -p outputs/bundles/baseline_training/code_package.zip \
  project/.taac_training_manifest.json
```

内部结构为：

```text
project/
├── .taac_training_manifest.json   # 推理包为 .taac_inference_manifest.json
├── pyproject.toml
├── src/taac2026/
└── experiments/<selected>/
```

manifest 的格式名分别为 `taac2026-training`、`taac2026-inference`，记录框架版本、实验路径、入口、代码包名与运行环境字段。确认选中的实验与上传入口对应。

包不包含 `uv.lock`、README、tests、docs、无关实验、数据或权重。实验如果跨目录 import 其他模型，在仓库里正常也可能在包中失败。zip 写入还会排除 `__pycache__` 与 `.pyc`。

## 执行过程

训练入口解压为 `<workdir>/project`，设置项目 import 路径与 bundle mode，读取 manifest，使用平台 Python 安装项目，再调用训练 CLI。推理入口执行相同的项目准备后，经 `evaluation/infer.py` 将环境变量映射到 `taac-evaluate infer`。

默认 workdir 是入口旁边的 `.taac_bundle`。训练 shell 每次会重建其中的 `project` 子目录，不应在该解压目录保存训练产物。路径变量使用绝对路径，避免 bootstrap 切换工作目录后改变含义。

## 平台变量

| 变量 | 训练 | 推理 |
| --- | --- | --- |
| `TRAIN_DATA_PATH` | parquet 文件或目录 | — |
| `TRAIN_CKPT_PATH` | 输出 run 目录 | — |
| `EVAL_DATA_PATH` | — | parquet 文件或目录 |
| `EVAL_RESULT_PATH` | — | 输出目录 |
| `MODEL_OUTPUT_PATH` | — | 模型路径，建议具体 checkpoint 目录或权重 |
| `TAAC_SCHEMA_PATH` | 显式 schema | 覆盖 checkpoint schema，通常不必设置 |
| `TAAC_BUNDLE_WORKDIR` | 解压根目录 | 解压根目录 |
| `TAAC_EXPERIMENT` | 覆盖 manifest 实验 | 覆盖 manifest 实验 |
| `TAAC_PYTHON` | 指定解释器 | 指定解释器 |

显式 checkpoint 目录应直接包含 `model.safetensors`、`schema.json` 与 `train_config.json`。当前 bootstrap 会将 `MODEL_OUTPUT_PATH` 转为显式 `--checkpoint`，因此平台变量也优先使用具体 checkpoint，不能假设任何 run 根目录都会递归解析。

推理还支持 `TAAC_INFER_BATCH_SIZE`、`TAAC_INFER_NUM_WORKERS`、`TAAC_INFER_AMP`、`TAAC_INFER_AMP_DTYPE`、`TAAC_INFER_COMPILE`。布尔环境值由桥接层解析；查看 `application/evaluation/infer.py` 确认接受形式。

## 本地模拟

先完成[快速开始](../getting-started.md)，并在具备所有项目依赖的环境里执行。下面的 `CHECKPOINT` 使用具体目录，替换为自己已有的训练产物。

```bash
export TRAIN_DATA_PATH="$(pwd)/data/sample_1000_raw/demo_1000.parquet"
export TAAC_SCHEMA_PATH="$(pwd)/data/sample_1000_raw/schema.json"
export TRAIN_CKPT_PATH="$(pwd)/outputs/bundle_smoke"
export TAAC_PYTHON="$(pwd)/.venv/bin/python"
export TAAC_SKIP_PIP_INSTALL=1
bash outputs/bundles/baseline_training/run.sh \
  --optimizer.device cpu --data.num_workers 0 \
  --data.batch_size 8 --optimizer.max_steps 1 \
  --runtime.no_amp --runtime.no_compile
```

本例跳过安装以检查解压与执行路径，未验证默认在线 pip 安装。若要验收安装，去掉 skip 开关，在干净的目标兼容环境复核。

```bash
export EVAL_DATA_PATH="$(pwd)/data/sample_1000_raw/demo_1000.parquet"
export EVAL_RESULT_PATH="$(pwd)/outputs/bundle_infer"
# 替换为训练实际生成的 global_step... 目录
export MODEL_OUTPUT_PATH="$(pwd)/outputs/bundle_smoke/<global_step目录>"
export TAAC_INFER_NUM_WORKERS=0
"$TAAC_PYTHON" outputs/bundles/baseline_inference/infer.py
```

输出应有 `predictions.json`。CPU 单独模拟的设备选择需与目标环境一致；不要把在 GPU 机器上未指定设备的命令称作 CPU 验证。

## 安装控制与失败定位

| 控制项 | 行为 |
| --- | --- |
| `TAAC_RUNNER=python` | shell 使用平台 Python；bundle 默认如此 |
| `TAAC_SKIP_PIP_INSTALL=1` | 跳过项目安装 |
| `TAAC_INSTALL_PROJECT_DEPS=0` | 关闭默认安装 |
| `TAAC_BUNDLE_PIP_EXTRAS` | 为 pip 安装追加项目 extras |
| `TAAC_PIP_INDEX_URL` | 替换默认 Tencent 镜像 |
| `TAAC_PIP_EXTRA_ARGS` | 追加 pip 参数，按 shell words 解析 |

缺依赖时先核对解释器与镜像，不要将不匹配的 CUDA extra 直接装入平台已有 GPU 栈。manifest 不存在时检查它是否位于 zip 的 `project/` 内；模型重建失败时检查三件套与代码版本；没有预测文件时先看入口是否为 EDA 维护实验。

Host Device Info 只可生成训练包；EDA 具有 infer 回调，可生成统计用途推理包。操作流程快照见[平台指南](official-competition-docs.md)。

源码：`application/packaging/`、`application/bootstrap/`、`infrastructure/bundles/`、`infrastructure/platform/deps.py`（均相对 `src/taac2026/`）。
