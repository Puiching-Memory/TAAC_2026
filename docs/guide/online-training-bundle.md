---
icon: lucide/package
---

# 线上训练打包

当线上平台只识别顶层 `run.sh` 时，使用 `bash run.sh package` 生成一个只包含运行脚本和代码包的上传目录。当前平台契约是双文件，不是单个 zip。

```text
<training_bundle>/
├── run.sh
└── code_package.zip
```

## 生成上传目录

```bash
bash run.sh package --experiment config/baseline --force

bash run.sh package --experiment config/interformer \
    --output-dir outputs/training_bundles/interformer_training_bundle \
    --force
```

默认输出目录为：

```text
outputs/training_bundles/<experiment>_training_bundle/
```

上传时保持 `run.sh` 和 `code_package.zip` 位于同一目录。平台执行 `run.sh` 后，脚本会解压旁边的 `code_package.zip`，设置 `PYTHONPATH`，再启动训练 CLI。

## code_package.zip 内容

代码包解压后是一个 `project/` 目录，通常包含：

```text
project/
├── .taac_training_manifest.json
├── pyproject.toml
├── uv.lock
├── README.md
├── tools/
│   └── log_host_device_info.sh
├── src/
│   └── taac2026/
└── config/
    ├── __init__.py
    └── <selected_experiment>/
        ├── __init__.py
        ├── model.py
        └── ns_groups.json
```

代码包不包含 `docs/`、`site/`、`tests/` 或其他未选择的实验包。`uv.lock` 会随包保存用于追溯和本地复现，但线上 bundle 默认不会执行 `uv sync`。

## 本地模式与 Bundle 模式

同一个 `run.sh` 支持两种模式：

| 模式         | 触发条件                             | 默认 runner | 说明                                      |
| ------------ | ------------------------------------ | ----------- | ----------------------------------------- |
| 本地仓库模式 | `run.sh` 同级没有 `code_package.zip` | `uv`        | 自动按命令同步 `cpu` 或 `cuda126` profile |
| Bundle 模式  | `run.sh` 同级存在 `code_package.zip` | `python`    | 解压代码包并复用平台 Python/Conda 环境    |

可用 `TAAC_RUNNER=python|uv` 显式覆盖 runner。线上平台通常没有 `uv`，因此推荐保持 `TAAC_RUNNER=python`。

## 线上运行

至少提供训练数据路径：

```bash
export TAAC_DATASET_PATH=/path/to/train.parquet_or_dataset_dir
export TAAC_SCHEMA_PATH=/path/to/schema.json
export TAAC_OUTPUT_DIR=/path/to/output
export TAAC_RUNNER=python

bash run.sh
```

如果要做一次短 smoke：

```bash
bash run.sh --num_epochs 1 --batch_size 8 --device cpu
```

GPU 训练通常只需要切换 device：

```bash
bash run.sh --device cuda
```

`run.sh` 会把环境变量映射为训练 CLI 参数：

```bash
python -m taac2026.application.training.cli \
    --experiment <manifest experiment> \
    --dataset-path "$TAAC_DATASET_PATH" \
    --schema-path "$TAAC_SCHEMA_PATH" \
    --run-dir "$TAAC_OUTPUT_DIR" \
    "$@"
```

不要把历史训练栈里的 runtime optimization 参数复制到当前 PCVR 训练命令中；当前共享 PCVR parser 不支持这些参数。

## 环境变量

| 变量                                     | 是否常用 | 作用                                                 |
| ---------------------------------------- | -------- | ---------------------------------------------------- |
| `TAAC_DATASET_PATH` / `TRAIN_DATA_PATH`  | 是       | parquet 文件或包含 parquet 的目录                    |
| `TAAC_SCHEMA_PATH` / `TRAIN_SCHEMA_PATH` | 常用     | schema 不在数据同目录时显式指定                      |
| `TAAC_OUTPUT_DIR` / `TRAIN_CKPT_PATH`    | 常用     | 训练输出目录                                         |
| `TAAC_EXPERIMENT`                        | 偶尔     | 覆盖 manifest 中的实验包路径                         |
| `TAAC_BUNDLE_WORKDIR`                    | 偶尔     | 控制 `code_package.zip` 解压目录                     |
| `TAAC_CODE_PACKAGE`                      | 偶尔     | 指向非默认位置的 `code_package.zip`                  |
| `TAAC_FORCE_EXTRACT`                     | 偶尔     | 设为 `1` 时强制重新解压                              |
| `TAAC_RUNNER`                            | 常用     | `python` 或 `uv`；bundle 默认 `python`               |
| `TAAC_PYTHON`                            | 偶尔     | 指定 Python 解释器，例如平台 Conda 环境中的 `python` |
| `TAAC_CUDA_PROFILE`                      | 本地     | 本地 uv 模式固定使用 `cuda126`；传其他值会直接失败   |
| `TAAC_SKIP_UV_SYNC`                      | 本地     | 本地 uv 模式跳过依赖同步                             |
| `TAAC_INSTALL_UV`                        | 本地     | 本地 uv 不存在时是否尝试安装                         |

## 检查 Bundle

```bash
python -m zipfile -l outputs/training_bundles/baseline_training_bundle/code_package.zip | head -80
```

重点确认：

- 有 `project/.taac_training_manifest.json`。
- 有 `project/src/taac2026/...`。
- 有目标实验包的 `model.py` 和 `ns_groups.json`。
- 没有 `project/tests/`、`project/docs/` 和其他实验包。

## 常见问题

### 线上提示找不到模块

确认 `run.sh` 与 `code_package.zip` 在同一目录，并查看日志中是否完成了解压。bundle 模式会设置 `PYTHONPATH=<workdir>/project/src:<workdir>/project`。

### 线上缺少 Python 包

优先使用平台或自定义镜像预装 CUDA、PyTorch、FBGEMM、TorchRec 等核心栈。若平台允许预运行依赖步骤，只在当前 Conda Python 内补装缺失的纯 Python 包：

```bash
python -m pip install numpy pyarrow scikit-learn rich tensorboard tqdm optuna tomli
```

不要在任务启动时依赖公网下载核心 GPU 栈，也不要把 bundle runner 切回 `uv`。

### 想重新解压代码包

```bash
export TAAC_FORCE_EXTRACT=1
bash run.sh --num_epochs 1 --batch_size 8 --device cpu
```

### 线上跑错实验包

检查 `code_package.zip` 内的 `project/.taac_training_manifest.json`，并确认没有设置 `TAAC_EXPERIMENT` 覆盖 manifest。

## 相关命令

```bash
bash run.sh train --experiment config/baseline \
    --dataset-path /path/to/train.parquet \
    --schema-path /path/to/schema.json

bash run.sh package --experiment config/baseline --force

TAAC_RUNNER=python TAAC_DATASET_PATH=/path/to/train.parquet \
    bash outputs/training_bundles/baseline_training_bundle/run.sh --num_epochs 1 --batch_size 8 --device cpu
```