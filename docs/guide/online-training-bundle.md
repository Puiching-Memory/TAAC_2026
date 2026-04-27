---
icon: lucide/package
---

# 线上 Bundle 上传指南

本页只说明线上训练和评测推理需要上传什么、怎么生成、运行时要配置哪些环境变量。打包在本地完成；线上运行时默认复用平台已经激活的 Python/Conda 环境，不在线执行 `uv sync`。

仓库提供两类双文件 bundle：

| 场景 | 本地生成命令                | 上传入口   | 同目录代码包       |
| ---- | --------------------------- | ---------- | ------------------ |
| 训练 | `uv run taac-package-train` | `run.sh`   | `code_package.zip` |
| 推理 | `uv run taac-package-infer` | `infer.py` | `code_package.zip` |

上传时保持入口脚本和 `code_package.zip` 位于同一目录。不要把整个输出目录再次压成单个 zip。

## 训练 Bundle

训练上传目录的顶层结构固定为：

```text
<training_bundle>/
├── run.sh
└── code_package.zip
```

生成 baseline bundle：

```bash
uv run taac-package-train --experiment config/baseline --force
```

指定输出目录：

```bash
uv run taac-package-train --experiment config/interformer \
    --output-dir outputs/training_bundles/interformer_training_bundle \
    --force
```

默认输出目录为：

```text
outputs/training_bundles/<experiment>_training_bundle/
```

`run.sh` 只作为运行入口使用。`bash run.sh package` 和 `bash run.sh package-infer` 不是有效命令；打包请使用上面的维护 CLI。

## 训练运行

线上平台进入上传目录后执行 `run.sh`。运行前至少提供训练数据路径；schema 和输出目录建议显式指定。

```bash
export TAAC_DATASET_PATH=/path/to/train.parquet_or_dataset_dir
export TAAC_SCHEMA_PATH=/path/to/schema.json
export TAAC_OUTPUT_DIR=/path/to/output
export TAAC_RUNNER=python

bash run.sh --device cuda
```

短 smoke 可以使用 CPU 和更小 batch：

```bash
bash run.sh --num_epochs 1 --batch_size 8 --device cpu
```

Bundle 模式由同目录的 `code_package.zip` 自动触发。脚本会解压代码包，设置 `PYTHONPATH=<workdir>/project/src:<workdir>/project`，读取 `project/.taac_training_manifest.json` 中的实验包路径，然后调用训练 CLI。

常用训练环境变量：

| 变量                                     | 作用                                                 |
| ---------------------------------------- | ---------------------------------------------------- |
| `TAAC_DATASET_PATH` / `TRAIN_DATA_PATH`  | parquet 文件或包含 parquet 的目录                    |
| `TAAC_SCHEMA_PATH` / `TRAIN_SCHEMA_PATH` | `schema.json` 路径；不在数据同目录时需要设置         |
| `TAAC_OUTPUT_DIR` / `TRAIN_CKPT_PATH`    | checkpoint、日志和训练输出目录                       |
| `TAAC_RUNNER`                            | 线上保持 `python`；本地仓库模式默认是 `uv`           |
| `TAAC_PYTHON`                            | 指定 Python 解释器，例如平台 Conda 环境中的 `python` |
| `TAAC_EXPERIMENT`                        | 覆盖 bundle manifest 中的实验包；通常不需要设置      |
| `TAAC_BUNDLE_WORKDIR`                    | 控制代码包解压目录                                   |
| `TAAC_CODE_PACKAGE`                      | 指向非默认位置的 `code_package.zip`                  |
| `TAAC_FORCE_EXTRACT`                     | 设为 `1` 时强制重新解压代码包                        |

## 推理 Bundle

评测阶段入口脚本必须命名为 `infer.py`。推理上传目录的顶层结构固定为：

```text
<inference_bundle>/
├── infer.py
└── code_package.zip
```

生成 baseline 推理 bundle：

```bash
uv run taac-package-infer --experiment config/baseline --force
```

指定输出目录：

```bash
uv run taac-package-infer --experiment config/baseline \
    --output-dir outputs/inference_bundles/baseline_inference_bundle \
    --force
```

默认输出目录为：

```text
outputs/inference_bundles/<experiment>_inference_bundle/
```

生成后的 `infer.py` 会读取同目录的 `code_package.zip`，默认把代码解压到 `USER_CACHE_PATH` 下的稳定缓存子目录。随后它会读取 `project/.taac_inference_manifest.json`，设置默认 `TAAC_EXPERIMENT`，再转调仓库内的推理入口。

这样做是为了兼容线上评测目录只读的情况，并且只依赖官方文档明确给出的可写目录。如果运行环境没有提供 `USER_CACHE_PATH`，请显式设置 `TAAC_BUNDLE_WORKDIR`，不要依赖系统临时目录。

常用推理环境变量：

| 变量                  | 作用                                      |
| --------------------- | ----------------------------------------- |
| `MODEL_OUTPUT_PATH`   | 已发布 checkpoint 目录                    |
| `EVAL_DATA_PATH`      | 评测测试集目录                            |
| `EVAL_RESULT_PATH`    | 输出 `predictions.json` 的目录            |
| `TAAC_SCHEMA_PATH`    | 可选，显式覆盖 schema 路径                |
| `TAAC_EXPERIMENT`     | 可选，覆盖 bundle manifest 中的实验包     |
| `TAAC_BUNDLE_WORKDIR` | 可选，控制代码包解压目录                  |
| `TAAC_CODE_PACKAGE`   | 可选，指向非默认位置的 `code_package.zip` |
| `TAAC_FORCE_EXTRACT`  | 设为 `1` 时强制重新解压代码包             |

## 代码包内容

`code_package.zip` 解压后是一个 `project/` 目录。训练 bundle 通常包含：

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

推理 bundle 使用同样的最小代码包结构，但 manifest 文件名是 `.taac_inference_manifest.json`，并且顶层入口是上传目录中的 `infer.py`。

代码包只带共享 runtime 和选中的实验包，不包含 `docs/`、`site/`、`tests/` 或其他实验包。`uv.lock` 会随包保存用于追溯和本地复现，线上默认不会用它安装依赖。

## 检查 Bundle

生成后可以先看顶层文件：

```bash
ls -l outputs/training_bundles/baseline_training_bundle
ls -l outputs/inference_bundles/baseline_inference_bundle
```

再检查代码包清单：

```bash
python -m zipfile -l outputs/training_bundles/baseline_training_bundle/code_package.zip | head -80
python -m zipfile -l outputs/inference_bundles/baseline_inference_bundle/code_package.zip | head -80
```

重点确认：

- 顶层只有入口脚本和 `code_package.zip`。
- 代码包里有 manifest、`project/src/taac2026/` 和目标实验包。
- 目标实验包包含 `model.py` 与 `ns_groups.json`。
- 代码包里没有 `project/tests/`、`project/docs/`、`project/site/` 或未选择的实验包。

## 依赖原则

线上 bundle 默认依赖平台预装 Python/Conda 环境。核心 CUDA、PyTorch、FBGEMM、TorchRec 等 GPU 栈优先由平台镜像提供，不建议在任务启动时重新下载或覆盖安装。

如果线上确实缺少纯 Python 包，可以在平台允许的预运行步骤中使用当前 Conda Python 补装：

```bash
python -m pip install numpy pyarrow scikit-learn rich tensorboard tqdm optuna tomli
```

不要把 bundle runner 切回 `uv`，除非平台明确提供 `uv` 且依赖源可用。

## 常见问题

### 找不到模块

确认入口脚本和 `code_package.zip` 在同一目录。训练 bundle 入口是 `run.sh`，推理 bundle 入口是 `infer.py`。

### 使用了旧代码

强制重新解压代码包：

```bash
export TAAC_FORCE_EXTRACT=1
bash run.sh --num_epochs 1 --batch_size 8 --device cpu
```

推理阶段同样可以设置 `TAAC_FORCE_EXTRACT=1` 后重新执行 `infer.py`。

### 跑错实验包

检查代码包中的 manifest：训练看 `project/.taac_training_manifest.json`，推理看 `project/.taac_inference_manifest.json`。同时确认环境中没有设置错误的 `TAAC_EXPERIMENT`。

### 线上缺少依赖

优先确认平台 Python 环境是否已经包含所需包。确实缺少时，只补装缺失的纯 Python 包；核心 GPU 依赖应通过平台镜像或基础环境解决。