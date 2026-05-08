---
icon: lucide/package
---

# 线上 Bundle 上传

本页记录线上 bundle 的打包格式和运行时契约。它面向要上传、调试或修改 bundle 逻辑的人，不只是列两条打包命令。

当前有两类上传物：

- 训练 bundle：`run.sh` + `code_package.zip`
- 推理 bundle：`infer.py` + `code_package.zip`

它们都由专门 CLI 生成，不走 `bash run.sh package`。

## 格式版本

当前 manifest 版本在 `src/taac2026/infrastructure/bundles/manifest_store.py` 中定义：

| bundle    | manifest                                | format                  | version | entrypoint |
| --------- | --------------------------------------- | ----------------------- | ------- | ---------- |
| training  | `project/.taac_training_manifest.json`  | `taac2026-training-v2`  | 2       | `run.sh`   |
| inference | `project/.taac_inference_manifest.json` | `taac2026-inference-v1` | 1       | `infer.py` |

manifest 会记录：

- `manifest_version`
- `bundle_kind`
- `bundle_format`
- `bundle_format_version`
- `framework.name` / `framework.version`
- `bundled_experiment_path`
- `entrypoint`
- `code_package`
- `runtime_env`

线上默认 runner 是 Python，而不是 `uv`。

## 训练 Bundle

生成：

```bash
uv run taac-package-train \
  --experiment experiments/baseline \
  --output-dir outputs/bundles/baseline_training
```

输出：

```text
outputs/bundles/baseline_training/
├── run.sh
└── code_package.zip
```

上传到平台时，把这两个文件放在同一个上传目录里。平台执行顶层 `run.sh`；脚本会解压 `code_package.zip`，安装项目，并调用训练 CLI。

训练 `run.sh` 的实际分发逻辑在 `src/taac2026/application/bootstrap/run_sh.py`。bundle 模式下的顺序是：

1. 找到同目录的 `code_package.zip`。
2. 解压到 `TAAC_BUNDLE_WORKDIR`，默认是入口旁边的 `.taac_bundle/project`。
3. 设置 `TAAC_BUNDLE_MODE=1` 和 `TAAC_PROJECT_DIR=<解压后的 project>`。
4. 读取 `project/.taac_training_manifest.json`。
5. 若未设置 `TAAC_EXPERIMENT`，使用 manifest 里的 `bundled_experiment_path`。
6. 调用当前 Python 的 pip 安装 `project/pyproject.toml` 对应项目。
7. 分发到 `taac2026.application.training.cli`。

训练时最常用的环境变量：

| 变量                   | 作用                              |
| ---------------------- | --------------------------------- |
| `TRAIN_DATA_PATH`      | 平台提供的训练 parquet 文件或目录 |
| `TRAIN_CKPT_PATH`      | checkpoint / 训练输出目录         |
| `TAAC_SCHEMA_PATH`     | `schema.json` 路径                |
| `TAAC_RUNNER=python`   | 线上使用平台 Python，不依赖 `uv`  |
| `TAAC_BUNDLE_WORKDIR`  | 自定义 bundle 解压目录            |

训练 bundle 还支持这些调试开关：

| 变量                          | 作用                                                |
| ----------------------------- | --------------------------------------------------- |
| `TAAC_EXPERIMENT`             | 覆盖 manifest 中的实验包路径                        |
| `TAAC_PYTHON`                 | 指定 Python 可执行文件，默认是当前 `sys.executable` |
| `TAAC_SKIP_PIP_INSTALL=1`     | 跳过 `pip install .`                                |
| `TAAC_INSTALL_PROJECT_DEPS=0` | 关闭默认项目安装                                    |
| `TAAC_BUNDLE_PIP_EXTRAS`      | 给 `pip install .[...]` 添加 extras                 |
| `TAAC_PIP_INDEX_URL`          | 覆盖默认 pip index，默认 Tencent PyPI               |
| `TAAC_PIP_EXTRA_ARGS`         | 追加 pip 参数，按 shell words 解析                  |

本地模拟训练 bundle 前，先按 [快速开始](../getting-started.md) 下载 `demo_1000.parquet`。训练 bundle 模拟使用本地 parquet 文件和归档 schema：

```bash
export TAAC_RUNNER=python
export TRAIN_DATA_PATH=data/sample_1000_raw/demo_1000.parquet
export TAAC_SCHEMA_PATH=docs/archive/files/schema/sample_1000_raw.schema.json
export TRAIN_CKPT_PATH=/tmp/taac-training-output

bash outputs/bundles/baseline_training/run.sh --device cpu --num_workers 0
```

## 推理 Bundle

生成：

```bash
uv run taac-package-infer \
  --experiment experiments/baseline \
  --output-dir outputs/bundles/baseline_inference
```

输出：

```text
outputs/bundles/baseline_inference/
├── infer.py
└── code_package.zip
```

上传评测时，平台要求主入口叫 `infer.py`。仓库生成的 `infer.py` 会解压同目录下的 `code_package.zip`，再调用共享推理入口。

推理 bundle 的执行顺序是：

1. `infer.py` 解压 `code_package.zip` 到 bundle workdir。
2. `inference_bundle.run_inference_bundle()` 读取 `project/.taac_inference_manifest.json`。
3. 安装项目依赖。
4. 若未设置 `TAAC_EXPERIMENT`，使用 manifest 中的实验包。
5. 设置 `project/src` 和 `project` 到 import path。
6. 调用 `taac2026.application.evaluation.infer.main()`。
7. `evaluation.infer` 从环境变量构造 `taac-evaluate infer` 参数。

推理时最常用的环境变量：

| 变量                   | 作用                              |
| ---------------------- | --------------------------------- |
| `EVAL_DATA_PATH`       | 平台提供的测试 parquet 文件或目录 |
| `EVAL_RESULT_PATH`     | `predictions.json` 输出目录       |
| `MODEL_OUTPUT_PATH`    | 已发布模型或 checkpoint 路径      |
| `TAAC_SCHEMA_PATH`     | `schema.json` 路径                |
| `TAAC_BUNDLE_WORKDIR`  | 自定义 bundle 解压目录            |

推理额外支持：

| 变量                     | 作用                                                  |
| ------------------------ | ----------------------------------------------------- |
| `TAAC_INFER_BATCH_SIZE`  | 映射为 `--batch-size`                                 |
| `TAAC_INFER_NUM_WORKERS` | 映射为 `--num-workers`                                |
| `TAAC_INFER_AMP`         | `1/0/true/false`，映射为 `--amp` / `--no-amp`         |
| `TAAC_INFER_AMP_DTYPE`   | 映射为 `--amp-dtype`                                  |
| `TAAC_INFER_COMPILE`     | `1/0/true/false`，映射为 `--compile` / `--no-compile` |

本地模拟推理 bundle 同样使用本地下载的 parquet 文件和归档 schema：

```bash
export TAAC_BUNDLE_WORKDIR=/tmp/taac-infer-bundle
export EVAL_DATA_PATH=data/sample_1000_raw/demo_1000.parquet
export EVAL_RESULT_PATH=/tmp/taac-infer-results
export MODEL_OUTPUT_PATH=outputs/quickstart_baseline
export TAAC_SCHEMA_PATH=docs/archive/files/schema/sample_1000_raw.schema.json

python outputs/bundles/baseline_inference/infer.py
```

## 代码包里有什么

`code_package.zip` 内部不是把整个仓库原样塞进去，而是一个精简的 `project/` 目录：

```text
code_package.zip
└── project/
    ├── .taac_training_manifest.json      # 或 .taac_inference_manifest.json
    ├── pyproject.toml
    ├── src/taac2026/
    └── experiments/<selected_experiment>/
```

它不会包含 `uv.lock`、`README.md`、`tests/`、无关实验包、顶层 `run.sh` 或顶层 `infer.py`。所以验证 bundle 时，一定要看 zip 内容，而不是只看本地仓库能不能 import。

zip 写入逻辑在 `src/taac2026/infrastructure/bundles/zip_writer.py`。它会包含：

- `project/<manifest>`
- `project/pyproject.toml`
- `project/src/taac2026/**`
- 当前实验包目录 `project/experiments/<name>/**`
- 当前实验包父级中存在的 `__init__.py`

它会跳过 `__pycache__` 和 `.pyc`。

检查命令：

```bash
python -m zipfile -l outputs/bundles/baseline_training/code_package.zip | sed -n '1,80p'
unzip -p outputs/bundles/baseline_training/code_package.zip project/.taac_training_manifest.json | python -m json.tool
```

重点看：

- `bundled_experiment_path` 是否是你要上传的实验包。
- `bundle_kind` 是否是 `training` 或 `inference`。
- `entrypoint` 是否分别是 `run.sh` 或 `infer.py`。
- `runtime_env` 是否列出了平台需要提供的环境变量。

如果你改了 bundle 生成逻辑，至少构建一个训练 bundle 和一个推理 bundle，并检查 zip：

```bash
uv run taac-package-train \
  --experiment experiments/baseline \
  --output-dir /tmp/taac-training-bundle \
  --json

uv run taac-package-infer \
  --experiment experiments/baseline \
  --output-dir /tmp/taac-inference-bundle \
  --json

python -m zipfile -l /tmp/taac-training-bundle/code_package.zip | sed -n '1,80p'
```

## 维护类实验

维护类实验只支持训练 bundle，不支持推理 bundle。

```bash
uv run taac-package-train \
  --experiment experiments/host_device_info \
  --output-dir outputs/bundles/host_device_info
```

`host_device_info` 不需要数据集。`online_dataset_eda` 需要平台提供 `TRAIN_DATA_PATH`，通常也需要 `TAAC_SCHEMA_PATH`。

## 常见问题

**线上提示 `uv: command not found`**

线上 bundle 应使用 `TAAC_RUNNER=python`，复用平台 Conda Python。

**跑错实验包**

先看 manifest 里的 `bundled_experiment_path`，再检查是否设置了 `TAAC_EXPERIMENT` 覆盖它。

**依赖安装失败**

不要在线重装 CUDA / PyTorch / TorchRec / FBGEMM。缺少纯 Python 依赖时，优先确认平台镜像是否可用，再调整 `pyproject.toml` 或 `TAAC_BUNDLE_PIP_EXTRAS`。

**manifest 路径找不到**

先确认 zip 内是否存在 `project/.taac_training_manifest.json` 或 `project/.taac_inference_manifest.json`。manifest 不在 zip 根目录。

**推理没有写出 `predictions.json`**

检查 `EVAL_RESULT_PATH` 是否设置、目录是否可写，以及 `MODEL_OUTPUT_PATH` 是否能解析到 `model.safetensors`。`MODEL_OUTPUT_PATH` 可以是 checkpoint 文件，也可以是包含 `global_step*.best_model/` 的根目录。

## 源码入口

- 打包 CLI：`src/taac2026/application/packaging/`
- zip 生成：`src/taac2026/infrastructure/bundles/`
- 训练入口：`src/taac2026/application/bootstrap/run_sh.py`
- 推理入口：`src/taac2026/application/bootstrap/inference_bundle.py`
- 推理环境变量桥接：`src/taac2026/application/evaluation/infer.py`
