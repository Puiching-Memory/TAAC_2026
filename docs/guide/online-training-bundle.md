---
icon: lucide/package
---

# 线上 Bundle 上传指南

当前仓库把线上上传物拆成两类：

- 训练 Bundle：`run.sh` + `code_package.zip`
- 推理 Bundle：`infer.py` + `code_package.zip`

两类 Bundle 都通过专门的 CLI 生成，不通过 `run.sh package` 生成。

## 训练 Bundle

### 生成命令

```bash
uv run taac-package-train \
  --experiment experiments/baseline \
  --output-dir outputs/bundles/baseline_training
```

### 输出结构

```text
outputs/bundles/baseline_training/
├── run.sh
└── code_package.zip
```

### code_package.zip 内部结构

当前训练 Bundle 会把 manifest 和源码都放在 `project/` 子目录下：

```text
code_package.zip
└── project/
    ├── .taac_training_manifest.json
    ├── pyproject.toml
    ├── src/taac2026/
    └── experiments/
      └── <experiment>/
```

代码包的几个重要特征：

- manifest 路径是 `project/.taac_training_manifest.json`，不是 zip 根目录。
- 只包含 `src/taac2026` 和当前选中的实验包源码。
- 不包含 `uv.lock`、`README.md`、`tests/`、顶层 `run.sh`，也不会把无关实验包一起打进去。
- 当前实验包契约以包内 `__init__.py` 配置为主；不要假设 bundle 中一定有独立的 `ns_groups.json` 文件。

manifest 是版本化对象。除了旧脚本仍会读取的 `bundle_format`、`bundled_experiment_path`、`entrypoint` 和 `code_package` 外，它还包含 `manifest_version`、`bundle_kind`、`bundle_format_version`、`framework.version`、`runtime_env` 和 `compatibility`。打包端使用 Pydantic 2 校验这个对象，但写入 bundle 的仍是普通 JSON。

### 运行时行为

上传后的平台入口仍是顶层 `run.sh`。Shell 层只负责定位和解压代码包，然后委托给 `taac2026.application.bootstrap.run_sh`。在 Bundle 模式下，它会：

1. 检测同目录下是否存在 `code_package.zip`
2. 解压到 `TAAC_BUNDLE_WORKDIR` 指定目录，默认是 `run.sh` 同目录下的 `.taac_bundle/project`
3. 设置 `PYTHONPATH=project/src:project`
4. 读取 `project/.taac_training_manifest.json`
5. 在当前 Python 环境中执行 `pip install .`
6. 调用 `taac2026.application.training.cli`

默认使用 manifest 里记录的实验包路径；如果显式设置了 `TAAC_EXPERIMENT`，它会覆盖 manifest。

### 训练时常用环境变量

| 变量                      | 用途                                      |
| ------------------------- | ----------------------------------------- |
| `TRAIN_DATA_PATH`         | 训练数据 parquet 文件或目录               |
| `TAAC_SCHEMA_PATH`        | `schema.json` 路径                        |
| `TRAIN_CKPT_PATH`         | checkpoint / 输出目录                     |
| `TAAC_RUNNER=python`      | 强制 Bundle 使用当前 Python，而不是 `uv`  |
| `TAAC_BUNDLE_PIP_EXTRAS`  | 需要时为 `pip install .[...]` 添加 extras |
| `TAAC_SKIP_PIP_INSTALL=1` | 跳过 bundle 里的 `pip install .`          |
| `TAAC_FORCE_EXTRACT=1`    | 强制重新解压 `code_package.zip`           |
| `TAAC_BUNDLE_WORKDIR`     | 指定 bundle 解压目录                      |
| `TAAC_CODE_PACKAGE`       | 自定义 `code_package.zip` 路径            |

### 本地模拟一次训练 Bundle

```bash
export TAAC_RUNNER=python
export TRAIN_DATA_PATH=data/sample_1000_raw/demo_1000.parquet
export TAAC_SCHEMA_PATH=data/sample_1000_raw/schema.json
export TRAIN_CKPT_PATH=/tmp/taac-training-output

bash outputs/bundles/baseline_training/run.sh --device cpu --num_workers 0
```

## 推理 Bundle

### 生成命令

```bash
uv run taac-package-infer \
  --experiment experiments/baseline \
  --output-dir outputs/bundles/baseline_inference
```

### 输出结构

```text
outputs/bundles/baseline_inference/
├── infer.py
└── code_package.zip
```

### code_package.zip 内部结构

推理 Bundle 的代码包也是同样的 `project/` 结构，只是 manifest 名称不同：

```text
code_package.zip
└── project/
    ├── .taac_inference_manifest.json
    ├── pyproject.toml
    ├── src/taac2026/
    └── experiments/
      └── <experiment>/
```

它同样不会包含 `uv.lock`、`README.md`、顶层 `infer.py`、顶层 `run.sh` 或测试文件。

### 运行时行为

`infer.py` 是一个自解压入口脚本。脚本只负责缓存目录选择、解压和导入共享 bootstrap 运行时；manifest 读取、依赖安装和默认实验包注入由 `taac2026.application.bootstrap.inference_bundle` 处理。它会：

1. 找到同目录下的 `code_package.zip`
2. 解压到 `TAAC_BUNDLE_WORKDIR` 指定目录；如果没设置，则默认使用 `USER_CACHE_PATH` 下的缓存目录
3. 设置 `PYTHONPATH=project/src:project`
4. 读取 `project/.taac_inference_manifest.json`
5. 在当前 Python 环境中执行 `pip install .`
6. 调用 `taac2026.application.evaluation.infer`

### 推理时常用环境变量

| 变量                      | 用途                                    |
| ------------------------- | --------------------------------------- |
| `EVAL_DATA_PATH`          | 待预测 parquet 文件或目录               |
| `EVAL_RESULT_PATH`        | 输出目录，脚本会写入 `predictions.json` |
| `MODEL_OUTPUT_PATH`       | 要加载的 checkpoint 路径                |
| `TAAC_SCHEMA_PATH`        | `schema.json` 路径                      |
| `TAAC_EXPERIMENT`         | 覆盖 manifest 中的实验包路径            |
| `TAAC_BUNDLE_PIP_EXTRAS`  | 需要时给 `pip install .` 加 extras      |
| `TAAC_SKIP_PIP_INSTALL=1` | 跳过 bundle 里的依赖安装                |
| `TAAC_FORCE_EXTRACT=1`    | 强制重新解压代码包                      |
| `TAAC_BUNDLE_WORKDIR`     | 指定 bundle 解压目录                    |
| `TAAC_CODE_PACKAGE`       | 自定义 `code_package.zip` 路径          |

### 本地模拟一次推理 Bundle

```bash
export TAAC_BUNDLE_WORKDIR=/tmp/taac-infer-bundle
export EVAL_DATA_PATH=data/sample_1000_raw/demo_1000.parquet
export EVAL_RESULT_PATH=/tmp/taac-infer-results
export MODEL_OUTPUT_PATH=outputs/quickstart_baseline/best_model/model.safetensors
export TAAC_SCHEMA_PATH=data/sample_1000_raw/schema.json

python outputs/bundles/baseline_inference/infer.py
```

## 维护类实验

维护 / 分析类实验只支持训练 Bundle，不支持推理 Bundle。当前仓库内置的两个例子是：

- `experiments/host_device_info`
- `experiments/online_dataset_eda`

示例：

```bash
uv run taac-package-train \
  --experiment experiments/host_device_info \
  --output-dir outputs/bundles/host_device_info
```

上传后仍然执行同一个 `run.sh`。其中 `host_device_info` 不需要数据集，`online_dataset_eda` 则需要 `TRAIN_DATA_PATH`，通常还会用到 `TAAC_SCHEMA_PATH`。

## 检查 Bundle 是否正确

```bash
# 查看 zip 里有哪些文件
unzip -l outputs/bundles/baseline_training/code_package.zip

# 查看训练 Bundle manifest
unzip -p outputs/bundles/baseline_training/code_package.zip project/.taac_training_manifest.json | python -m json.tool

# 查看推理 Bundle manifest
unzip -p outputs/bundles/baseline_inference/code_package.zip project/.taac_inference_manifest.json | python -m json.tool
```

重点检查：

- `manifest_version` 是否为当前支持的 manifest 版本
- `bundle_kind` 是否为 `training` 或 `inference`
- `bundle_format` 是否分别为 `taac2026-training-v2` 和 `taac2026-inference-v1`
- `bundle_format_version` 是否与格式号匹配
- `framework.version` 是否记录了生成 bundle 的框架版本
- `bundled_experiment_path` 是否指向你期望的实验包
- `entrypoint` 是否分别为 `run.sh` 和 `infer.py`
- `runtime_env` 是否列出了平台需要提供的环境变量
- `compatibility.requires_uv_online` 是否为 `false`

## 依赖原则

- Bundle 运行时默认复用当前 Python / Conda 环境，不在线执行 `uv sync`
- Bundle 的平台适配逻辑集中在 `src/taac2026/infrastructure/platform`，训练和推理入口共享同一套 pip / manifest / experiment 处理契约
- 核心 CUDA / PyTorch / TorchRec / FBGEMM 栈应尽量复用平台预装环境
- 只有缺少纯 Python 依赖时，才考虑让 bundle 通过 `pip install .` 补齐
- 如果必须附带额外 extras，再显式设置 `TAAC_BUNDLE_PIP_EXTRAS`

## 常见问题

### Bundle 里还是旧代码

重新执行对应的打包命令，覆盖旧目录：

```bash
uv run taac-package-train --experiment experiments/<name> --output-dir outputs/bundles/<name>_training
```

必要时加上 `TAAC_FORCE_EXTRACT=1` 重新解压。

### 线上跑错实验包

先检查 manifest 中的 `bundled_experiment_path`，再确认是否设置了 `TAAC_EXPERIMENT` 覆盖它。

### 线上缺少依赖

优先确认平台环境是否已有对应包；缺的是纯 Python 依赖时，再补到 `pyproject.toml` 或通过平台允许的镜像安装。

### 推理脚本找不到缓存目录

推理 Bundle 默认依赖 `USER_CACHE_PATH` 推导缓存位置。若平台没有提供它，请显式设置 `TAAC_BUNDLE_WORKDIR`。
