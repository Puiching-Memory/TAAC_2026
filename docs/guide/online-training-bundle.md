---
icon: lucide/package
---

# 线上 Bundle 上传指南

使用 `taac-package-train` 和 `taac-package-infer` 生成符合比赛平台要求的 Bundle。

## 训练 Bundle

### 生成

```bash
uv run taac-package-train \
  --experiment config/symbiosis \
  --output-dir outputs/bundle
```

### 输出

```
outputs/bundle/
├── run.sh              # 平台执行入口
└── code_package.zip    # 代码包
```

### 代码包内容

```
code_package.zip
├── .taac_training_manifest.json   # Bundle 元信息
├── pyproject.toml                 # 依赖声明
├── uv.lock                        # 依赖锁定
├── README.md
├── config/__init__.py
├── config/<experiment>/           # 实验包
│   ├── __init__.py
│   ├── model.py / task 入口
│   └── ns_groups.json             # 仅 PCVR 训练实验需要
└── src/taac2026/                  # 框架源码
```

### 训练运行

平台执行 `run.sh`，它会：

1. 检测 Bundle 模式（存在 `code_package.zip`）
2. 解压到临时目录
3. 读取 `.taac_training_manifest.json`
4. 安装项目依赖（Bundle 模式默认使用当前 Python 的 `pip install .`）
5. 执行 `taac-train`

### 运维/分析类实验

除 PCVR 模型训练外，以下任务也可打包成同样的双文件 Bundle：

- `config/host_device_info`：采集主机、GPU、网络、依赖源探测等环境信息，结果直接打印到日志
- `config/online_dataset_eda`：对线上 parquet 做 EDA，结果直接打印到日志

示例：

```bash
uv run taac-package-train \
  --experiment config/host_device_info \
  --output-dir outputs/bundles/host_device_info

uv run taac-package-train \
  --experiment config/online_dataset_eda \
  --output-dir outputs/bundles/online_dataset_eda
```

上传后仍然执行同一个 `run.sh`：

```bash
# host/device info 不需要数据集路径
bash run.sh

# online dataset EDA 需要数据集和 schema
export TAAC_DATASET_PATH=/path/to/train.parquet_or_dir
export TAAC_SCHEMA_PATH=/path/to/schema.json
bash run.sh
```

如需限制扫描行数或调整批大小，请直接编辑 [config/online_dataset_eda/__init__.py](config/online_dataset_eda/__init__.py) 中的 `ONLINE_DATASET_EDA_CONFIG`。

## 推理 Bundle

### 生成

```bash
uv run taac-package-infer \
  --experiment config/symbiosis \
  --output-dir outputs/bundle
```

### 输出

```
outputs/bundle/
├── infer.py            # 自解压推理脚本
└── code_package.zip    # 代码包（含 Checkpoint）
```

### 推理流程

`infer.py` 是自包含脚本：

1. 解压 `code_package.zip`
2. 安装依赖
3. 读取环境变量（`EVAL_DATA_PATH`、`EVAL_RESULT_PATH`、`MODEL_OUTPUT_PATH`、`TAAC_SCHEMA_PATH`）
4. 加载模型和 Checkpoint
5. 运行推理
6. 输出 `predictions.json` 到 `EVAL_RESULT_PATH`

## 与官方参考 Baseline 的关系

Bundle 格式与官方 Baseline 兼容：

- `run.sh` 支持平台的环境变量约定
- `infer.py` 输出标准 `predictions.json` 格式
- `pyproject.toml` 声明所有必要依赖

## pyproject 依赖安装

平台使用 `uv` 安装依赖：

```bash
uv sync --extra pcvr
```

确保 `pyproject.toml` 中的依赖在平台上可安装。如果使用自定义依赖，需要在 Bundle 中包含。

## 检查 Bundle

```bash
# 查看 Bundle 内容
unzip -l outputs/bundle/code_package.zip

# 查看训练 manifest
unzip -p outputs/bundle/code_package.zip .taac_training_manifest.json | python -m json.tool

# 本地测试训练 Bundle
cd /tmp && bash outputs/bundle/run.sh

# 本地测试推理 Bundle
EVAL_DATA_PATH=data/sample_1000_raw/demo_1000.parquet \
EVAL_RESULT_PATH=/tmp/results \
MODEL_OUTPUT_PATH=outputs/pcvr_symbiosis-*/global_step*.best_model \
TAAC_SCHEMA_PATH=data/sample_1000_raw/schema.json \
python outputs/bundle/infer.py
```

## 依赖原则

- 框架依赖通过 `pyproject.toml` 声明，不要手动 pip install
- 使用 `--extra pcvr` 安装 PyTorch 相关依赖
- 避免依赖本地编译的 C++ 扩展

## 常见问题

### 找不到模块

检查 `model_class_name` 是否与 `model.py` 中的类名一致。

### 使用了旧代码

重新生成 Bundle：`uv run taac-package-train --experiment config/<name> --output-dir outputs/bundle`

### 跑错实验包

检查 `--experiment` 参数是否指向正确的实验包目录。

### 线上缺少依赖

在 `pyproject.toml` 中添加缺失的依赖，重新生成 Bundle。
