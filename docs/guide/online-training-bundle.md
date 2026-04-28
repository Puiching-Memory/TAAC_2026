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
├── tools/                         # 工具脚本
├── config/__init__.py
├── config/<experiment>/           # 实验包
│   ├── __init__.py
│   ├── model.py
│   └── ns_groups.json
└── src/taac2026/                  # 框架源码
```

### 训练运行

平台执行 `run.sh`，它会：

1. 检测 Bundle 模式（存在 `code_package.zip`）
2. 解压到临时目录
3. 读取 `.taac_training_manifest.json`
4. 安装依赖（`uv sync`）
5. 执行 `taac-train`

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
