---
icon: lucide/rocket
---

# 快速开始

## 前置要求

- Python 3.10 - 3.13
- CUDA 12.6（GPU 训练需要）
- `uv` 包管理器（推荐）或 `pip`
- Git

## 安装

=== "uv（推荐）"

    ```bash
    git clone https://github.com/Puiching-Memory/TAAC_2026.git
    cd TAAC_2026
    uv sync --extra dev --extra pcvr
    ```

=== "pip"

    ```bash
    git clone https://github.com/Puiching-Memory/TAAC_2026.git
    cd TAAC_2026
    pip install -e ".[dev,pcvr]"
    ```

依赖说明：

- `--extra dev`：Ruff、Pytest、覆盖率工具
- `--extra pcvr`：PyTorch、TorchRec、FBGEMM

## 准备数据

仓库附带 `data/sample_1000_raw/` 含 1000 条示例数据：

```
data/sample_1000_raw/
├── demo_1000.parquet    # 1000 行样本
└── schema.json          # 列定义（特征 FID、词表大小、序列域等）
```

比赛正式数据需从官方平台下载，放置为相同格式。

## 训练第一个模型

```bash
uv run taac-train \
  --experiment config/baseline \
  --dataset-path data/sample_1000_raw/demo_1000.parquet \
  --schema-path data/sample_1000_raw/schema.json
```

训练完成后，Checkpoint 和日志保存在 `outputs/<run-slug>/` 目录下。

## 评估

```bash
uv run taac-evaluate single \
  --experiment config/baseline \
  --dataset-path data/sample_1000_raw/demo_1000.parquet \
  --schema-path data/sample_1000_raw/schema.json
```

## 运行其他实验包

切换 `--experiment` 参数即可：

```bash
uv run taac-train --experiment config/symbiosis \
  --dataset-path data/sample_1000_raw/demo_1000.parquet \
  --schema-path data/sample_1000_raw/schema.json
```

所有实验包位于 `config/` 目录下，每个包含 `__init__.py`、`model.py`、`ns_groups.json`。

## 线上训练打包

```bash
uv run taac-package-train \
  --experiment config/symbiosis \
  --output-dir outputs/bundle
```

生成 `run.sh` + `code_package.zip`，详见 [线上 Bundle 上传指南](guide/online-training-bundle.md)。

## 测试

```bash
uv run pytest tests/unit -v
```

## 本地文档站

```bash
uv run zensical serve
```

默认在 `http://127.0.0.1:8000` 启动开发服务器。

## 统一入口速查

| 命令                                   | 用途               |
| -------------------------------------- | ------------------ |
| `taac-train`                           | 训练实验           |
| `taac-evaluate single`                 | 评估模型           |
| `taac-evaluate infer`                  | 推理（无标签数据） |
| `taac-package-train`                   | 打包训练 Bundle    |
| `taac-package-infer`                   | 打包推理 Bundle    |
| `taac-benchmark-pcvr-data-pipeline`    | 数据管道吞吐压测   |
| `taac-generate-pcvr-synthetic-dataset` | 生成合成压测数据   |
| `taac-plot-model-performance`          | 绘制 Pareto 前沿图 |

运维/分析类实验通过 `taac-train --experiment config/<name>` 或同等的 `bash run.sh train --experiment config/<name>` 调用。

仓库缓存清理和文档结构裁剪保留为 shell 脚本：`bash tools/cache-cleanup.sh`、`bash tools/strip_docs_content.sh docs/<path-or-dir>`。

所有命令均通过 `run.sh` 也可调用：

```bash
./run.sh train --experiment config/baseline --dataset-path ...
./run.sh val   --experiment config/baseline --dataset-path ...
./run.sh infer --experiment config/baseline --dataset-path ...
./run.sh package --experiment config/symbiosis
```
