---
icon: lucide/rocket
---

# 快速开始

## 前置要求

- Python 3.10 - 3.13
- CUDA 12.6（GPU 训练需要）
- `uv` 包管理器（推荐）；`pip` 仅适用于已准备好兼容 Python / CUDA / PyTorch 环境的场景
- Git

## 安装

=== "uv（推荐）"

    ```bash
    git clone https://github.com/Puiching-Memory/TAAC_2026.git
    cd TAAC_2026
    uv sync --extra dev --extra cuda126
    ```

=== "pip（仅限已备好兼容 GPU 环境）"

    ```bash
    git clone https://github.com/Puiching-Memory/TAAC_2026.git
    cd TAAC_2026
  python -m pip install -e ".[dev]"
  python -m pip install --extra-index-url https://download.pytorch.org/whl/cu126 \
    torch==2.7.1+cu126 fbgemm-gpu==1.2.0+cu126 torchrec==1.2.0+cu126
    ```

依赖说明：

- `--extra dev`：Ruff、Pytest、Hypothesis、Benchmark、覆盖率插件、Zensical
- `cuda126`：通过 `uv` 的 `pytorch-cu126` 索引解析 PyTorch、TorchRec、FBGEMM；如果不用 `uv`，需要像上面那样单独安装匹配的 CUDA 12.6 wheel，或直接复用已准备好的兼容环境

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
  --experiment experiments/pcvr/baseline \
  --dataset-path data/sample_1000_raw/demo_1000.parquet \
  --schema-path data/sample_1000_raw/schema.json
```

训练完成后，Checkpoint 和日志保存在 `outputs/<run-slug>/` 目录下。

## 评估

```bash
uv run taac-evaluate single \
  --experiment experiments/pcvr/baseline \
  --dataset-path data/sample_1000_raw/demo_1000.parquet \
  --schema-path data/sample_1000_raw/schema.json
```

## 运行其他实验包

切换 `--experiment` 参数即可：

```bash
uv run taac-train --experiment experiments/pcvr/symbiosis \
  --dataset-path data/sample_1000_raw/demo_1000.parquet \
  --schema-path data/sample_1000_raw/schema.json
```

PCVR 实验包位于 `experiments/pcvr/` 目录下，每个包含 `__init__.py` 和 `model.py`，其中 NS 分组在 `__init__.py` 的 `PCVRNSConfig` 中显式声明；运维/分析实验位于 `experiments/maintenance/`。

## 线上训练打包

```bash
uv run taac-package-train \
  --experiment experiments/pcvr/symbiosis \
  --output-dir outputs/bundle
```

生成 `run.sh` + `code_package.zip`，详见 [线上 Bundle 上传指南](guide/online-training-bundle.md)。

## 测试

```bash
# 已执行 uv sync --extra dev --extra cuda126
uv run pytest tests/unit -v
```

## 本地文档站

```bash
# 已执行 uv sync --extra dev
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

运维/分析类实验通过 `taac-train --experiment experiments/maintenance/<name>` 或同等的 `bash run.sh train --experiment experiments/maintenance/<name>` 调用。

仓库缓存清理和文档结构裁剪保留为 shell 脚本：`bash tools/cache-cleanup.sh`、`bash tools/strip_docs_content.sh docs/<path-or-dir>`。

所有命令均通过 `run.sh` 也可调用：

```bash
./run.sh train --experiment experiments/pcvr/baseline --dataset-path ...
./run.sh val   --experiment experiments/pcvr/baseline --dataset-path ...
./run.sh infer --experiment experiments/pcvr/baseline --dataset-path ...
./run.sh package --experiment experiments/pcvr/symbiosis
```
