---
icon: lucide/rocket
---

# 快速开始：完成训练、评估和推理

本页用 Baseline 和公开 demo 数据验证文件流转。一次训练步只检查流程，不能据此比较模型质量。

## 1. 准备 Linux 环境

项目要求 Linux、Git、`uv`，以及 Python 3.12 - 3.14。GPU 路径另需匹配 `cuda132` 的 NVIDIA 驱动与 CUDA 工具链；`pyproject.toml` 的 uv 环境限定为 Linux，原生 Windows 训练不在当前支持范围内。

```bash
git clone https://github.com/Puiching-Memory/TAAC_2026.git
cd TAAC_2026
```

选择一种依赖环境：

| 用途 | 命令 |
| --- | --- |
| GPU 训练 | `uv sync --locked --extra cuda132` |
| GPU 开发与测试 | `uv sync --locked --extra dev --extra cuda132` |
| CPU 流程检查 | `uv sync --locked --extra dev`，后续命令显式加 `--with torch==2.13.0` |
| 只构建文档 | `uv sync --locked --extra dev` |

Torch 由 CUDA extra 或 `uv run --with` 提供，不在基础依赖中。`.python-version` 指定本地默认解释器；复现 CI 时用 `--python 3.13` 显式选择版本。

## 2. 准备数据和 schema

```bash
mkdir -p data/sample_1000_raw
uv run python - <<'PY'
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="TAAC2026/data_sample_1000",
    repo_type="dataset",
    filename="demo_1000.parquet",
    local_dir="data/sample_1000_raw",
)
PY
cp docs/archive/files/schema/sample_1000_raw.schema.json data/sample_1000_raw/schema.json
```

下载需要访问 [Hugging Face 样例数据集](https://huggingface.co/datasets/TAAC2026/data_sample_1000)。后续命令固定使用这个本地文件，不会在默认缓存与自定义目录之间混用。

不传 `--dataset-path` 时，普通 PCVR 实验会自动下载 demo；同时仍需要可用的 schema。默认 schema 查找位置是 `data/sample_1000_raw/schema.json`，不会自动改用 `docs/archive/`。自定义 parquet 必须配套实际 schema，不能直接套用 demo 的词表和字段长度。

## 3. 在 CPU 上训练一步

```bash
uv run --with torch==2.13.0 taac-train \
  --experiment experiments/baseline \
  --dataset-path data/sample_1000_raw/demo_1000.parquet \
  --schema-path data/sample_1000_raw/schema.json \
  --run-dir outputs/quickstart_baseline \
  --optimizer.device cpu \
  --data.num_workers 0 \
  --data.batch_size 8 \
  --optimizer.max_steps 1 \
  --runtime.no_amp \
  --runtime.no_compile
```

`taac-train` 外层参数使用连字符；实验训练配置使用带分组前缀的下划线形式，例如 `--data.batch_size`。不要写成 `--batch-size`。

检查输出目录：

```text
outputs/quickstart_baseline/
├── global_step*/
│   ├── model.safetensors
│   ├── schema.json
│   └── train_config.json
├── training_summary.json
└── training_telemetry.json
```

权重和两份 sidecar 应一起保存。评估、推理使用训练时保存的配置重建模型，不会用当前实验默认值补全缺失字段。

## 4. 评估 checkpoint

```bash
uv run --with torch==2.13.0 taac-evaluate single \
  --experiment experiments/baseline \
  --dataset-path data/sample_1000_raw/demo_1000.parquet \
  --run-dir outputs/quickstart_baseline \
  --device cpu --num-workers 0 --batch-size 8
```

`--run-dir` 会按 step 选择目录下最新的 `global_step*/model.safetensors`。默认 schema 来自该 checkpoint 目录。输出包含 `evaluation.json`、`validation_predictions.jsonl`、`evaluation_observed_schema.json` 和 `evaluation_telemetry.json`。

这条独立评估命令扫描所给数据，不能当作训练时 held-out 切分的分数。模型对比应固定独立验证数据，并核对训练摘要里的切分信息。

## 5. 推理并检查结果

先从评估报告读取它实际使用的权重路径，避免依赖目录名排序：

```bash
CHECKPOINT="$(uv run python - <<'PY'
import json
from pathlib import Path
payload = json.loads(Path("outputs/quickstart_baseline/evaluation.json").read_text())
print(payload["checkpoint_path"])
PY
)"

uv run --with torch==2.13.0 taac-evaluate infer \
  --experiment experiments/baseline \
  --dataset-path data/sample_1000_raw/demo_1000.parquet \
  --checkpoint "$CHECKPOINT" \
  --result-dir outputs/quickstart_infer \
  --device cpu --num-workers 0 --batch-size 8
```

显式 `--checkpoint` 接受权重文件，或直接包含 `model.safetensors` 的目录；不要传仅包含多个 `global_step*/` 子目录的 run 根目录。

`outputs/quickstart_infer/predictions.json` 的结构是：

```json
{"predictions": {"user_001": 0.8732}}
```

推理还会写 `inference_telemetry.json`。映射按 user id 存储，同一 id 不能表示多个独立候选记录；自定义数据应检查 key 的唯一性与平台约定。

## 6. 切换到 GPU 或其他实验

GPU 环境同步 `cuda132` 后，可使用统一 shell 入口：

```bash
bash run.sh train \
  --experiment experiments/baseline \
  --dataset-path data/sample_1000_raw/demo_1000.parquet \
  --schema-path data/sample_1000_raw/schema.json \
  --run-dir outputs/baseline_gpu \
  --optimizer.device cuda
```

`run.sh` 本地默认会同步 CUDA extra；CPU 检查使用上面的 console scripts 更直接。shell 入口支持 `train`、`val`/`eval`、`infer`，不写子命令时为训练。不同实验只需替换 `--experiment`，但默认模型尺寸、增强、AMP 和 compile 可能不同，先查看[实验表](experiments/index.md)。

## 7. 生成上传物

```bash
uv run --with torch==2.13.0 taac-package-train \
  --experiment experiments/baseline --output-dir outputs/bundles/baseline_training
uv run --with torch==2.13.0 taac-package-infer \
  --experiment experiments/baseline --output-dir outputs/bundles/baseline_inference
```

训练上传 `run.sh + code_package.zip`，推理上传 `infer.py + code_package.zip`。继续阅读 [Bundle 指南](guide/online-training-bundle.md)，先核对平台 Python 与依赖要求。归档平台 Python 3.10 环境不满足当前包的 Python 下限。

## 排障

| 现象 | 检查位置 |
| --- | --- |
| `No module named torch` | 当前命令是否带 CUDA extra 或 `--with torch==2.13.0` |
| 默认 schema 不存在 | 准备 `data/sample_1000_raw/schema.json` 或传显式路径 |
| CLI 拒绝参数 | 区分训练 `--data.batch_size` 与评估 `--batch-size` |
| 找不到权重 | 评估传 `--run-dir`；推理显式传具体权重或 checkpoint 目录 |
| sidecar 校验失败 | 保留三件套，核对产物与代码版本，见[架构](architecture.md) |
| CUDA OOM | 先降低 batch、序列上限或宽度，见[梯度检查点](guide/gradient-checkpointing.md) |
