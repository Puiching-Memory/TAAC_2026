---
icon: lucide/rocket
---

# 快速开始

这页只做一件事：让你从干净环境跑通一次 Baseline 训练、评估和推理，并说明这些命令实际会读写哪些文件。

## 准备环境

需要：

- Linux
- Python 3.10 - 3.13
- `uv`
- Git
- CUDA 12.6，本地 GPU 训练时需要

安装依赖：

```bash
git clone https://github.com/Puiching-Memory/TAAC_2026.git
cd TAAC_2026

uv python install 3.10.20
uv sync --locked --extra cuda126
```

如果你还要跑测试、lint 或文档站：

```bash
uv sync --locked --extra dev --extra cuda126
```

说明：

- `cuda126` 是当前仓库支持的本地 CUDA profile。
- `dev` 包含 Ruff、Vulture、Pytest、Coverage 和 Zensical。
- 线上 bundle 不依赖 `uv`，见 [线上 Bundle 上传](guide/online-training-bundle.md)。

## 数据从哪里来

本地 PCVR smoke 默认使用 Hugging Face 上的 `TAAC2026/data_sample_1000` demo parquet，并使用仓库内的 `data/sample_1000_raw/schema.json`。

本地 PCVR 训练、评估和推理不接受显式 `--dataset-path`；真实比赛数据只在 bundle 模式下由平台环境变量注入。维护类实验例外，例如 Online Dataset EDA 本身就需要指定数据路径。

## 训练 Baseline

```bash
bash run.sh train \
  --experiment experiments/baseline \
  --run-dir outputs/quickstart_baseline \
  --device cpu \
  --num_workers 0 \
  --batch_size 8 \
  --max_steps 1
```

训练完成后，输出目录里会有 checkpoint 和 sidecar。当前 checkpoint 通常位于：

```text
outputs/quickstart_baseline/
└── global_step*.best_model/
    ├── model.safetensors
    ├── schema.json
    └── train_config.json
```

如果你有可用 GPU，可以把 `--device cpu` 改成 `--device cuda`，并适当调大 batch size / max steps。

## 评估训练结果

```bash
bash run.sh val \
  --experiment experiments/baseline \
  --run-dir outputs/quickstart_baseline \
  --device cpu \
  --num-workers 0
```

传 `--run-dir` 时，运行时会在目录下自动寻找最新的 `global_step*.best_model/model.safetensors`。

评估会写出：

```text
outputs/quickstart_baseline/
├── evaluation.json
├── validation_predictions.jsonl
└── evaluation_observed_schema.json
```

`evaluation.json` 里包含 metrics、checkpoint 路径、schema 路径和数据诊断；`validation_predictions.jsonl` 保留逐样本验证预测，便于排查分数异常。

## 生成推理结果

```bash
bash run.sh infer \
  --experiment experiments/baseline \
  --checkpoint outputs/quickstart_baseline \
  --result-dir outputs/quickstart_infer \
  --device cpu \
  --num-workers 0
```

输出文件：

```text
outputs/quickstart_infer/
└── predictions.json
```

`predictions.json` 的顶层 key 是 `predictions`，内部是 `user_id -> probability`：

```json
{
  "predictions": {
    "user_001": 0.8732
  }
}
```

## 换一个实验包

只改 `--experiment`：

```bash
bash run.sh train \
  --experiment experiments/baseline_plus \
  --run-dir outputs/quickstart_baseline_plus \
  --device cpu \
  --num_workers 0 \
  --batch_size 8 \
  --max_steps 1
```

实验选择见 [实验包总览](experiments/index.md)。

## 生成线上 Bundle

训练 bundle：

```bash
uv run taac-package-train \
  --experiment experiments/baseline \
  --output-dir outputs/bundles/baseline_training
```

推理 bundle：

```bash
uv run taac-package-infer \
  --experiment experiments/baseline \
  --output-dir outputs/bundles/baseline_inference
```

训练 bundle 输出 `run.sh + code_package.zip`，推理 bundle 输出 `infer.py + code_package.zip`。上传和本地模拟方式见 [线上 Bundle 上传](guide/online-training-bundle.md)。

## 命令边界

`run.sh` 只支持：

| 命令                                   | 用途                    |
| -------------------------------------- | ----------------------- |
| `bash run.sh train`                    | 训练实验                |
| `bash run.sh val` / `bash run.sh eval` | 本地评估                |
| `bash run.sh infer`                    | 生成 `predictions.json` |

打包、测试和文档站分别使用独立命令：`taac-package-*`、`pytest` 和 `zensical`。

`run.sh` 的分发逻辑在 `src/taac2026/application/bootstrap/run_sh.py`：

- 不写子命令时默认等价于 `train`。
- `train` 最终调用 `taac-train`。
- `val` / `eval` / `infer` 最终调用 `taac-evaluate`。
- 本地默认 runner 是 `uv`；bundle 模式或 `TAAC_RUNNER=python` 会改用当前 Python。
- `--cuda-profile` 目前只接受 `cuda126`，也可以用 `TAAC_CUDA_PROFILE=cuda126`。

训练 CLI 继承了历史参数命名，常见参数是下划线形式：`--num_workers`、`--batch_size`、`--max_steps`。评估和推理 CLI 使用连字符形式：`--num-workers`、`--batch-size`。

## 常见失败

| 现象 | 处理 |
| ---- | ---- |
| `uv is required but not found` | 本地安装 `uv`，或显式设置 `TAAC_RUNNER=python` 并确认依赖已安装 |
| `local PCVR runs no longer accept --dataset-path` | 普通 PCVR 本地 smoke 不传数据路径；维护实验例外 |
| 找不到 checkpoint | 确认 `--run-dir` 或 `--checkpoint` 指向包含 `global_step*.best_model/` 的目录 |
| 推理缺 `train_config.json` 或 `schema.json` | checkpoint 目录不完整，需要使用训练产物目录而不是只拷贝权重 |
| CUDA OOM | 降低 `--batch_size`、序列长度、模型宽度或改用 `--device cpu` 做链路检查 |
