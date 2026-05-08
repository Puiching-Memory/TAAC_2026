---
icon: lucide/activity
---

# PCVR Smoke Diagnostics

本地只有 `demo_1000.parquet` 时，AUC 只能作为 smoke signal。更可靠的图应该看运行成本、预测行为、模型间差异和 seed 稳定性。

## 生成方式

先发现当前仓库支持的 PCVR 模型实验包。`host_device_info`、`online_dataset_eda` 这类 maintenance/EDA 包不会产出 PCVR 预测文件，不纳入这组图。

```bash
uv run python - <<'PY'
from pathlib import Path

from taac2026.application.experiments.discovery import discover_experiment_paths
from taac2026.application.experiments.registry import load_experiment_package

for path in discover_experiment_paths(Path("experiments")):
    experiment = load_experiment_package(path)
    if experiment.metadata.get("kind") == "pcvr":
        print(path)
PY
```

然后分别训练、评估和推理这些实验包。下面命令会对所有 PCVR 模型包生成 `outputs/smoke/<name>_seed42`；本地 demo smoke 显式使用 torch 后端和 `--no-compile`，避免把 TileLang/NVCC 或 `torch.compile` 的一次性编译成本混进诊断图。

```bash
export CUDA_VISIBLE_DEVICES=0
SCHEMA="outputs/perf/pcvr_synthetic_300x/schema.json"

for exp in baseline baseline_plus interformer onetrans symbiosis unitok; do
  run_dir="outputs/smoke/${exp}_seed42"
  bash run.sh train \
    --experiment "experiments/${exp}" \
    --run-dir "$run_dir" \
    --schema-path "$SCHEMA" \
    --seed 42 \
    --num_workers 0 \
    --flash-attention-backend torch \
    --rms-norm-backend torch \
    --no-compile

  checkpoint="$(find "$run_dir" -mindepth 2 -maxdepth 2 -name model.safetensors | sort | tail -n 1)"

  bash run.sh val \
    --experiment "experiments/${exp}" \
    --run-dir "$run_dir" \
    --schema-path "$SCHEMA" \
    --num-workers 0 \
    --no-compile

  bash run.sh infer \
    --experiment "experiments/${exp}" \
    --schema-path "$SCHEMA" \
    --checkpoint "$checkpoint" \
    --result-dir "$run_dir" \
    --num-workers 0 \
    --no-compile
done
```

然后把多个 run 目录交给诊断绘图命令：

```bash
uv run taac-plot-pcvr-diagnostics \
  --run baseline=outputs/smoke/baseline_seed42 \
  --run baseline_plus=outputs/smoke/baseline_plus_seed42 \
  --run interformer=outputs/smoke/interformer_seed42 \
  --run onetrans=outputs/smoke/onetrans_seed42 \
  --run symbiosis=outputs/smoke/symbiosis_seed42 \
  --run unitok=outputs/smoke/unitok_seed42 \
  --output-dir figures/pcvr_diagnostics
```

默认命令只在每个 run 目录都已经有 `evaluation.json` 和 `validation_predictions.jsonl` 时生成图；如果这些文件缺失，CLI 会直接报错并给出需要先跑的 `bash run.sh val ...` 命令。完整机器可读摘要写入 `pcvr_diagnostics_summary.json`，终端只打印精简报告；需要把完整 JSON 打到 stdout 时加 `--json`。

输出目录会包含：

| 文件 | 含义 |
| ---- | ---- |
| `pcvr_runtime_resources.svg` | 训练耗时、评估/推理吞吐、CPU / CUDA 峰值资源占用 |
| `pcvr_prediction_distribution.svg` | 每个模型的预测概率分布，带正负样本对比 |
| `pcvr_prediction_correlation.svg` | 模型间逐样本预测相关性热力图 |
| `pcvr_sample_disagreement.svg` | 样本级模型预测分歧和 top disagreement 样本 |
| `pcvr_stability.svg` | 按实验或标签分组的 AUC、LogLoss、预测 std、评估耗时稳定性 |
| `pcvr_diagnostics_summary.json` | 绘图所用 run、metrics、telemetry 和图路径摘要 |

## 输入约定

每个 `--run` 可以是目录，也可以是 `label=目录`。目录下优先读取这些文件：

- `evaluation.json`
- `validation_predictions.jsonl`
- `training_summary.json`
- `training_telemetry.json`
- `evaluation_telemetry.json`
- `inference_telemetry.json`

`bash run.sh train` 会写 `training_summary.json` 和 `training_telemetry.json`。`bash run.sh val` 会写 `evaluation.json`、`validation_predictions.jsonl` 和 `evaluation_telemetry.json`。`bash run.sh infer` 会在 result dir 写 `predictions.json` 和 `inference_telemetry.json`。

如果你只是想检查路径或预览占位图，可以加 `--allow-partial`，但这种输出不应该用于分析：

```bash
uv run taac-plot-pcvr-diagnostics \
  --run baseline=outputs/smoke/baseline_seed42 \
  --output-dir figures/pcvr_diagnostics \
  --allow-partial
```

## 稳定性分组

默认按 `evaluation.json` 里的 `experiment_name` 分组。多 seed 跑法可以这样：

```bash
uv run taac-plot-pcvr-diagnostics \
  --run baseline_seed1=outputs/smoke/baseline_seed1 \
  --run baseline_seed2=outputs/smoke/baseline_seed2 \
  --run unitok_seed1=outputs/smoke/unitok_seed1 \
  --run unitok_seed2=outputs/smoke/unitok_seed2 \
  --group-by label-prefix \
  --output-dir figures/pcvr_diagnostics
```

`--group-by label-prefix` 会把 `baseline_seed1`、`baseline_seed2` 归到 `baseline`。如果想完全按标签分组，用 `--group-by label`。

## 解读原则

- `runtime_resources` 适合回答“这个实验在本地 smoke 中是否太慢、太吃资源”。
- `prediction_distribution` 适合看输出是否塌缩到 0.5、0 或 1。
- `prediction_correlation` 适合判断两个模型是否真的给出了不同排序信号。
- `sample_disagreement` 适合抽样排查模型意见分歧最大的样本。
- `stability` 在 demo1000 上比单次 AUC 更重要；多 seed 下优先看方差和预测分布是否乱飘。

这些图仍然不是正式 leaderboard 结论。它们的定位是本地 smoke benchmark 的工程诊断面板。
