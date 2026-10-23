---
icon: lucide/gauge
---

# 性能测量：从可运行到可比较

性能报告至少记录 commit、完整命令、硬件、Python/Torch/CUDA、数据规模、精度、shape、warmup、重复次数与数值误差。先确认计时边界，再比较速度。

## 环境与 JSON 输出

CUDA 测量使用已同步的 `cuda132` 环境；CPU 通路命令可在 `uv run` 后加 `--with torch==2.13.0`。下面所有重定向到 JSON 的 benchmark 命令均显式加 `--json`；不加时 CLI 默认是面向人的表格摘要。

```bash
mkdir -p outputs/benchmarks
```

## 生成可重复的合成负载

先完成[样例数据准备](../getting-started.md)，确保目录内同时有 demo parquet 和 `schema.json`：

```bash
uv run taac-generate-pcvr-synthetic-dataset \
  --source-dir data/sample_1000_raw \
  --output-dir outputs/perf/pcvr_synthetic_300x \
  --multiplier 300 --json
```

输出含 `demo_300000.parquet` 和 schema。该生成器复制样本并作部分 offset，适合系统负载检查，不增加真实数据多样性，不能用其 AUC 评估泛化。`--force` 会替换输出目录，只对专用可再生目录使用。

## 数据管道

```bash
uv run taac-benchmark-pcvr-data-pipeline \
  --dataset-path outputs/perf/pcvr_synthetic_300x/demo_300000.parquet \
  --schema-path outputs/perf/pcvr_synthetic_300x/schema.json \
  --preset none --passes 3 --json \
  > outputs/benchmarks/data_pipeline_none.json
```

| preset | 默认行为 |
| --- | --- |
| `none` | 无 cache，无增强 |
| `cache` | LRU，无增强 |
| `opt` | OPT，无增强 |
| `augment` | LRU + 裁剪/mask/domain dropout |
| `opt-augment` | OPT + 增强 |

只测增强时给 `augment` 加 `--cache-mode none`。同时记录 workers、cache 容量、采样策略、命中率、实际行数和 RSS；冷启动与稳态分开。

`--benchmark-mode convert --converter-batches 256 --warmup-batches 16` 会预加载 Arrow batch 后只测转换，适合隔离 `_convert_batch()`，不代表磁盘或 DataLoader 吞吐。

## Optimizer

```bash
uv run taac-benchmark-pcvr-optimizer \
  --device cuda --batch-size 512 --feature-dim 128 \
  --hidden-dim 512 --depth 4 --steps 50 --warmup-steps 10 \
  --repeats 5 --optimizers adamw,orthogonal_adamw,muon --json \
  > outputs/benchmarks/optimizer_cuda.json
```

这是合成 MLP 的更新成本，不覆盖真实 sparse embedding、数据加载或模型收敛。优化器行为见[优化器指南](optimizers.md)。

## Accelerator

```bash
uv run taac-benchmark-pcvr-tilelang-ops \
  --operator rms_norm --device cuda --rows 8192 --cols 128 \
  --dtype float16 --backends torch,tilelang,triton \
  --steps 100 --warmup-steps 20 --repeats 5 --json \
  > outputs/benchmarks/rms_norm.json
```

其他算子的形状和 backend 限制分别见 [LayerNorm](../benchmark/layer-norm.md)、[Flash Attention](../benchmark/flash-attention.md)、[Embedding Bag Mean](../benchmark/embedding-bag-mean.md)。`unsupported`、准确率失败和有效加速是不同结果，不能把缺失 backend 当作零耗时。

先核对绝对误差、相对误差与梯度，再看稳态时间。reference 接近零时相对误差会放大，应结合容差和绝对误差解释。首次 JIT、Python launch 与完整模型时间分开报告。

## 模型诊断

训练产生 `training_telemetry.json`，独立评估产生 `evaluation.json`、`validation_predictions.jsonl` 与 `evaluation_telemetry.json`，推理产生 `inference_telemetry.json`。收集同一验证样本后：

```bash
uv run taac-plot-pcvr-diagnostics \
  --run baseline=outputs/smoke/baseline_seed42 \
  --run interformer=outputs/smoke/interformer_seed42 \
  --group-by label --output-dir outputs/benchmarks/diagnostics
```

输入目录须已有评估产物，不能靠缺失数据生成“完成”图表。样本相关性、分歧和分桶指标需先确认对齐；无 inference telemetry 时使用评估成本，不能把缺失推理时间记作零。完整流程见[诊断页](../benchmark/pcvr-diagnostics.md)。

## 如何发布结论

同机器同口径保存 before/after JSON，报告相对变化和离散程度。算子加速不等于完整训练加速；缓存命中率改善不等于吞吐改善；CPU 通路不外推 CUDA。历史表格只用于追溯，当前代码需重新运行。

CLI 源码位于 `src/taac2026/application/benchmarking/`。历史测量与候选优化见 [Benchmark 索引](../benchmark/index.md)。
