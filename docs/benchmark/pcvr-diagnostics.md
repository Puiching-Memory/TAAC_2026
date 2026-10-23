---
icon: lucide/chart-no-axes-combined
---

# PCVR 诊断：成本、预测与稳定性

小样本 smoke 更适合发现流程、输出分布和成本异常，而非产生模型排行榜。本页将训练、独立评估和推理产物组合成可比较图表。

## 先收集同口径产物

对每个模型和 seed 完成[快速开始](../getting-started.md)的训练、评估、推理，使用同一数据文件、schema 和实际样本范围。CPU 检查关闭 AMP/compile；GPU 对照则保持这些设置一致。

| 文件 | 用途 |
| --- | --- |
| `training_summary.json` | 训练留出验证指标、切分与配置摘要 |
| `training_telemetry.json` | 训练成本 |
| `evaluation.json` | 评估范围与指标 |
| `validation_predictions.jsonl` | 逐样本预测与对齐 |
| `evaluation_telemetry.json` | 评估成本 |
| `inference_telemetry.json` | 推理成本，通常位于推理 result dir |

绘图器从输入 run 目录直接读取 `inference_telemetry.json`。需要比较推理时，让 infer 的 `--result-dir` 使用同一 run 目录，或把独立 result dir 中的这份文件复制到对应 run 目录；不要把缺失时间填成零。

## 生成图表

```bash
uv run taac-plot-pcvr-diagnostics \
  --run baseline_seed1=outputs/smoke/baseline_seed1 \
  --run baseline_seed2=outputs/smoke/baseline_seed2 \
  --run tokenformer_seed1=outputs/smoke/tokenformer_seed1 \
  --run tokenformer_seed2=outputs/smoke/tokenformer_seed2 \
  --group-by label-prefix --output-dir outputs/benchmarks/pcvr_diagnostics
```

`label-prefix` 合并同模型 seeds；`label` 把每个输入视为独立组；默认按 experiment name 分组。分组会影响资源、预测聚合、相关性、分歧和稳定性全部图表。

## 图表应回答什么

| 图表 | 应观察的问题 |
| --- | --- |
| runtime resources | 质量接近时，时间、参数、内存的代价 |
| prediction distribution | 正负分布、置信度与输出塌缩 |
| prediction correlation | 模型排序信号与跨 seed 稳定性 |
| sample disagreement | 哪些样本值得回看原始字段 |
| stability | 指标离散和逐样本 prediction drift |

质量指标优先来自训练摘要 `validation_metrics`，缺失时回退独立评估并标注来源。`±` 为 sample SD，误差线为 min–max。预测比较始终基于 JSONL，先按样本键对齐，同组 seeds 取均值；相关性使用 Spearman。

只有所有模型组都有 inference telemetry 时才比较推理成本，否则使用评估成本。单独执行的 `val` 可能扫描整个输入，并不等同于训练内的 held-out 验证，必须看 metric_source。

## 分析边界

默认拒绝缺少评估产物的输入。`--allow-partial` 仅用于路径诊断和预览，不适合质量结论。少量 demo 样本、多次重复数据或单 seed 的高 AUC，都不能代替独立数据泛化验证。

更深的学习轨迹分析见[研究笔记](../ideas/posts/003-learning-mechanics-analysis.md)，通用计时见[性能指南](../guide/performance-benchmarks.md)。
