---
icon: lucide/gauge
---

# 性能基准

数据管道吞吐量基准测试结果。

## 当前状态

基准测试图表已生成（`docs/assets/figures/benchmarks/`，6 个 ECharts JSON）。原始数据来自 `tools/benchmark_pcvr_data_pipeline.py`。

测试场景：

| Preset    | 说明                                 |
| --------- | ------------------------------------ |
| `none`    | 无增强，纯数据加载                   |
| `cache`   | 启用 `PCVRMemoryBatchCache`          |
| `augment` | 启用序列裁剪 + 域 Dropout + 特征掩码 |

## 恢复基准套件时

运行基准测试：

```bash
python tools/benchmark_pcvr_data_pipeline.py \
  --dataset-path data/sample_1000_raw/demo_1000.parquet \
  --schema-path data/sample_1000_raw/schema.json \
  --preset none

python tools/benchmark_pcvr_data_pipeline.py \
  --dataset-path data/sample_1000_raw/demo_1000.parquet \
  --schema-path data/sample_1000_raw/schema.json \
  --preset cache

python tools/benchmark_pcvr_data_pipeline.py \
  --dataset-path data/sample_1000_raw/demo_1000.parquet \
  --schema-path data/sample_1000_raw/schema.json \
  --preset augment
```

指标：samples/sec、batch latency (ms)、GPU utilization (%)。

生成报告图表：

```bash
uv run taac-bench-report \
  --benchmark-dir outputs/benchmarks \
  --output-dir docs/assets/figures/benchmarks
```
