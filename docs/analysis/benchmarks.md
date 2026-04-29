---
icon: lucide/gauge
---

# 性能基准

数据管道吞吐量基准测试结果。

## 当前状态

基准测试图表已生成（`docs/assets/figures/benchmarks/`，6 个 ECharts JSON）。原始数据来自原生 CLI `taac-benchmark-pcvr-data-pipeline`。

测试场景：

| Preset    | 说明                                 |
| --------- | ------------------------------------ |
| `none`    | 无增强，纯数据加载                   |
| `cache`   | 启用 `PCVRMemoryBatchCache`          |
| `augment` | 启用序列裁剪 + 域 Dropout + 特征掩码 |

## 恢复基准套件时

运行基准测试：

```bash
uv run taac-benchmark-pcvr-data-pipeline \
  --dataset-path data/sample_1000_raw/demo_1000.parquet \
  --schema-path data/sample_1000_raw/schema.json \
  --preset none

uv run taac-benchmark-pcvr-data-pipeline \
  --dataset-path data/sample_1000_raw/demo_1000.parquet \
  --schema-path data/sample_1000_raw/schema.json \
  --preset cache

uv run taac-benchmark-pcvr-data-pipeline \
  --dataset-path data/sample_1000_raw/demo_1000.parquet \
  --schema-path data/sample_1000_raw/schema.json \
  --preset augment
```

指标：samples/sec、batch latency (ms)、GPU utilization (%)。

推荐把每次运行的 JSON 输出重定向到 `outputs/benchmarks/`，后续再统一做图表汇总：

```bash
mkdir -p outputs/benchmarks

uv run taac-benchmark-pcvr-data-pipeline \
  --dataset-path data/sample_1000_raw/demo_1000.parquet \
  --schema-path data/sample_1000_raw/schema.json \
  --preset none \
  > outputs/benchmarks/none.json
```
