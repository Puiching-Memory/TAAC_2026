---
icon: lucide/gauge
---

# 性能基准

数据管道吞吐量基准测试结果。

## 当前状态

基准测试图表已生成（`docs/assets/figures/benchmarks/`，6 个 ECharts JSON）。原始数据来自原生 CLI `taac-benchmark-pcvr-data-pipeline`。

性能测试统一使用由 demo 数据放大得到的增广数据集，不再直接用 1000 行 demo 数据衡量吞吐。当前最低口径是 `300x`，即从 `data/sample_1000_raw/demo_1000.parquet` 生成至少 300,000 行的测试集。

测试场景：

| Preset        | 说明                                                    |
| ------------- | ------------------------------------------------------- |
| `none`        | 无增强，纯数据加载                                      |
| `cache`       | 启用 `PCVRMemoryBatchCache`                             |
| `opt`         | 启用 OPT batch cache                                    |
| `augment`     | 启用序列裁剪 + 域 Dropout + 特征掩码                    |
| `opt-augment` | 启用 OPT batch cache + 序列裁剪 + 域 Dropout + 特征掩码 |

## 恢复基准套件时

先生成 300x 增广数据集：

```bash
uv run taac-generate-pcvr-synthetic-dataset \
  --source-dir data/sample_1000_raw \
  --output-dir outputs/perf/pcvr_synthetic_300x \
  --multiplier 300 \
  --force
```

再运行基准测试：

```bash
uv run taac-benchmark-pcvr-data-pipeline \
  --dataset-path outputs/perf/pcvr_synthetic_300x/demo_300000.parquet \
  --schema-path outputs/perf/pcvr_synthetic_300x/schema.json \
  --preset none

uv run taac-benchmark-pcvr-data-pipeline \
  --dataset-path outputs/perf/pcvr_synthetic_300x/demo_300000.parquet \
  --schema-path outputs/perf/pcvr_synthetic_300x/schema.json \
  --preset opt

uv run taac-benchmark-pcvr-data-pipeline \
  --dataset-path outputs/perf/pcvr_synthetic_300x/demo_300000.parquet \
  --schema-path outputs/perf/pcvr_synthetic_300x/schema.json \
  --preset opt-augment \
  --passes 3
```

指标：samples/sec、batch latency (ms)、GPU utilization (%)。

推荐把每次运行的 JSON 输出重定向到 `outputs/benchmarks/`，后续再统一做图表汇总：

```bash
mkdir -p outputs/benchmarks

uv run taac-benchmark-pcvr-data-pipeline \
  --dataset-path outputs/perf/pcvr_synthetic_300x/demo_300000.parquet \
  --schema-path outputs/perf/pcvr_synthetic_300x/schema.json \
  --preset none \
  > outputs/benchmarks/none.json
```
