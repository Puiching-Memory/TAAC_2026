---
icon: lucide/gauge
---

# 性能基准

本页展示仓库 benchmark 套件生成的 ECharts 图表。图表源数据来自两类产物：

- `pytest-benchmark` 导出的运行结果
- `outputs/performance/` 下的补充 JSON 统计

如果当前工作区尚未执行 benchmark，本页会显示占位图；运行下面的命令后即可刷新：

```bash
uv run pytest tests/benchmarks -o python_files='bench_*.py' --benchmark-json=benchmark-result.json -v
uv run taac-bench-report --input benchmark-result.json --summary-path docs/assets/figures/benchmarks/benchmark_acceptance.json
```

除了图表外，`taac-bench-report` 还会生成验收摘要 JSON：

- `assets/figures/benchmarks/benchmark_acceptance.json`

这个摘要会聚合 phase 基准数据，并给出 embedding 吞吐、attention 延迟和量化记录是否满足路线图验收条件。

当命令行没有显式传入统一的 candidate phase 时，摘要会按组件自动解析最近可用的候选 phase：例如 embedding 默认读取 phase-2 记录，attention 默认读取 phase-3 记录，量化记录继续使用 phase-6。

## 组件延迟对比

<div class="echarts" data-src="assets/figures/benchmarks/component_latency.echarts.json"></div>

## Embedding 吞吐趋势

<div class="echarts" data-src="assets/figures/benchmarks/throughput_trend.echarts.json"></div>

## 端到端训练步延迟

<div class="echarts" data-src="assets/figures/benchmarks/e2e_train_step.echarts.json"></div>

## 推理延迟分布

<div class="echarts" data-src="assets/figures/benchmarks/inference_boxplot.echarts.json"></div>

## 量化前后对比

<div class="echarts" data-src="assets/figures/benchmarks/quantization_comparison.echarts.json"></div>