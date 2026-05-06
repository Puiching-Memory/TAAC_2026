---
icon: lucide/gauge
---

# 性能 Benchmark

Benchmark 文档应该回答“怎么重新测”，而不是保存一组很快会过时的数字。本页只保留当前仓库可运行的 benchmark 入口、推荐口径和结果解读方法。

## 什么时候跑

- 改了数据读取、cache、batch transform 或 schema 解析。
- 改了 dense optimizer、scheduler 或 AMP / compile 默认值。
- 改了 TileLang / torch backend 的 attention 或 RMSNorm。
- 想比较两个实验包的吞吐差异，但不想把模型指标和系统吞吐混在一起。

正式结论必须记录代码版本、硬件、CUDA / PyTorch 版本、命令和 JSON 输出。不要复用旧机器上的绝对数值。

## 准备数据

不要直接用 1000 行 demo 数据做吞吐结论。先把 demo 数据放大到一个稳定口径，例如 300x：

```bash
uv run taac-generate-pcvr-synthetic-dataset \
  --source-dir data/sample_1000_raw \
  --output-dir outputs/perf/pcvr_synthetic_300x \
  --multiplier 300 \
  --force
```

这个命令会生成 parquet 和 `schema.json`，默认对 user / item / 时间字段做轻量 offset，避免完全重复的 id 干扰部分缓存观察。

## 数据管道 Benchmark

数据管道 benchmark 不跑模型，只测 parquet 读取、batch 构建、cache 和增强带来的吞吐变化。

```bash
mkdir -p outputs/benchmarks

uv run taac-benchmark-pcvr-data-pipeline \
  --dataset-path outputs/perf/pcvr_synthetic_300x/demo_300000.parquet \
  --schema-path outputs/perf/pcvr_synthetic_300x/schema.json \
  --preset none \
  --passes 3 \
  > outputs/benchmarks/data_pipeline_none.json
```

常用 preset：

| preset        | 含义                                      |
| ------------- | ----------------------------------------- |
| `none`        | 纯数据加载，不启用 cache 或增强           |
| `cache`       | 启用 memory cache                         |
| `opt`         | 启用 OPT batch cache                      |
| `augment`     | 启用序列裁剪、特征 mask 和 domain dropout |
| `opt-augment` | OPT cache + 数据增强                      |

建议至少跑：

```bash
for preset in none opt augment opt-augment; do
  uv run taac-benchmark-pcvr-data-pipeline \
    --dataset-path outputs/perf/pcvr_synthetic_300x/demo_300000.parquet \
    --schema-path outputs/perf/pcvr_synthetic_300x/schema.json \
    --preset "$preset" \
    --passes 3 \
    > "outputs/benchmarks/data_pipeline_${preset}.json"
done
```

重点看 JSON 里的 `rows_per_sec`、`batches_per_sec`、`measured_rows`、`measured_batches` 和 `cache_impl`。如果 `measured_rows` 太小，结论通常不稳定。

## Optimizer Benchmark

Dense optimizer benchmark 用一个合成 MLP 负载比较优化器 step 成本，适合看 `adamw`、`fused_adamw`、`orthogonal_adamw`、`muon` 这类选择的系统开销。

```bash
uv run taac-benchmark-pcvr-optimizer \
  --device cuda \
  --batch-size 512 \
  --feature-dim 128 \
  --hidden-dim 512 \
  --depth 4 \
  --steps 50 \
  --warmup-steps 10 \
  --repeats 5 \
  --optimizers adamw,orthogonal_adamw,muon \
  > outputs/benchmarks/optimizer_cuda.json
```

如果只是本地 CPU 可用性检查，可以把 `--device cuda` 改成 `--device cpu`，但不要拿 CPU 数字推断线上 GPU 训练性能。

## TileLang 算子 Benchmark

TileLang benchmark 用来比较 torch backend 和 tilelang backend 的算子耗时，并同时做误差检查。

RMSNorm：

```bash
uv run taac-benchmark-pcvr-tilelang-ops \
  --operator rms_norm \
  --device cuda \
  --rows 8192 \
  --cols 128 \
  --dtype float16 \
  --backends torch,tilelang \
  > outputs/benchmarks/tilelang_rms_norm.json
```

Flash Attention：

```bash
uv run taac-benchmark-pcvr-tilelang-ops \
  --operator flash_attention \
  --device cuda \
  --batch 8 \
  --heads 8 \
  --query-len 128 \
  --kv-len 128 \
  --head-dim 64 \
  --dtype float16 \
  --backends torch,tilelang \
  > outputs/benchmarks/tilelang_flash_attention.json
```

如果在 CPU 上运行，TileLang backend 可能报告 unsupported，这是正常的环境信号，不等于 torch fallback 有问题。

## 解读原则

- 先比较同一机器、同一代码版本、同一命令参数下的相对变化。
- `rows_per_sec` 适合看数据管道；step time 更适合看 optimizer 和算子。
- cache benchmark 至少跑多 pass，否则 warmup 和首次访问会掩盖真实收益。
- TileLang 结果要同时看速度和误差状态；速度快但 accuracy failure 不能算可用。
- benchmark 输出放在 `outputs/benchmarks/`，不要提交生成结果，除非要做一次明确的报告快照。

## 源码入口

- 数据生成：`src/taac2026/application/benchmarking/generate_pcvr_synthetic_dataset.py`
- 数据管道 benchmark：`src/taac2026/application/benchmarking/pcvr_data_pipeline_benchmark.py`
- Optimizer benchmark：`src/taac2026/application/benchmarking/pcvr_optimizer_benchmark.py`
- TileLang benchmark：`src/taac2026/application/benchmarking/pcvr_tilelang_ops_benchmark.py`
