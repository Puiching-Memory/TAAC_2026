---
icon: lucide/database-zap
---

# Cache Policies：收益与历史测量

缓存保存增强前的基础 batch，目标是减少重复读取和转换。命中率只是中间指标，实际收益还受 clone、共享内存同步、worker 调度和 trace 覆盖影响。实现语义见[数据管道](../guide/pcvr-data-pipeline.md)。

## 历史记录：2026-08-02

以下保留原文 30x 合成数据复测：30000 行、30 row groups、约 1.2 GB，batch=256、cache=64 batches、max_batches=200、warmup=5、buffer=1、step_random、torch_threads=4，计时前已加载 native extension。硬件和完整 commit 未随该记录固定，本次未复测。

| workers | mode | rows/s | batches/s | vs none | hit rate |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | none | 801 | 3.2 | 1.00x | 0.0% |
| 0 | lru | 1304 | 5.2 | 1.63x | 46.8% |
| 0 | opt | 1488 | 6.0 | 1.86x | 54.1% |
| 4 | none | 1642 | 6.6 | 1.00x | 0.0% |
| 4 | lru | 2916 | 11.7 | 1.78x | 45.9% |
| 4 | opt | 2889 | 11.6 | 1.76x | 52.7% |

OPT 在这次多 worker 记录中命中率更高，但吞吐略低于 LRU。不能由单 worker 的排序推出所有 worker 数、数据规模和 cache 容量的排序。

同一历史负载的三轮 sweep trace 实验如下。trace 窗口 2×sweep=232 步、容量64；刷新保留 cache 内容，第三轮才显示旧窗口失效的差异。

| sweep | 每轮刷新命中率 | 不刷新命中率 |
| ---: | ---: | ---: |
| 0 | 33.6% | 33.6% |
| 1 | 75.9% | 75.9% |
| 2 | 82.8% | 62.1% |

更早 2026-05-07 的绝对吞吐属于不同转换实现。2026-08-02 的向量化及 trace 生命周期变更后，不再将旧数值当作当前基准；需要追溯完整旧表可查看仓库历史。

## 重测矩阵

按[性能指南](../guide/performance-benchmarks.md)生成 300x 数据后，在相同环境运行：

```bash
mkdir -p outputs/benchmarks/cache
for workers in 0 4; do
  for mode in none lru fifo lfu rr opt; do
    uv run taac-benchmark-pcvr-data-pipeline \
      --dataset-path outputs/perf/pcvr_synthetic_300x/demo_300000.parquet \
      --schema-path outputs/perf/pcvr_synthetic_300x/schema.json \
      --preset cache --cache-mode "$mode" --cache-batches 512 \
      --max-batches 1500 --warmup-batches 0 --num-workers "$workers" \
      --buffer-batches 1 --torch-threads 4 --json \
      > "outputs/benchmarks/cache/${mode}_workers${workers}.json"
  done
done
```

这是冷/热混合的有限访问窗口。另测 warmup 后的稳态，多次运行时交错策略顺序，避免 OS page cache 和后台负载偏差；记录内存占用与 native 编译成本。

## 必须检查的字段

查看 `cache_mode`、`cache_impl`、`shared_cache`、`rows_per_sec`、`measured_batches`、`measured_passes`，以及 `data_cache_stats` 中的 hits、misses、hit_rate、trace_length、opt_active 和 native 状态。OPT 没优势时先确认 trace 与真实访问顺序一致。

有放回采样会重复访问，缓存命中率不能解释为“每个 epoch 覆盖率”。无 trace 时 OPT 的 LRU 行为也要明确记录。
