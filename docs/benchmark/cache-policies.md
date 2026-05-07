---
icon: lucide/database-zap
---

# Cache Policies Benchmark

## 支持记录

- CLI 入口：`taac-benchmark-pcvr-data-pipeline`。
- 主要源码：`src/taac2026/application/benchmarking/pcvr_data_pipeline_benchmark.py`。
- cache 实现：`src/taac2026/infrastructure/data/cache.py`。
- 非 OPT 策略使用 `cachetools`：`lru`、`fifo`、`lfu`、`rr`。
- OPT 仍由项目内实现，因为它依赖已知访问 trace 和多 worker shared-memory 索引。
- `memory` 不是合法模式；普通最近最少使用缓存现在写作 `lru`。

## 算法差异

| mode   | 驱逐依据       | 典型优势                             | 典型风险                                       |
| ------ | -------------- | ------------------------------------ | ---------------------------------------------- |
| `lru`  | 最近访问时间   | 适合局部性强、热点随时间漂移的数据   | 扫描式访问可能把未来还会用的热点挤掉           |
| `fifo` | 写入顺序       | 维护开销低，行为稳定，适合简单基线   | 不看访问频次，老热点会被直接淘汰               |
| `lfu`  | 历史访问频次   | 适合长期热点明显的数据               | 热点变化后旧高频 key 可能停留太久              |
| `rr`   | 随机选择       | 策略开销小，可作为“无结构假设”对照组 | 方差较大，单次结果不稳定                       |
| `opt`  | 下一次使用距离 | trace 准确时接近理论最优淘汰         | 需要访问顺序可预测；trace 不匹配会退回安全行为 |

这些策略的端到端差异通常不只来自命中率。PCVR 数据管道 cache 存的是增强前的基础 batch，命中后会 clone batch，后续 transform 仍然会执行。因此读数要同时考虑 parquet 读取成本、batch 构建成本、clone 成本、cache 维护成本和 transform 成本。

## 推荐命令

先准备一个足够大的合成数据集。不要用 1000 行 demo 数据推断 cache 策略差异：

```bash
uv run taac-generate-pcvr-synthetic-dataset \
  --source-dir data/sample_1000_raw \
  --output-dir outputs/perf/pcvr_synthetic_300x \
  --multiplier 300 \
  --force
```

只比较基础 cache 策略时，使用 `cache` preset 并覆盖 `--cache-mode`：

```bash
mkdir -p outputs/benchmarks

for mode in none lru fifo lfu rr opt; do
  uv run taac-benchmark-pcvr-data-pipeline \
    --dataset-path outputs/perf/pcvr_synthetic_300x/demo_300000.parquet \
    --schema-path outputs/perf/pcvr_synthetic_300x/schema.json \
    --preset cache \
    --cache-mode "$mode" \
    --cache-batches 512 \
    --passes 3 \
    > "outputs/benchmarks/data_pipeline_cache_${mode}.json"
done
```

如果要看 cache 与随机增强叠加后的效果，把 preset 改成 `augment`，仍然用同一组 `--cache-mode`：

```bash
for mode in none lru fifo lfu rr opt; do
  uv run taac-benchmark-pcvr-data-pipeline \
    --dataset-path outputs/perf/pcvr_synthetic_300x/demo_300000.parquet \
    --schema-path outputs/perf/pcvr_synthetic_300x/schema.json \
    --preset augment \
    --cache-mode "$mode" \
    --cache-batches 512 \
    --views-per-row 2 \
    --seq-window-min-len 8 \
    --passes 3 \
    > "outputs/benchmarks/data_pipeline_augment_cache_${mode}.json"
done
```

多 worker 下的 OPT 会走 shared-memory cache。比较这一条路径时要显式记录 worker 数：

```bash
uv run taac-benchmark-pcvr-data-pipeline \
  --dataset-path outputs/perf/pcvr_synthetic_300x/demo_300000.parquet \
  --schema-path outputs/perf/pcvr_synthetic_300x/schema.json \
  --preset cache \
  --cache-mode opt \
  --cache-batches 512 \
  --num-workers 4 \
  --passes 3 \
  > outputs/benchmarks/data_pipeline_cache_opt_workers4.json
```

## 读数要点

优先比较这些 JSON 字段：

| 字段               | 含义                                       |
| ------------------ | ------------------------------------------ |
| `cache_mode`       | 实际启用的 cache mode                      |
| `cache_impl`       | per-process cache 或 shared-memory cache   |
| `shared_cache`     | 是否使用全局访问 trace / shared cache 路径 |
| `rows_per_sec`     | 端到端行吞吐                               |
| `batches_per_sec`  | 端到端 batch 吞吐                          |
| `warmup_rows`      | warmup 消耗的数据量                        |
| `measured_passes`  | 实际完成的 pass 数                         |
| `measured_batches` | 计入测速的 batch 数                        |
| `batches_per_pass` | 每个 pass 的估算 batch 数                  |

cache 策略对比必须至少跑多 pass。第一轮大多是填充缓存；第二轮以后才更接近重复访问的收益。`measured_passes` 如果只有 1，通常只能说明冷启动成本，不能说明策略优劣。

## 结果解读

- 如果 `none` 已经很快，说明瓶颈可能不在数据转换，cache 算法差异会被噪声淹没。
- 如果 `lru` 明显优于 `fifo`，通常说明近期局部性强，最近访问比写入顺序更能预测未来访问。
- 如果 `lfu` 优于 `lru`，通常说明存在稳定长周期热点，而不是短期滑动窗口热点。
- 如果 `rr` 接近 `lru/fifo/lfu`，说明当前访问模式下驱逐策略不敏感，或者 cache 容量已经足够大。
- 如果 `opt` 没有优势，先检查 `shared_cache`、`num_workers`、shuffle 和 trace 是否与预期一致。
- 如果打开增强后所有 cache 模式收益下降，可能是 transform 成本主导，cache 只省掉了基础 batch 构建成本。

## 最近验收观察

最近一次本地验收时间：2026-05-06 UTC。原始 JSON 输出保存在 `outputs/benchmarks/cache_policies/`，该目录不提交到仓库。

环境记录：

| 项目         | 值                                                                 |
| ------------ | ------------------------------------------------------------------ |
| commit       | `947db86`                                                          |
| 主机         | `HGX-076`                                                          |
| CPU          | Intel Xeon Platinum 8468，2 sockets，192 logical CPUs              |
| GPU          | NVIDIA H800                                                        |
| Python       | 3.10.20                                                            |
| PyTorch      | 2.7.1+cu126                                                        |
| CUDA         | 12.6                                                               |
| cachetools   | 7.1.1                                                              |
| 数据集       | `outputs/perf/pcvr_synthetic_300x/demo_300000.parquet`，约 12.0 GB |
| schema       | `outputs/perf/pcvr_synthetic_300x/schema.json`                     |
| 通用参数     | `--preset cache --passes 3 --warmup-batches 5 --buffer-batches 1 --torch-threads 4 --no-shuffle` |

### 小工作集：cache 能覆盖工作集

这组用 `--train-ratio 0.1 --cache-batches 128 --num-workers 0`。训练集为 27 Row Groups / 27,000 rows，每个 pass 约 106 batches，cache 容量可以覆盖一个 pass 的基础 batch。这个口径主要观察“重复访问时，热 cache 能省多少基础 batch 构建成本”。

```bash
for mode in none lru fifo lfu rr opt; do
  uv run taac-benchmark-pcvr-data-pipeline \
    --dataset-path outputs/perf/pcvr_synthetic_300x/demo_300000.parquet \
    --schema-path outputs/perf/pcvr_synthetic_300x/schema.json \
    --preset cache \
    --cache-mode "$mode" \
    --cache-batches 128 \
    --passes 3 \
    --warmup-batches 5 \
    --num-workers 0 \
    --buffer-batches 1 \
    --torch-threads 4 \
    --no-shuffle \
    --train-ratio 0.1 \
    > "outputs/benchmarks/cache_policies/cache_${mode}_workers0_train0p1_fit.json"
done
```

| mode   | rows/s | batches/s | elapsed sec | vs `none` | cache impl             |
| ------ | -----: | --------: | ----------: | --------: | ---------------------- |
| `none` |   4308 |     17.22 |        18.5 |     1.00x | PCVRMemoryBatchCache   |
| `lru`  |   9050 |     36.18 |         8.8 |     2.10x | PCVRMemoryBatchCache   |
| `fifo` |   8739 |     34.94 |         9.1 |     2.03x | PCVRMemoryBatchCache   |
| `lfu`  |   8617 |     34.45 |         9.2 |     2.00x | PCVRMemoryBatchCache   |
| `rr`   |   8895 |     35.56 |         8.9 |     2.06x | PCVRMemoryBatchCache   |
| `opt`  |   9025 |     36.08 |         8.8 |     2.10x | PCVRMemoryBatchCache   |

观察：当 cache 能覆盖工作集时，所有缓存策略都接近 2x `none`。这说明本数据管道里可缓存的基础 batch 构建成本确实明显；在这个口径下，策略之间的差异小于“是否命中热 cache”的差异。

### 全量工作集：cache 容量不足

这组用 `--cache-batches 512 --num-workers 0` 跑完整训练切分。训练集为 270 Row Groups / 270,000 rows，每个 pass 约 1055 batches，cache 无法覆盖一个完整 pass。这个口径主要观察“cache 容量不足时，驱逐策略和维护成本是否划算”。

| mode   | rows/s | batches/s | elapsed sec | vs `none` | cache impl             |
| ------ | -----: | --------: | ----------: | --------: | ---------------------- |
| `none` |   4386 |     17.54 |       180.4 |     1.00x | PCVRMemoryBatchCache   |
| `lru`  |   4147 |     16.59 |       190.8 |     0.95x | PCVRMemoryBatchCache   |
| `fifo` |   3863 |     15.45 |       204.8 |     0.88x | PCVRMemoryBatchCache   |
| `lfu`  |   5411 |     21.64 |       146.2 |     1.23x | PCVRMemoryBatchCache   |
| `rr`   |   4350 |     17.40 |       181.9 |     0.99x | PCVRMemoryBatchCache   |
| `opt`  |   3877 |     15.51 |       204.1 |     0.88x | PCVRMemoryBatchCache   |

观察：全量顺序压力下，`lru/fifo/opt` 没有赢过 `none`，说明 cache 维护、clone 和驱逐成本可能抵消收益。`lfu` 在这次顺序运行中最高，但这组是按 `none -> lru -> fifo -> lfu -> rr -> opt` 顺序执行，OS page cache 会逐步变热，所以不要把这张表解释成稳定排名。正式比较需要交错顺序并重复多轮。

### 多 worker：OPT shared cache 路径

这组只比较 `none` 和 `opt`，参数为 `--cache-batches 512 --num-workers 4`。`opt` 在多 worker 训练路径会启用 `PCVRSharedBatchCache`。

| mode   | cache impl           | shared cache | rows/s | batches/s | elapsed sec | vs `none` |
| ------ | -------------------- | ------------ | -----: | --------: | ----------: | --------: |
| `none` | PCVRMemoryBatchCache | false        |  14436 |     57.74 |        54.8 |     1.00x |
| `opt`  | PCVRSharedBatchCache | true         |  12241 |     48.97 |        64.6 |     0.85x |

观察：这个本地口径下，shared OPT 没有提升吞吐，反而慢于 `none`。这不代表 OPT 永远不可用，只说明在 270k rows、4 workers、512 batches cache、无 shuffle 的 synthetic x300 数据上，shared-memory cache 的同步/拷贝开销大于它省下的数据转换成本。Baseline+ 默认打开 OPT cache 时，应继续用真实训练口径验证端到端收益。

## 当前结论

- cache 能覆盖重复工作集时，`lru/fifo/lfu/rr/opt` 都能显著提升吞吐，本次约 2.0x - 2.1x。
- cache 容量不足时，非命中成本很真实：部分策略比 `none` 慢。
- 这次数据不支持“OPT 一定更快”的结论；尤其多 worker shared OPT 需要结合真实 worker 数、shuffle、cache 容量和 transform 成本继续验证。
- 以后记录性能结论时，需要至少保留小工作集和全量工作集两个口径，并交错模式顺序重复运行，避免 OS page cache 暖身造成假排名。
