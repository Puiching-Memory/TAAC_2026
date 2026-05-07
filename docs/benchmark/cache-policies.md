---
icon: lucide/database-zap
---

# Cache Policies Benchmark

## 当前结论

这页记录当前代码下 PCVR 数据管线 cache 策略的完整测评。比较对象包括 `none`、`lru`、`fifo`、`lfu`、`rr`、`opt`。非 OPT 策略使用 `cachetools`，项目内只保留 OPT，因为它依赖已知访问 trace 和多 worker shared-memory 索引。

2026-05-07 这轮单 worker、无 shuffle、synthetic x300 数据集测评显示：

- 小工作集可以被 cache 覆盖时，所有 cache 策略都明显快于 `none`，吞吐约为 `none` 的 `1.94x` 到 `2.07x`。
- 全量工作集无法被 cache 覆盖时，策略差异变大：`opt` 和 `lfu` 分别约为 `none` 的 `1.25x` 和 `1.24x`；`rr` 接近 `none`；`lru` 和 `fifo` 低于 `none`。
- 这轮数据支持在“顺序访问、trace 准确、单 worker、cache 容量不足”的口径下使用 `opt`。多 worker shared OPT 仍需要单独测，因为同步和张量拷贝成本不同。

## 支持范围

| 项目             | 当前状态                                                                |
| ---------------- | ----------------------------------------------------------------------- |
| CLI 入口         | `taac-benchmark-pcvr-data-pipeline`                                     |
| benchmark 源码   | `src/taac2026/application/benchmarking/pcvr_data_pipeline_benchmark.py` |
| cache 源码       | `src/taac2026/infrastructure/data/cache.py`                             |
| native OPT index | `src/taac2026/infrastructure/data/native/opt_cache.cpp`                 |
| 非 OPT 实现      | `cachetools` 的 `LRUCache`、`FIFOCache`、`LFUCache`、`RRCache`          |
| OPT 实现         | 项目内实现，依赖访问 trace                                              |
| legacy 名称      | `memory` 不是合法模式；普通内存 LRU 写作 `lru`                          |

## 策略差异

| mode   | 实现                   | 驱逐依据       | 适合场景                   | 主要风险                                      |
| ------ | ---------------------- | -------------- | -------------------------- | --------------------------------------------- |
| `none` | 无 cache               | 不缓存         | I/O 和 batch 构建基线      | 不复用已转换 batch                            |
| `lru`  | `cachetools.LRUCache`  | 最近访问时间   | 局部性强、热点随时间漂移   | 顺序扫描且容量不足时可能无收益                |
| `fifo` | `cachetools.FIFOCache` | 写入顺序       | 低维护开销基线             | 不看未来访问和访问频次                        |
| `lfu`  | `cachetools.LFUCache`  | 历史访问频次   | 重复 pass 中热点稳定       | 热点漂移后旧高频 key 可能滞留                 |
| `rr`   | `cachetools.RRCache`   | 随机选择       | 无结构假设对照组           | 单次方差较大                                  |
| `opt`  | 项目内 OPT             | 下一次使用距离 | trace 准确、访问顺序可预测 | trace 错位会 fallback；shared path 有同步成本 |

PCVR cache 存的是增强前的基础 batch。cache 命中后仍会 clone batch，后续 transform 也仍会执行。因此端到端吞吐同时受 parquet 读取、batch 转换、clone、cache 维护和 transform 成本影响，不要只用命中率解释结果。

## 测评环境

| 项目       | 值                                                                 |
| ---------- | ------------------------------------------------------------------ |
| 时间       | 2026-05-07 UTC                                                     |
| commit     | `28a346a`                                                          |
| 主机       | `HGX-076`                                                          |
| CPU        | Intel Xeon Platinum 8468，2 sockets，192 logical CPUs              |
| GPU        | NVIDIA H800                                                        |
| Python     | 3.10.20                                                            |
| PyTorch    | 2.7.1+cu126                                                        |
| CUDA       | 12.6                                                               |
| cachetools | 7.1.1                                                              |
| 数据集     | `outputs/perf/pcvr_synthetic_300x/demo_300000.parquet`，约 12.0 GB |
| schema     | `outputs/perf/pcvr_synthetic_300x/schema.json`                     |
| 原始输出   | `outputs/benchmarks/cache_policies_current/`                       |

`outputs/benchmarks/` 是本地 benchmark 输出目录，不提交到仓库。正式报告需要重新记录 commit、硬件、依赖版本、完整命令和 JSON 输出。

## 完整测评结果

### 小工作集：cache 能覆盖一个 pass

口径：`--train-ratio 0.1 --cache-batches 128 --num-workers 0 --passes 3 --warmup-batches 5 --buffer-batches 1 --torch-threads 4 --no-shuffle`。

训练集为 27 Row Groups / 27,000 rows，每个 pass 约 106 batches，cache 容量 128 batches 可以覆盖一个 pass 的基础 batch。

| mode   | rows/s | batches/s | elapsed sec | vs `none` | measured batches |
| ------ | -----: | --------: | ----------: | --------: | ---------------: |
| `none` |   4396 |     17.58 |       18.09 |     1.00x |              318 |
| `lru`  |   9104 |     36.40 |        8.74 |     2.07x |              318 |
| `fifo` |   9032 |     36.11 |        8.81 |     2.05x |              318 |
| `lfu`  |   8965 |     35.84 |        8.87 |     2.04x |              318 |
| `rr`   |   8950 |     35.78 |        8.89 |     2.04x |              318 |
| `opt`  |   8545 |     34.16 |        9.31 |     1.94x |              318 |

观察：小工作集下 cache 本身的收益很明显，各策略之间差异小于是否开启 cache 的差异。`opt` 没有领先，因为工作集已被 cache 覆盖，理论最优驱逐的优势很小，额外 trace 维护仍有成本。

### 全量工作集：cache 容量不足

口径：`--cache-batches 512 --num-workers 0 --passes 3 --warmup-batches 5 --buffer-batches 1 --torch-threads 4 --no-shuffle`。

训练集为 270 Row Groups / 270,000 rows，每个 pass 约 1055 batches，cache 容量 512 batches 无法覆盖完整 pass。

| mode   | rows/s | batches/s | elapsed sec | vs `none` | measured batches |
| ------ | -----: | --------: | ----------: | --------: | ---------------: |
| `none` |   4476 |     17.90 |      176.79 |     1.00x |             3165 |
| `lru`  |   3906 |     15.62 |      202.60 |     0.87x |             3165 |
| `fifo` |   3891 |     15.56 |      203.37 |     0.87x |             3165 |
| `lfu`  |   5550 |     22.20 |      142.59 |     1.24x |             3165 |
| `rr`   |   4322 |     17.29 |      183.09 |     0.97x |             3165 |
| `opt`  |   5591 |     22.36 |      141.54 |     1.25x |             3165 |

观察：全量顺序访问且 cache 容量不足时，`opt` 最高，`lfu` 接近 `opt`。`lru` 和 `fifo` 低于 `none`，说明 miss、clone 和维护成本会抵消转换复用收益。`rr` 接近 `none`，适合作为无结构假设对照。

## 复现命令

先生成足够大的合成数据集。不要用 1000 行 demo 数据推断 cache 策略差异。

```bash
uv run taac-generate-pcvr-synthetic-dataset \
  --source-dir data/sample_1000_raw \
  --output-dir outputs/perf/pcvr_synthetic_300x \
  --multiplier 300 \
  --force
```

完整矩阵命令：

```bash
mkdir -p outputs/benchmarks/cache_policies_current
modes=(none lru fifo lfu rr opt)

for mode in "${modes[@]}"; do
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
    > "outputs/benchmarks/cache_policies_current/cache_${mode}_fit.json"
done

for mode in "${modes[@]}"; do
  uv run taac-benchmark-pcvr-data-pipeline \
    --dataset-path outputs/perf/pcvr_synthetic_300x/demo_300000.parquet \
    --schema-path outputs/perf/pcvr_synthetic_300x/schema.json \
    --preset cache \
    --cache-mode "$mode" \
    --cache-batches 512 \
    --passes 3 \
    --warmup-batches 5 \
    --num-workers 0 \
    --buffer-batches 1 \
    --torch-threads 4 \
    --no-shuffle \
    > "outputs/benchmarks/cache_policies_current/cache_${mode}_full.json"
done
```

多 worker 下 OPT 会走 shared-memory cache，需要单独测：

```bash
uv run taac-benchmark-pcvr-data-pipeline \
  --dataset-path outputs/perf/pcvr_synthetic_300x/demo_300000.parquet \
  --schema-path outputs/perf/pcvr_synthetic_300x/schema.json \
  --preset cache \
  --cache-mode opt \
  --cache-batches 512 \
  --num-workers 4 \
  --passes 3 \
  --warmup-batches 5 \
  --buffer-batches 1 \
  --torch-threads 4 \
  --no-shuffle \
  > outputs/benchmarks/cache_policies_current/cache_opt_workers4.json
```

## JSON 字段

| 字段                             | 含义                                             |
| -------------------------------- | ------------------------------------------------ |
| `cache_mode`                     | 实际启用的 cache mode                            |
| `cache_impl`                     | `PCVRMemoryBatchCache` 或 `PCVRSharedBatchCache` |
| `shared_cache`                   | 是否使用全局访问 trace / shared cache 路径       |
| `rows_per_sec`                   | 端到端行吞吐                                     |
| `batches_per_sec`                | 端到端 batch 吞吐                                |
| `warmup_batches` / `warmup_rows` | warmup 消耗的数据量                              |
| `measured_passes`                | 实际完成的 pass 数                               |
| `measured_batches`               | 计入测速的 batch 数                              |
| `batches_per_pass`               | 每个 pass 的估算 batch 数                        |

## 解读规则

- `none` 很快时，cache 算法差异容易被 parquet 读取和 batch 转换噪声淹没。
- 小工作集更适合证明 cache 是否有价值；全量工作集更适合比较驱逐和维护成本。
- `opt` 需要 trace 和实际访问顺序一致。发现 OPT 没有优势时，先看 `warmup_batches`、pass 边界、shuffle、`num_workers` 和 `shared_cache`。
- 打开随机增强后，cache 只能省掉基础 batch 构建成本，不能省掉 transform 成本。
- 多 worker shared OPT 的同步和张量复制成本可能大于节省的转换成本，需要用真实训练 worker 数验证。
- 当前表格是单次顺序运行。正式排名建议交错 mode 顺序并重复运行，避免 OS page cache 和后台负载影响结论。
