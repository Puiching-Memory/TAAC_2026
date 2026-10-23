---
icon: lucide/database-zap
---

# PCVR 数据管道：切分、缓存和增强

共享数据管道把 parquet 转换为 `PCVRBatch`，管理采样、切分、基础 batch cache 和训练增强。模型负责字段选择与解释，数据层不应加入实验特有语义。

## 输入和输出

本地可显式传 parquet 文件或目录与 schema；不传数据路径时，普通 PCVR 实验使用公开 demo。线上通过平台变量注入真实路径。schema 来源规则见[架构](../architecture.md)，不要把 observed schema 统计报告当作模型声明。

```text
parquet → batch key / row filter → 基础 batch 转换与 cache
        → clone → 训练 transforms → DataLoader → 模型输入
```

训练主要使用 step-indexed map-style dataset。PyTorch DataLoader 管理 worker、prefetch、pin memory；验证与推理顺序扫描，不执行随机训练增强。

## 选择验证切分

| `--data.split_strategy` | 切分单位 | 适用问题 |
| --- | --- | --- |
| `row_group_tail` | 尾部 row group | 本地流程检查、低启动开销 |
| `timestamp_auto` | timestamp 行过滤 | 用较新时间段检验时间泛化 |
| `user_hash` | 原始 user id 的稳定 hash | 检验未见用户 |
| `sample_hash` | 文件名、row group、行位置 hash | 缺少可靠用户或时间列时的稳定划分 |

行级过滤的三种策略会将不兼容的 `step_random` 切为 `row_group_sweep`，避免随机取到过滤后空 batch。实际选择需看日志和训练摘要，不能仅凭配置字符串判断。

hash seed 来自 `optimizer.seed`。先按 `valid_ratio` 选验证桶，再从剩余桶按 `train_ratio` 取训练子集。`train_ratio` 范围 `(0,1]`，`valid_ratio` 范围 `(0,1)`；前者小于 1 时会有未参与任一侧的行。user id 为空时回退 sample key；sample key 忽略父目录，但文件重命名会影响划分。

训练启动的 hash 行数是估算值。结束后的 `train_split_observed_schema.json`、`valid_split_observed_schema.json` 使用相同过滤规则统计精确行数，并记录 filter。检查 `data_split.is_disjoint` 和 `is_l1_ready`；小数据可能得到空侧，不能静默复用另一侧样本。

## 采样与训练预算

`--data.sampling_strategy step_random` 进行有放回采样；`row_group_sweep` 按 sweep 访问。`--data.train_steps_per_sweep` 控制逻辑 sweep 长度，0 时从数据推导。总 optimizer step 仍由 `--optimizer.max_steps` 截断，逻辑 sweep 不应自动解释为每行恰好训练一次。

## Cache 的含义

| mode | 淘汰策略 |
| --- | --- |
| `none` | 不缓存 |
| `lru` | 最近最少使用 |
| `fifo` | 最早进入 |
| `lfu` | 低访问频次 |
| `rr` | 随机 |
| `opt` | 根据已知 trace 的下一次使用距离 |

配置位于 `data_pipeline.cache`，`max_batches` 是 batch 容量，不是字节数。缓存保存增强前的基础 batch，取出时 clone，避免随机增强污染缓存。policy index 使用 native C++ 实现；首次使用需可用构建环境。

单 worker 使用内存 cache，多 worker 启用 shared cache。OPT 依赖未来访问窗口，没有 trace 时使用 LRU 行为；训练器在 sweep 边界刷新 trace，保留相同 key universe 的缓存内容。高命中率不保证更高吞吐，还需计入同步、clone 和内存开销。

## 增强配置

| 类型 | 行为 |
| --- | --- |
| `PCVRSequenceCropConfig` | tail、random_tail、rolling，支持多视图 |
| `PCVRFeatureMaskConfig` | 对 sparse 与序列事件随机置零，更新序列状态 |
| `PCVRNonSequentialSparseDropoutConfig` | 对非序列 sparse 做 dropout |
| `PCVRDomainDropoutConfig` | 按行丢弃完整序列域及长度/时间桶 |

配置在实验 `TRAIN_DEFAULTS.data_pipeline` 声明，随机源由 pipeline seed 与读取上下文管理。多视图产生更多行后，step dataset 会重新取到 optimizer batch 大小；`views_per_row=2` 不等于每步计算量必然翻倍。

```python
from taac2026.api import (
    PCVRDataCacheConfig, PCVRDataPipelineConfig,
    PCVRFeatureMaskConfig, PCVRDomainDropoutConfig,
)

pipeline = PCVRDataPipelineConfig(
    cache=PCVRDataCacheConfig(mode="lru", max_batches=64),
    transforms=(
        PCVRFeatureMaskConfig(probability=0.03),
        PCVRDomainDropoutConfig(probability=0.02),
    ),
    seed=42,
    strict_time_filter=True,
)
```

`pipeline.enabled` 只表示有启用的 transform；cache 是否开启看 `cache.enabled`。时间过滤、增强和 cache 是独立变量，做消融时分别记录。

## 验证和性能

```bash
uv run --with torch==2.13.0 pytest tests/unit/infrastructure/data -q
uv run --with torch==2.13.0 pytest tests/contract/experiments/test_runtime_contract_matrix.py -q
```

数据管道 benchmark 的 `augment` preset 默认也启用 LRU cache；若想只测增强，用 `--cache-mode none` 显式覆盖。完整命令见[性能测量](performance-benchmarks.md)，历史对照见[缓存测量](../benchmark/cache-policies.md)。

源码重点：`domain/config.py`、`infrastructure/data/step_dataset.py`、`batch_converter.py`、`cache.py`、`transforms.py`、`hash_split.py`、`observation.py`。
