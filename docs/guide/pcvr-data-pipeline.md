---
icon: lucide/database-zap
---

# PCVR 数据管道

PCVR 共享 runtime 的数据路径分为几层：

| 层                | 位置                                         | 职责                                                              |
| ----------------- | -------------------------------------------- | ----------------------------------------------------------------- |
| Row Group 规划    | `taac2026.infrastructure.pcvr.data`          | 收集 parquet Row Group，并生成 train/valid 切分计划               |
| 基础读取转换      | `PCVRParquetDataset`                         | 解析 `schema.json`，把 Arrow batch 转为预分批 tensor dict         |
| 组合式 batch 组件 | `taac2026.infrastructure.pcvr.data_pipeline` | 内存 cache、batch transform、row-level shuffle buffer、训练期增广 |
| 训练装配          | `get_pcvr_data`                              | 创建 train/valid dataset 和 DataLoader                            |

评估和推理路径仍然直接创建无增广的 `PCVRParquetDataset`，保持确定性。

## 实验侧组合

推荐在 experiment package 的 `train_defaults` 中像模型配置一样组装数据管道：

```python
from taac2026.infrastructure.pcvr.config import (
    PCVRDataCacheConfig,
    PCVRDataPipelineConfig,
    PCVRDomainDropoutConfig,
    PCVRFeatureMaskConfig,
    PCVRSequenceCropConfig,
    PCVRTrainConfig,
)


train_defaults = PCVRTrainConfig(
    data_pipeline=PCVRDataPipelineConfig(
        cache=PCVRDataCacheConfig(mode="memory", max_batches=256),
        seed=42,
        strict_time_filter=True,
        transforms=(
            PCVRSequenceCropConfig(
                views_per_row=2,
                seq_window_mode="random_tail",
                seq_window_min_len=8,
            ),
            PCVRFeatureMaskConfig(probability=0.05),
            PCVRDomainDropoutConfig(probability=0.05),
        ),
    ),
)
```

`transforms` 元组顺序就是实际执行顺序；可以只放一个 transform，也可以调换顺序或关闭某个组件。默认 `PCVRDataPipelineConfig()` 没有 transform、没有 cache，训练行为保持旧逻辑。

增广 transform 只装配到 train dataset。valid dataset 会保留 cache 配置但不接收 transform，离线 evaluation 和线上 inference 也不接收训练期增广。

训练 CLI 不提供数据管道覆盖参数。数据管道属于 experiment package 的 typed contract，和模型结构默认值一起由 `PCVRTrainConfig` 管理；需要比较不同数据管道时，应创建或调整对应 experiment package 的 `train_defaults`。

## 组件说明

### `PCVRDataPipelineConfig`

这是数据管道的总装配点，负责把 cache、随机种子、时间安全过滤和 transform 序列绑定到一个 experiment 的 `train_defaults`。它本身不做数据修改，只描述 runtime 应该如何组合组件。

常用字段：

| 字段                 | 作用                                       | 建议                                   |
| -------------------- | ------------------------------------------ | -------------------------------------- |
| `cache`              | 配置基础 batch cache                       | 多 epoch 或重复扫描 Row Group 时再开启 |
| `transforms`         | 按顺序执行的 batch transform 元组          | 从轻量组件开始逐个加，不要一次堆满     |
| `seed`               | 控制 crop、mask、dropout 的 batch 级随机性 | 消融实验固定 seed，正式多 seed 复核    |
| `strict_time_filter` | 在增广前移除未来序列事件                   | 默认保持 `True`，除非明确做吞吐对照    |

最小空管道：

```python
from taac2026.infrastructure.pcvr.config import PCVRDataPipelineConfig, PCVRTrainConfig


train_defaults = PCVRTrainConfig(
    data_pipeline=PCVRDataPipelineConfig(),
)
```

### `PCVRDataCacheConfig`

内存 cache 缓存的是 Arrow batch 转换后的基础 tensor dict，也就是增广之前的 batch。后续 transform 会拿到 clone，因此不会污染 cache 中的原始数据。

适合场景：

- 多 epoch 训练，同一 worker 会重复扫到相同 Row Group。
- `persistent_workers=True` 时，worker 内 cache 能跨 epoch 保留。
- 正在调增广组件，希望减少基础 parquet 转换开销对 benchmark 的干扰。

不适合场景：

- 单 epoch、超大数据流式训练，重复命中率低。
- `max_batches` 设得过大导致 worker 内存压力明显上升。

cache-only 示例：

```python
from taac2026.infrastructure.pcvr.config import (
    PCVRDataCacheConfig,
    PCVRDataPipelineConfig,
    PCVRTrainConfig,
)


train_defaults = PCVRTrainConfig(
    data_pipeline=PCVRDataPipelineConfig(
        cache=PCVRDataCacheConfig(mode="memory", max_batches=512),
    ),
)
```

### `PCVRSequenceCropConfig`

`sequence_crop` 对每一行的序列域生成一个或多个窗口视图，核心用途是让模型在训练期见到不同长度、不同近期窗口的行为历史。它会同步裁剪序列 token、`*_len` 和 `*_time_bucket`。

参数含义：

| 字段                            | 作用                     | 例子                       |
| ------------------------------- | ------------------------ | -------------------------- |
| `views_per_row`                 | 每条样本扩成几个训练视图 | `2` 表示一行变两行         |
| `seq_window_mode="tail"`        | 保留完整尾部窗口         | 稳定、接近原始读取         |
| `seq_window_mode="random_tail"` | 随机选择尾部窗口长度     | 模拟近期历史长度变化       |
| `seq_window_mode="rolling"`     | 随机选择任意连续窗口     | 更强扰动，适合长序列鲁棒性 |
| `seq_window_min_len`            | 随机窗口最短长度         | 避免裁得过短导致噪声过大   |

轻量近期窗口示例：

```python
from taac2026.infrastructure.pcvr.config import (
    PCVRDataPipelineConfig,
    PCVRSequenceCropConfig,
    PCVRTrainConfig,
)


train_defaults = PCVRTrainConfig(
    data_pipeline=PCVRDataPipelineConfig(
        seed=2026,
        transforms=(
            PCVRSequenceCropConfig(
                views_per_row=2,
                seq_window_mode="random_tail",
                seq_window_min_len=8,
            ),
        ),
    ),
)
```

使用建议：先从 `views_per_row=2` 开始。它会增加训练样本数和 loader 开销，收益需要用 AUC、logloss 和吞吐一起判断。

### `PCVRFeatureMaskConfig`

`feature_mask` 会随机把非序列 sparse 特征和有效序列事件置零，并压紧序列长度。它模拟特征缺失、采集失败、部分历史行为不可见等情况，目标是减少模型对单个 id 或单个事件的过拟合。

适合场景：

- 模型对少数高频 user/item sparse 特征依赖过强。
- 线上特征缺失或延迟到达较常见。
- 需要让序列 encoder 对缺失事件更稳健。

保守 feature mask 示例：

```python
from taac2026.infrastructure.pcvr.config import (
    PCVRDataPipelineConfig,
    PCVRFeatureMaskConfig,
    PCVRTrainConfig,
)


train_defaults = PCVRTrainConfig(
    data_pipeline=PCVRDataPipelineConfig(
        seed=42,
        transforms=(PCVRFeatureMaskConfig(probability=0.02),),
    ),
)
```

使用建议：`probability` 从 `0.01` 到 `0.05` 之间扫起。过高会让训练分布偏离真实样本，尤其是当前数据已经稀疏时。

### `PCVRDomainDropoutConfig`

`domain_dropout` 会按行丢弃整个序列域，例如某一行的 `seq_b` 被清零且长度归零。它模拟某个行为域完全缺失，也能迫使模型不要只依赖单一强 domain。

适合场景：

- 多个 sequence domain 信息量不均衡，模型过度依赖最强 domain。
- 线上某些 domain 覆盖率不稳定。
- 想验证模型在 domain 缺失切片上的鲁棒性。

序列域鲁棒性示例：

```python
from taac2026.infrastructure.pcvr.config import (
    PCVRDataPipelineConfig,
    PCVRDomainDropoutConfig,
    PCVRTrainConfig,
)


train_defaults = PCVRTrainConfig(
    data_pipeline=PCVRDataPipelineConfig(
        seed=42,
        transforms=(PCVRDomainDropoutConfig(probability=0.05),),
    ),
)
```

使用建议：先从 `0.02` 或 `0.05` 开始。它比单点 feature mask 更强，可能提升缺失 domain 切片，但也可能伤害全量 AUC。

### `PCVRShuffleBuffer`

shuffle buffer 不是 experiment 侧显式配置的 transform，而是 `get_pcvr_data` 用 `buffer_batches` 控制的 row-level shuffle 组件。它把多个预分批 batch 合并后按行打散，再切回 batch，减少 Row Group 内部顺序对训练的影响。

使用建议：

- `buffer_batches=1` 基本等于不做跨 batch shuffle，吞吐最高，适合 loader benchmark。
- `buffer_batches=20` 更接近训练默认值，随机性更好但会增加 tensor 拼接和 row slicing 开销。
- 做模型训练对比时固定 `buffer_batches`，否则随机性和吞吐都会变。

## 组合例子

### 保守鲁棒性组合

适合先做 A/B 的第一组增广：轻量 crop 加低概率 mask，不启用 domain dropout。

```python
from taac2026.infrastructure.pcvr.config import (
    PCVRDataCacheConfig,
    PCVRDataPipelineConfig,
    PCVRFeatureMaskConfig,
    PCVRSequenceCropConfig,
    PCVRTrainConfig,
)


train_defaults = PCVRTrainConfig(
    data_pipeline=PCVRDataPipelineConfig(
        cache=PCVRDataCacheConfig(mode="memory", max_batches=256),
        seed=42,
        strict_time_filter=True,
        transforms=(
            PCVRSequenceCropConfig(
                views_per_row=2,
                seq_window_mode="random_tail",
                seq_window_min_len=8,
            ),
            PCVRFeatureMaskConfig(probability=0.02),
        ),
    ),
)
```

### 强鲁棒性组合

适合在有足够大验证集时测试，重点观察长序列、domain 缺失和低频 item 切片。

```python
from taac2026.infrastructure.pcvr.config import (
    PCVRDataPipelineConfig,
    PCVRDomainDropoutConfig,
    PCVRFeatureMaskConfig,
    PCVRSequenceCropConfig,
    PCVRTrainConfig,
)


train_defaults = PCVRTrainConfig(
    data_pipeline=PCVRDataPipelineConfig(
        seed=2026,
        strict_time_filter=True,
        transforms=(
            PCVRSequenceCropConfig(
                views_per_row=2,
                seq_window_mode="rolling",
                seq_window_min_len=16,
            ),
            PCVRFeatureMaskConfig(probability=0.05),
            PCVRDomainDropoutConfig(probability=0.05),
        ),
    ),
)
```

### 消融用单组件组合

每次只开一个组件，方便判断收益来自哪里：

```python
from taac2026.infrastructure.pcvr.config import (
    PCVRDataPipelineConfig,
    PCVRDomainDropoutConfig,
    PCVRTrainConfig,
)


train_defaults = PCVRTrainConfig(
    data_pipeline=PCVRDataPipelineConfig(
        seed=7,
        transforms=(PCVRDomainDropoutConfig(probability=0.03),),
    ),
)
```

## 后续可扩展的增广组件

下面这些组件目前还没有实现，适合作为下一轮数据管道扩展候选。实现时仍建议保持 typed config 形态，例如 `PCVRTimeJitterConfig`、`PCVRSequenceEventDropConfig` 这类独立配置对象，并继续只在 train dataset 装配。

| 组件想法                 | 作用                                                                        | 关键参数                                    | 主要风险                                    |
| ------------------------ | --------------------------------------------------------------------------- | ------------------------------------------- | ------------------------------------------- |
| `sequence_event_drop`    | 随机删除序列中的部分有效事件，比 feature mask 更专注序列历史                | `probability`, `min_remaining_events`       | 删除过多会破坏真实兴趣轨迹                  |
| `sequence_tail_truncate` | 只随机截断尾部最近若干事件，模拟日志延迟或近期行为缺失                      | `max_drop_events`, `probability`            | 可能伤害强近期信号                          |
| `time_jitter`            | 对历史事件 timestamp 或 time bucket 加小扰动，降低对精确时间桶过拟合        | `max_seconds`, `probability`                | 必须保持事件仍早于样本 timestamp            |
| `domain_permutation`     | 随机打乱同一 domain 内部分非时间关键 side feature，测试模型是否依赖偶然共现 | `feature_ids`, `probability`                | 容易制造不真实样本，需谨慎                  |
| `feature_value_dropout`  | 只对指定 user/item fid 做 dropout，而不是所有 sparse 特征同概率             | `feature_ids`, `probability`                | 需要 schema fid 白名单，配置更复杂          |
| `dense_noise`            | 给 user dense 特征加小幅高斯噪声，提升 dense embedding 鲁棒性               | `std`, `feature_ids`                        | dense 特征可能已经归一化，噪声尺度要小      |
| `negative_resample`      | 对 batch 内负样本做轻量重采样或重权重，改善正负排序学习                     | `negative_ratio`, `hard_negative_score_key` | 会改变训练分布，必须看 logloss 和校准       |
| `mixup_embeddings_input` | 在 tensor batch 层对 dense 特征做 mixup，label 做软混合                     | `alpha`, `probability`                      | sparse/序列字段不适合直接 mixup，设计要克制 |
| `domain_token_mask`      | 只 mask 某些 domain 的某些 side feature，不丢整域                           | `domain`, `feature_slots`, `probability`    | 需要把 schema fid 映射到 sequence slot      |
| `recent_history_swap`    | 在同一 batch、同类 label 或同 user 活跃度桶内交换少量近期事件               | `swap_probability`, `bucket_key`            | 很容易引入语义噪声，优先级低                |

优先级建议：先做 `sequence_event_drop`、`feature_value_dropout`、`dense_noise`。这三类最容易和现有 batch dict 对齐，风险可控，也能分别覆盖序列缺失、稀疏特征缺失和 dense 特征扰动。`negative_resample` 更接近采样策略或 loss 设计，建议等基础增广组件的收益确认后再做。

## 时间安全

当增广配置启用且 `strict_time_filter=True` 时，序列转换阶段会先按样本级 `timestamp` 过滤事件：事件 timestamp 必须为正，并且严格早于样本 timestamp。过滤后的序列再进入 crop、masking 和 shuffle buffer。

没有开启增广时，默认训练命令保持原始读取行为。

## 内存 Cache

数据 cache 是 per-process LRU cache，缓存的是增广前的基础 tensor batch。它适合多 epoch、`persistent_workers` 或重复扫描同一 Row Group 的训练场景。

cache 会返回 clone，后续增广和 shuffle 不会污染缓存中的基础 batch。

## 吞吐压测

本地样本只有 1000 行、单个 Row Group，只适合 smoke。需要压测数据管道时，可以先生成 100x 放大模拟集：

```bash
uv run python tools/generate_pcvr_synthetic_dataset.py --force
```

默认输出到 `outputs/perf/pcvr_synthetic_100x/`，包含 100000 行和 100 个 Row Group，并复用 `sample_1000_raw` 的真实 schema。

然后只跑 loader，不跑模型：

```bash
uv run python tools/benchmark_pcvr_data_pipeline.py \
    --dataset-path outputs/perf/pcvr_synthetic_100x/demo_100000.parquet \
    --schema-path outputs/perf/pcvr_synthetic_100x/schema.json \
    --batch-size 256 \
    --num-workers 0 \
    --buffer-batches 1
```

增广路径可以用 `--pipeline-preset augment` 单独压测；训练入口本身仍然不提供数据管道覆盖参数。