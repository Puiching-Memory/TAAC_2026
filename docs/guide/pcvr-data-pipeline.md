---
icon: lucide/database-zap
---

# PCVR 数据管道

PCVR 数据管道由多个可组合的增强组件构成，通过 `PCVRDataPipeline` 统一编排。

## 实验侧组合

各实验包在 `PCVRTrainConfig` 中配置数据管道组件：

```python
PCVRTrainConfig(
    data=PCVRDataConfig(batch_size=256, num_workers=0),
    sequence_crop=PCVRSequenceCropConfig(strategy="tail", max_len=50),
    feature_mask=PCVRFeatureMaskConfig(mask_prob=0.1),
    domain_dropout=PCVRDomainDropoutConfig(dropout_prob=0.05),
    data_cache=PCVRDataCacheConfig(enabled=True, max_batches=100),
    data_pipeline=PCVRDataPipelineConfig(shuffle_buffer_batches=8),
)
```

## 组件说明

### `PCVRDataPipelineConfig`

| 参数                     | 类型 | 说明                                |
| ------------------------ | ---- | ----------------------------------- |
| `shuffle_buffer_batches` | int  | Shuffle Buffer 窗口大小（batch 数） |
| `seed`                   | int  | 随机种子                            |

### `PCVRDataCacheConfig`

| 参数          | 类型 | 说明                  |
| ------------- | ---- | --------------------- |
| `enabled`     | bool | 是否启用内存缓存      |
| `max_batches` | int  | LRU 缓存最大 batch 数 |

### `PCVRSequenceCropConfig`

| 参数       | 类型 | 说明                                         |
| ---------- | ---- | -------------------------------------------- |
| `strategy` | str  | 裁剪策略：`tail` / `random_tail` / `rolling` |
| `max_len`  | int  | 最大序列长度                                 |
| `seed`     | int  | 随机种子                                     |

### `PCVRFeatureMaskConfig`

| 参数        | 类型  | 说明         |
| ----------- | ----- | ------------ |
| `mask_prob` | float | 特征掩码概率 |
| `seed`      | int   | 随机种子     |

### `PCVRDomainDropoutConfig`

| 参数           | 类型  | 说明            |
| -------------- | ----- | --------------- |
| `dropout_prob` | float | 域 Dropout 概率 |
| `seed`         | int   | 随机种子        |

### `PCVRShuffleBuffer`

Row-level shuffle，在 `shuffle_buffer_batches` 窗口内随机打乱样本。稳定种子由文件 CRC、Worker ID、Row Group ID、Batch Index 计算。

## 组合例子

### 保守鲁棒性组合

```python
PCVRTrainConfig(
    data_pipeline=PCVRDataPipelineConfig(shuffle_buffer_batches=4),
)
```

仅启用 Shuffle，不使用增强。

### 强鲁棒性组合

```python
PCVRTrainConfig(
    sequence_crop=PCVRSequenceCropConfig(strategy="tail", max_len=50),
    feature_mask=PCVRFeatureMaskConfig(mask_prob=0.1),
    domain_dropout=PCVRDomainDropoutConfig(dropout_prob=0.05),
    data_cache=PCVRDataCacheConfig(enabled=True, max_batches=100),
    data_pipeline=PCVRDataPipelineConfig(shuffle_buffer_batches=8),
)
```

全部组件启用，适合大规模训练。

### 消融用单组件组合

```python
# 仅序列裁剪
PCVRTrainConfig(
    sequence_crop=PCVRSequenceCropConfig(strategy="tail", max_len=50),
)

# 仅特征掩码
PCVRTrainConfig(
    feature_mask=PCVRFeatureMaskConfig(mask_prob=0.1),
)

# 仅域 Dropout
PCVRTrainConfig(
    domain_dropout=PCVRDomainDropoutConfig(dropout_prob=0.05),
)
```

## 后续可扩展的增广组件

当前管道支持的组件已在 `data_pipeline.py` 中实现。未来可扩展：

- 序列 Mask（随机遮蔽序列中的部分行为）
- 特征交叉增强
- 负采样策略

## 时间安全

所有随机操作的种子由以下因素确定性计算：

- 文件 CRC32
- Worker ID
- Row Group ID
- Batch Index

确保不同 Worker 和不同 Epoch 的随机性独立且可复现。

## 内存 Cache

`PCVRMemoryBatchCache` 是 LRU 缓存，缓存已构建的基础 batch（增强前）。适用于：

- 数据集较小，可以全部缓存到内存
- 增强操作开销较大，需要加速基础数据加载

## 吞吐压测

```bash
python tools/benchmark_pcvr_data_pipeline.py \
  --dataset-path data/sample_1000_raw/demo_1000.parquet \
  --schema-path data/sample_1000_raw/schema.json \
  --preset augment
```

预设：`none`（纯加载）、`cache`（缓存）、`augment`（全增强）。
