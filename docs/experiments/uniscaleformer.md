---
icon: lucide/flask-conical
---

# UniScaleFormer

**缩放序列建模 + 特征融合**

## 概述

UniScaleFormer 内置 InterFormer / OneTrans / HyFormer / base 配置对比与 scaling law 脚本，是当前实验包中正则化最强（weight_decay=0.02）、memory slots 最多（6 个）的包。

## 模型架构

- 3 层 Transformer，4 头注意力
- Embedding 维度 128
- **feature_cross_layers = 1**：特征交叉层
- **sequence_layers = 1** / **static_layers = 1**
- **fusion_layers = 1**：融合层
- **6 个 memory slots**
- **4 个 Query**
- 8 个行为分段

当前仓库实现已经接入框架级 `sparse_features` / `sequence_features` 数据流。UniScaleFormer 会从 TorchRec `KeyedJaggedTensor` 重建历史事件流并按语义拆回多条序列，而不再依赖实验包私有的 legacy collate 序列张量。

## 默认配置

| 参数                   | 值   |
| ---------------------- | ---- |
| `embedding_dim`        | 128  |
| `num_layers`           | 3    |
| `num_heads`            | 4    |
| `segment_count`        | 8    |
| `memory_slots`         | 6    |
| `num_queries`          | 4    |
| `feature_cross_layers` | 1    |
| `sequence_layers`      | 1    |
| `static_layers`        | 1    |
| `fusion_layers`        | 1    |
| `epochs`               | 10   |
| `batch_size`           | 64   |
| `learning_rate`        | 8e-4 |
| `weight_decay`         | 0.02 |
| `pairwise_weight`      | 0.0  |

## 快速运行

```bash
uv run taac-train --experiment config/uniscaleformer
uv run taac-evaluate single --experiment config/uniscaleformer
```

## 输出目录

```
outputs/config/uniscaleformer/
```

## 来源

[twx145/Unirec](https://github.com/twx145/Unirec)
