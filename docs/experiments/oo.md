---
icon: lucide/flask-conical
---

# O_o

**简化统一设计**

## 概述

O_o 是 2025 届 TAAC 初赛 Top 1% 队伍的公开方案，采用简化的统一建模设计。

## 模型架构

- 4 层 Transformer，4 头注意力
- Embedding 维度 128
- 无分段建模、无 memory slots、无 Query
- 简洁的端到端设计
- 时间感知 HSTU、相对时间偏置与 Fourier 时间编码直接复用框架 `src/taac2026/infrastructure/nn/hstu.py`

当前仓库实现已经接入框架级 `sparse_features` / `sequence_features` 数据流。O_o 会从 TorchRec `KeyedJaggedTensor` 重建时间感知历史事件流，而不再依赖实验包私有的 legacy collate 序列张量。

## 默认配置

| 参数              | 值   |
| ----------------- | ---- |
| `embedding_dim`   | 128  |
| `num_layers`      | 4    |
| `num_heads`       | 4    |
| `segment_count`   | 0    |
| `num_queries`     | 0    |
| `epochs`          | 10   |
| `batch_size`      | 64   |
| `learning_rate`   | 1e-3 |
| `pairwise_weight` | 0.0  |

## 快速运行

```bash
uv run taac-train --experiment config/oo
uv run taac-evaluate single --experiment config/oo
```

## 输出目录

```
outputs/config/oo/
```

## 来源

[salmon1802/O_o](https://github.com/salmon1802/O_o)（2025 初赛 Top 1%）
