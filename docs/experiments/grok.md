---
icon: lucide/flask-conical
---

# Grok

**分段序列建模 + Pairwise Loss**

## 概述

从旧 baseline 中拆分出来的本地 Grok 方案。独特之处在于使用 `segment_count=4` 进行分段建模，并启用 pairwise ranking loss 作为辅助损失。

## 模型架构

- 3 层 Transformer，4 头注意力
- Embedding 维度 128
- 4 个行为分段（segment_count=4）
- Recent sequence length 16

当前仓库实现已经接入框架级 `sparse_features` / `sequence_features` 数据流。Grok 会从 TorchRec `KeyedJaggedTensor` 重建分段历史事件流，而不再依赖实验包私有的 legacy collate 序列张量。

## 默认配置

| 参数              | 值   |
| ----------------- | ---- |
| `embedding_dim`   | 128  |
| `num_layers`      | 3    |
| `num_heads`       | 4    |
| `segment_count`   | 4    |
| `epochs`          | 10   |
| `batch_size`      | 64   |
| `learning_rate`   | 3e-4 |
| `pairwise_weight` | 0.15 |

## 快速运行

```bash
uv run taac-train --experiment config/grok
uv run taac-evaluate single --experiment config/grok
```

## 输出目录

```
outputs/config/grok/
```

## 来源

本仓库原创（从历史 baseline 拆分）。
