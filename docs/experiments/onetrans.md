---
icon: lucide/flask-conical
---

# OneTrans

**统一 Tokenizer + 单 Transformer**

## 概述

OneTrans 提出在单个 Transformer 架构内同时处理用户行为序列和非序列多域特征。核心创新是统一 Tokenization 方案和混合因果注意力（Mixed Causal Attention），让序列 token 和非序列 token 使用不同的参数矩阵。

→ 完整论文笔记见 [OneTrans 论文](../papers/onetrans.md)

## 模型架构

- **AutoSplitNSTokenizer**：自动将非序列特征分区为 NS token
- **UnifiedSequentialTokenizer**：将行为事件与分组分隔符合并为统一序列
- **RMSNorm**：使用 RMSNorm 替代 LayerNorm
- **MixedCausalAttention**：token 类型级别的权重矩阵（NS token 有独立的 Q/K/V 投影）
- **MixedFFN**：token 类型级别的 up/down 投影

当前仓库实现已经接入框架级 `sparse_features` / `sequence_features` 数据流。OneTrans 会从 TorchRec `KeyedJaggedTensor` 重建统一行为事件流与分组分隔符，而不再依赖实验包私有的 legacy collate 序列张量。

关键特性：行为组之间插入分隔符，因果掩码基于位置的 Q/K 拆分。

## 默认配置

| 参数              | 值   |
| ----------------- | ---- |
| `embedding_dim`   | 128  |
| `num_layers`      | 4    |
| `num_heads`       | 4    |
| `segment_count`   | 8    |
| `recent_seq_len`  | 0    |
| `num_queries`     | 0    |
| `epochs`          | 10   |
| `batch_size`      | 64   |
| `learning_rate`   | 1e-3 |
| `pairwise_weight` | 0.0  |

## 快速运行

```bash
uv run taac-train --experiment config/onetrans
uv run taac-evaluate single --experiment config/onetrans
```

## 输出目录

```
outputs/config/onetrans/
```

## 来源

[论文：Unified Feature Interaction and Sequence Modeling with One Transformer in Industrial Recommender](https://arxiv.org/abs/2510.26104)（NTU + ByteDance，WWW 2026 accepted）
