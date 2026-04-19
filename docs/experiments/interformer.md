---
icon: lucide/flask-conical
---

# InterFormer

**双向序列-特征交互**

## 概述

InterFormer 提出了行为感知交互块（Behavior-Aware Interaction Block），实现序列建模与特征交互之间的双向信息流。此外引入了个性化 FFN（PersonalizedFFN），根据上下文摘要动态生成用户级别的网络权重。

→ 完整论文笔记见 [InterFormer 论文](../papers/interformer.md)

## 模型架构

- **SelfGating**：门控值路径
- **StandardSelfAttention**：标准自注意力
- **LinearCompressedEmbedding**：学习输入→输出 token 压缩
- **SequencePreprocessor**：融合历史/发布/作者/时间/动作/分组特征
- **PersonalizedFFN**：从上下文摘要生成用户特定权重
- **Memory Slots**：4 个可学习记忆槽

当前仓库实现已经接入框架级 `sparse_features` / `sequence_features` 数据流。InterFormer 会从 TorchRec `KeyedJaggedTensor` 重建非序列与历史事件 token 网格，而不再依赖实验包私有的 legacy collate 张量。

关键特性：自注意力 + 交叉注意力上下文的门控融合，2 个可学习 Query。

## 默认配置

| 参数              | 值   |
| ----------------- | ---- |
| `embedding_dim`   | 128  |
| `num_layers`      | 3    |
| `num_heads`       | 4    |
| `segment_count`   | 4    |
| `memory_slots`    | 4    |
| `num_queries`     | 2    |
| `recent_seq_len`  | 2    |
| `epochs`          | 10   |
| `batch_size`      | 64   |
| `learning_rate`   | 1e-3 |
| `pairwise_weight` | 0.0  |

## 快速运行

```bash
uv run taac-train --experiment config/interformer
uv run taac-evaluate single --experiment config/interformer
```

## 输出目录

```
outputs/config/interformer/
```

## 来源

[论文：Effective Heterogeneous Interaction Learning for Click-Through Rate Prediction](https://arxiv.org/abs/2411.09852)（Meta AI，CIKM 2025）
