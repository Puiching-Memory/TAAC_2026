---
icon: lucide/flask-conical
---

# HyFormer

**多序列分支 + Query Decode/Boost**

## 概述

HyFormer（Hybrid Former）重新审视了 CTR 预估中序列建模与特征交互的角色定位。核心思想是将用户历史按语义分组为多条独立序列，分别编码后通过可学习 Query 进行跨序列注意力解码和逐 token 增强。

→ 完整论文笔记见 [HyFormer 论文](../papers/hyformer.md)

## 模型架构

- **SemanticGroupedNSTokenizer**：将非序列特征分为 5 个基础组 + 稠密衍生 NS token
- **MultiSequenceEventTokenizer**：将历史行为按 `segment_count=13` 分组为 13 条独立序列
- **SequenceEncoderLayer**：每条序列独立过 Transformer
- **QueryGenerator**：为每条序列生成可学习 Query
- **QueryDecodingBlock**：跨序列 Cross-Attention
- **QueryBoosting**：逐 token FFN 增强

当前仓库实现已经接入框架级 `sparse_features` / `sequence_features` 数据流。HyFormer 会从 TorchRec `KeyedJaggedTensor` 重建历史事件流并按语义分组拆回多条序列，而不再依赖实验包私有的 legacy collate 序列张量。

关键特性：独立的位置编码和序列 ID 编码。

## 默认配置

| 参数              | 值   |
| ----------------- | ---- |
| `embedding_dim`   | 128  |
| `num_layers`      | 4    |
| `num_heads`       | 4    |
| `segment_count`   | 13   |
| `recent_seq_len`  | 0    |
| `epochs`          | 10   |
| `batch_size`      | 64   |
| `learning_rate`   | 1e-3 |
| `pairwise_weight` | 0.0  |

## 快速运行

```bash
uv run taac-train --experiment config/hyformer
uv run taac-evaluate single --experiment config/hyformer
```

## 输出目录

```
outputs/config/hyformer/
```

## 来源

[论文：Revisiting the Roles of Sequence Modeling and Feature Interaction in CTR Prediction](https://arxiv.org/abs/2601.12681)（ByteDance）
