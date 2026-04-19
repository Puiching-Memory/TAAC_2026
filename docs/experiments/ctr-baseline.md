---
icon: lucide/flask-conical
---

# CTR Baseline

**DIN 风格注意力 CTR 预估**

## 概述

基于 DIN（Deep Interest Network）注意力机制的 CTR baseline，采用流式清洗与单机训练工程化设计。

## 模型架构

- 2 层 Transformer，4 头注意力
- Embedding 维度 96
- DIN 风格的目标-兴趣注意力
- 无特征交叉层

当前仓库实现已经接入框架级 `sparse_features` / `sequence_features` 数据流。CTR Baseline 会从 TorchRec `KeyedJaggedTensor` 重建历史事件三元组，而不再依赖实验包私有的 legacy collate 序列张量。

## 默认配置

| 参数              | 值   |
| ----------------- | ---- |
| `embedding_dim`   | 96   |
| `num_layers`      | 2    |
| `num_heads`       | 4    |
| `epochs`          | 10   |
| `batch_size`      | 64   |
| `learning_rate`   | 5e-4 |
| `pairwise_weight` | 0.0  |

## 快速运行

```bash
uv run taac-train --experiment config/ctr_baseline
uv run taac-evaluate single --experiment config/ctr_baseline
```

## 输出目录

```
outputs/config/ctr_baseline/
```

## 来源

[creatorwyx/TAAC2026-CTR-Baseline](https://github.com/creatorwyx/TAAC2026-CTR-Baseline)
