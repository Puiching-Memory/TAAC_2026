---
icon: lucide/flask-conical
---

# Baseline

**最小参考实现 / Starter Package**

## 概述

Baseline 是本仓库维护的 starter/reference package，强调可扩展性、注释和二次开发体验。推荐作为新实验包开发的起点。

## 模型架构

标准 Transformer 编码器，无特殊的序列-特征交互机制：

- 2 层 Transformer，4 头注意力
- Embedding 维度 96
- FFN 倍率 2.0
- 无分段建模、无 memory slots、无特征交叉层

当前仓库实现已经接入框架级 `sparse_features` / `sequence_features` 数据流。Baseline 会从 TorchRec `KeyedJaggedTensor` 重建历史事件序列和多路 sequence grid，而不再依赖实验包私有的 legacy collate 序列张量。

## 默认配置

| 参数              | 值   |
| ----------------- | ---- |
| `embedding_dim`   | 96   |
| `num_layers`      | 2    |
| `num_heads`       | 4    |
| `epochs`          | 5    |
| `batch_size`      | 64   |
| `learning_rate`   | 5e-4 |
| `pairwise_weight` | 0.0  |

## 快速运行

```bash
uv run taac-train --experiment config/baseline
uv run taac-evaluate single --experiment config/baseline
```

## 输出目录

```
outputs/config/baseline/
```

## 来源

本仓库原创。
