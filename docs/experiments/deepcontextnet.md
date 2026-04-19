---
icon: lucide/flask-conical
---

# DeepContextNet

**上下文感知深度建模**

## 概述

DeepContextNet 当前在仓库里的实现是标准 Transformer 风格的上下文序列建模实验包，不是框架级 HSTU 实现。它复用了默认数据管道、默认 ranking loss、TorchRec sparse embedding 路径，以及框架默认的混合优化器路由。

## 模型架构

- 4 层 Transformer，**8 头**注意力
- Embedding 维度 128
- Recent sequence length 32（最长）
- Batch size 32（最小，匹配更大模型的显存需求）
- 默认走框架 `FeatureSchema` + `TorchRecEmbeddingBagAdapter`
- 默认数据管道与默认 loss builder 已交给框架层处理

当前仓库实现已经接入框架级 `sparse_features` / `sequence_features` 数据流。DeepContextNet 会从 TorchRec `KeyedJaggedTensor` 重建最近行为上下文，而不再依赖实验包私有的 legacy collate 序列张量。

## 默认配置

| 参数              | 值   |
| ----------------- | ---- |
| `embedding_dim`   | 128  |
| `num_layers`      | 4    |
| `num_heads`       | 8    |
| `epochs`          | 10   |
| `batch_size`      | 32   |
| `learning_rate`   | 2e-4 |
| `pairwise_weight` | 0.0  |

## 快速运行

```bash
uv run taac-train --experiment config/deepcontextnet
uv run taac-evaluate single --experiment config/deepcontextnet
```

## 当前自定义部分

- `model.py`：保留 DeepContextNet 自己的序列建模块
- `__init__.py`：`build_data_pipeline=None`、`build_loss_stack=None`、`build_optimizer_component=None`，训练侧完全复用框架默认 builder
- `utils.py`：仅保留兼容性 helper，不再承载独立优化器实现

## 输出目录

```
outputs/config/deepcontextnet/
```

## 来源

[suyanli220/TAAC-2026-Baseline-Tencent-Advertisement-Contest](https://github.com/suyanli220/TAAC-2026-Baseline-Tencent-Advertisement-Contest)
