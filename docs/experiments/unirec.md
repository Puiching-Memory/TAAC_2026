---
icon: lucide/flask-conical
---

# UniRec

**多阶段融合**

## 概述

UniRec 强调 unified tokenization、混合 attention mask 和 scaling law 探索，支持 2 卡 DDP 训练。独特之处在于使用独立的特征交叉层、序列层、静态层和融合层组成多阶段处理管道。

## 模型架构

- 4 层 Transformer，4 头注意力
- Embedding 维度 128
- **feature_cross_layers = 1**：特征交叉层
- **sequence_layers = 1**（通过 static_layers=3 配置）：序列建模层
- **fusion_layers = 1**：融合层（Attention Truncation）
- 2 个 memory slots
- 8 个行为分段
- Pairwise ranking loss（weight=0.25）

## 工程优化（2026-04-18 同步主干）

- **框架级 HSTU 原语复用**：`HSTUBlock`、`BranchTransducer`、`MixtureOfTransducers` 与 `BlockAttnRes` 已下沉到共享框架层，实验包只保留 UniRec 的组合逻辑
- **Block Attention Residuals 重写**：逐层 pseudo-query（初始化为零）+ last-token summary O(N·D)，等效 ~1.25x 计算优势
- **注意力 mask 缓存**：按 (seq_len, n_feature_tokens, n_special_tokens, global_window, local_window, device) 缓存，参数相同时只构建一次
- **Semi-Local Attention**：global_window=128, local_window=128
- **SiLU Gated Attention**：去除冗余归一化，与 HSTU 2.0 对齐
- **MoT CUDA stream 并行**：多分支独立 Transducer 在 GPU 上并行执行
- **共享混合优化器**：Embedding 走 RowWiseAdagrad，2D 结构化权重走 Muon，其余参数走 AdamW

当前仓库实现已经接入框架级 `sparse_features` / `sequence_features` 数据流。UniRec 会从 TorchRec `KeyedJaggedTensor` 重建多分支历史事件流，而不再依赖实验包私有的 legacy collate 序列张量。

## 默认配置

| 参数                   | 值   |
| ---------------------- | ---- |
| `embedding_dim`        | 128  |
| `num_layers`           | 4    |
| `num_heads`            | 4    |
| `segment_count`        | 8    |
| `memory_slots`         | 2    |
| `feature_cross_layers` | 1    |
| `static_layers`        | 3    |
| `fusion_layers`        | 1    |
| `recent_seq_len`       | 32   |
| `epochs`               | 10   |
| `batch_size`           | 64   |
| `learning_rate`        | 1e-3 |
| `pairwise_weight`      | 0.25 |

## 快速运行

```bash
uv run taac-train --experiment config/unirec
uv run taac-evaluate single --experiment config/unirec
```

## 输出目录

```
outputs/config/unirec/
```

## 来源

[hojiahao/TAAC2026](https://github.com/hojiahao/TAAC2026)
