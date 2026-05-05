---
icon: lucide/shuffle
---

# HyFormer

HyFormer 实验包，使用 RankMixer NS Tokenizer 和多查询解码机制。

## 概述

与 Baseline 共享 `PCVRHyFormer` 模型类，但配置差异显著：使用 RankMixer NS Tokenizer 替代 Group Tokenizer，并启用多查询解码。

## 模型架构

- **RankMixer NS Tokenizer** -- 全部 Embedding 拼接后分组投影，参数更多但表达力更强
- **多查询解码** -- `num_queries=2`，使用多个查询向量从序列中提取信息
- **RankMixerBlock** -- Token 混合 + FFN

## 与 Baseline 的区别

| 维度         | Baseline  | HyFormer  |
| ------------ | --------- | --------- |
| NS Tokenizer | group     | rankmixer |
| user_tokens  | 0         | 5         |
| item_tokens  | 0         | 2         |
| num_queries  | 1（默认） | 2         |

## 默认配置

| 参数                 | 值                                         |
| -------------------- | ------------------------------------------ |
| 模型类               | `PCVRHyFormer`                             |
| NS Tokenizer         | `rankmixer` (user_tokens=5, item_tokens=2) |
| `num_queries`        | 2                                          |
| `emb_skip_threshold` | 1,000,000                                  |

## 快速运行

```bash
uv run taac-train \
  --experiment experiments/pcvr/hyformer
```

## 输出目录

`outputs/pcvr_hyformer-<slug>/`

## 来源

- 模型源码：`experiments/pcvr/hyformer/model.py`
- NS Groups：`experiments/pcvr/hyformer/__init__.py` 中的 `USER_NS_GROUPS` / `ITEM_NS_GROUPS`
- HyFormer 论文解读：[papers/hyformer.md](../papers/hyformer.md)
