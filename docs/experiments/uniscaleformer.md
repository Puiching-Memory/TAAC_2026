---
icon: lucide/trending-up
---

# UniScaleFormer

多尺度统一 Transformer，所有实验包中层数最深（4 层）。

## 概述

UniScaleFormer 使用最深的 Transformer 栈（4 层），探索模型深度的上限。使用 RankMixer NS Tokenizer。

## 模型架构

- 4 层 Transformer 编码器（所有实验包中最深）
- RankMixer NS Tokenizer (user_tokens=5, item_tokens=2)
- 多尺度特征融合

## 默认配置

| 参数           | 值                                         |
| -------------- | ------------------------------------------ |
| 模型类         | `PCVRUniScaleFormer`                       |
| NS Tokenizer   | `rankmixer` (user_tokens=5, item_tokens=2) |
| `num_blocks`   | 4                                          |
| `num_heads`    | 4                                          |
| `hidden_mult`  | 4                                          |
| `dropout_rate` | 0.02                                       |

## 快速运行

```bash
uv run taac-train \
  --experiment experiments/pcvr/uniscaleformer
```

## 输出目录

`outputs/pcvr_uniscaleformer-<slug>/`

## 来源

- 模型源码：`experiments/pcvr/uniscaleformer/model.py`
- NS Groups：`experiments/pcvr/uniscaleformer/__init__.py` 中的 `USER_NS_GROUPS` / `ITEM_NS_GROUPS`
