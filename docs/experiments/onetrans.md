---
icon: lucide/arrow-right-left
---

# OneTrans

单 Transformer 实验包，将用户和物品特征统一编码，使用 RankMixer NS Tokenizer。

## 概述

OneTrans 使用 `PCVROneTrans` 模型类，特点是将所有特征（用户 + 物品 + 序列）拼接后统一输入单个 Transformer。

## 模型架构

- 所有特征拼接后统一编码（不区分用户/物品分支）
- RankMixer NS Tokenizer (user_tokens=5, item_tokens=2)
- 2 层 Transformer 编码器

## 默认配置

| 参数           | 值                                         |
| -------------- | ------------------------------------------ |
| 模型类         | `PCVROneTrans`                             |
| NS Tokenizer   | `rankmixer` (user_tokens=5, item_tokens=2) |
| `num_blocks`   | 2                                          |
| `num_heads`    | 4                                          |
| `hidden_mult`  | 4                                          |
| `dropout_rate` | 0.02                                       |

## 快速运行

```bash
uv run taac-train \
  --experiment experiments/onetrans
```

## 输出目录

`outputs/pcvr_onetrans-<slug>/`

## 来源

- 模型源码：`experiments/onetrans/model.py`
- NS Groups：`experiments/onetrans/__init__.py` 中的 `USER_NS_GROUPS` / `ITEM_NS_GROUPS`
- OneTrans 论文解读：[papers/onetrans.md](../papers/onetrans.md)
