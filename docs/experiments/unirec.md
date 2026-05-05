---
icon: lucide/users
---

# UniRec

统一推荐模型实验包，3 层 Transformer + RankMixer NS Tokenizer。

## 概述

UniRec 使用 `PCVRUniRec` 模型类，3 层深度与 DeepContextNet 相同，但使用 RankMixer NS Tokenizer 获取更强的特征表达能力。

## 模型架构

- 3 层 Transformer 编码器
- RankMixer NS Tokenizer (user_tokens=5, item_tokens=2)
- 统一的推荐表示学习

## 默认配置

| 参数           | 值                                         |
| -------------- | ------------------------------------------ |
| 模型类         | `PCVRUniRec`                               |
| NS Tokenizer   | `rankmixer` (user_tokens=5, item_tokens=2) |
| `num_blocks`   | 3                                          |
| `num_heads`    | 4                                          |
| `hidden_mult`  | 4                                          |
| `dropout_rate` | 0.02                                       |

## 快速运行

```bash
uv run taac-train \
  --experiment experiments/pcvr/unirec
```

## 输出目录

`outputs/pcvr_unirec-<slug>/`

## 来源

- 模型源码：`experiments/pcvr/unirec/model.py`
- NS Groups：`experiments/pcvr/unirec/__init__.py` 中的 `USER_NS_GROUPS` / `ITEM_NS_GROUPS`
