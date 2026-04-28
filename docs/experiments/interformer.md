---
icon: lucide/arrows-up-from-line
---

# InterFormer

交叉注意力 Transformer，使用独立的模型类和 Group NS Tokenizer。

## 概述

InterFormer 使用 `PCVRInterFormer` 模型类，核心特点是用户和物品特征之间的交叉注意力机制。

## 模型架构

- 用户特征和物品特征分别编码
- 通过交叉注意力在用户-物品之间进行交互
- Group NS Tokenizer 用于非序列特征分组
- 2 层 Transformer 编码器

## 默认配置

| 参数           | 值                |
| -------------- | ----------------- |
| 模型类         | `PCVRInterFormer` |
| NS Tokenizer   | `group`           |
| `num_blocks`   | 2                 |
| `num_heads`    | 4                 |
| `hidden_mult`  | 4                 |
| `dropout_rate` | 0.02              |

## 快速运行

```bash
uv run taac-train \
  --experiment config/interformer \
  --dataset-path data/sample_1000_raw/demo_1000.parquet \
  --schema-path data/sample_1000_raw/schema.json
```

## 输出目录

`outputs/pcvr_interformer-<slug>/`

## 来源

- 模型源码：`config/interformer/model.py`
- NS Groups：`config/interformer/ns_groups.json`
- InterFormer 论文解读：[papers/interformer.md](../papers/interformer.md)
