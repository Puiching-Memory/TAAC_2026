---
icon: lucide/bar-chart-3
---

# CTR Baseline

基于经典 CTR 模型结构的基线实验包，使用最简单的 Group NS Tokenizer 和低 Dropout。

## 概述

CTRBaseline 是一个精简的 CTR 模型，设计目标是验证最基本的 Transformer 结构在 PCVR 任务上的表现。

## 模型架构

- 特征 Embedding 层
- Group NS Tokenizer（每组一个 token）
- Transformer 编码器（2 层）
- MLP 预测头

## 默认配置

| 参数           | 值                |
| -------------- | ----------------- |
| 模型类         | `PCVRCTRBaseline` |
| NS Tokenizer   | `group`           |
| `num_blocks`   | 2                 |
| `num_heads`    | 4                 |
| `hidden_mult`  | 4                 |
| `dropout_rate` | 0.01              |

所有实验包中 Dropout 最低（0.01）。

## 快速运行

```bash
uv run taac-train \
  --experiment config/ctr_baseline \
  --dataset-path data/sample_1000_raw/demo_1000.parquet \
  --schema-path data/sample_1000_raw/schema.json
```

## 输出目录

`outputs/pcvr_ctr_baseline-<slug>/`

## 来源

- 模型源码：`config/ctr_baseline/model.py`
- NS Groups：`config/ctr_baseline/ns_groups.json`
