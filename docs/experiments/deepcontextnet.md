---
icon: lucide/layers
---

# DeepContextNet

深层上下文网络，使用 3 层 Transformer 和 Group NS Tokenizer，在 group 类实验包中层数最深。

## 概述

DeepContextNet 通过增加 Transformer 层深度来捕获更丰富的上下文信息。与 Baseline 和 InterFormer 相同使用 Group NS Tokenizer，但多 1 层 Transformer。

## 包结构

```
config/deepcontextnet/
├── __init__.py
├── model.py
└── ns_groups.json
```

## 模型要点

- 3 层 Transformer 编码器（比 baseline 和 interformer 多 1 层）
- Group NS Tokenizer
- 深层特征交互

## 默认配置

| 参数           | 值                   |
| -------------- | -------------------- |
| 模型类         | `PCVRDeepContextNet` |
| NS Tokenizer   | `group`              |
| `num_blocks`   | 3                    |
| `num_heads`    | 4                    |
| `hidden_mult`  | 4                    |
| `dropout_rate` | 0.02                 |

## 快速运行

```bash
uv run taac-train \
  --experiment config/deepcontextnet \
  --dataset-path data/sample_1000_raw/demo_1000.parquet \
  --schema-path data/sample_1000_raw/schema.json
```

## 打包

```bash
uv run taac-package-train --experiment config/deepcontextnet --output-dir outputs/bundle
```

## 来源

- 模型源码：`config/deepcontextnet/model.py`
- NS Groups：`config/deepcontextnet/ns_groups.json`
