---
icon: lucide/flask-conical
---

# DeepContextNet

## 概述

`config/deepcontextnet` 是一个 PCVR 实验包，使用 `PCVRExperiment` 接入共享 runtime。包内 `model.py` 暴露 `PCVRDeepContextNet`，核心思路是用非序列 token 得到上下文表示，再让上下文向各域序列做注意力读取。

## 包结构

```text
config/deepcontextnet/
├── __init__.py
├── model.py
└── ns_groups.json
```

`__init__.py` 中的实验定义：

| 字段 | 值 |
| --- | --- |
| `name` | `pcvr_deepcontextnet` |
| `model_class_name` | `PCVRDeepContextNet` |
| `--ns_tokenizer_type` | `group` |
| `--ns_groups_json` | `ns_groups.json` |
| `--num_blocks` | `3` |
| `--num_heads` | `4` |
| `--hidden_mult` | `4` |
| `--dropout_rate` | `0.02` |

## 模型要点

- `NonSequentialTokenizer` 编码 user/item 稀疏特征组。
- `DenseTokenProjector` 把 user/item dense 特征投影为 token。
- `SequenceTokenizer` 编码各域行为序列，并加 sinusoidal position。
- `ContextBlock` 使用 `nn.MultiheadAttention` 让上下文 token 读取序列信息。
- `predict(inputs)` 返回 `(logits, embeddings)`，供共享评估/推理流程使用。

## 快速运行

```bash
bash run.sh train --experiment config/deepcontextnet \
	--dataset-path /path/to/parquet_or_dataset_dir \
	--schema-path /path/to/schema.json

bash run.sh val --experiment config/deepcontextnet \
	--dataset-path /path/to/parquet_or_dataset_dir \
	--schema-path /path/to/schema.json \
	--run-dir outputs/config/deepcontextnet
```

## 打包

```bash
bash run.sh package --experiment config/deepcontextnet \
	--output-dir outputs/training_bundles/deepcontextnet_training_bundle \
	--force
```

打包产物会包含 `config/deepcontextnet/model.py` 与 `config/deepcontextnet/ns_groups.json`。

## 来源

[suyanli220/TAAC-2026-Baseline-Tencent-Advertisement-Contest](https://github.com/suyanli220/TAAC-2026-Baseline-Tencent-Advertisement-Contest)
