---
icon: lucide/flask-conical
---

# Baseline

**官方 Day0 参考实现 / Official PCVR HyFormer**

## 概述

Baseline 已切换为官方页面提供的 Day0 baseline，官方数据集、模型、训练器、工具函数、`ns_groups.json` 与运行脚本都已同步到 `config/baseline`。框架入口读取官方 `schema.json`，构造 `config.baseline.dataset.PCVRParquetDataset`，并把 batch dict 转为 `config.baseline.model.ModelInput` 后调用 `PCVRHyFormer`。

## 模型架构

官方 baseline 使用 `PCVRHyFormer`：

- 4 个序列域：`seq_a`、`seq_b`、`seq_c`、`seq_d`
- RankMixer tokenizer：`user_ns_tokens=5`、`item_ns_tokens=2`
- `num_queries=2`，默认 hidden / embedding 维度为 64
- 稀疏参数使用官方模型暴露的 `get_sparse_params()`，默认 optimizer 会用 Adagrad + AdamW 组合

官方管道的 batch 是 Python `dict`，核心键包括：`user_int_feats`、`item_int_feats`、`user_dense_feats`、`item_dense_feats`、`label`、`timestamp`、`user_id`、`_seq_domains`，以及每个序列域的 `seq_*`、`seq_*_len`、`seq_*_time_bucket`。Baseline 不再消费旧的 `BatchTensors.sparse_features` / `sequence_features`。

## 默认配置

| 参数              | 值   |
| ----------------- | ---- |
| `embedding_dim`   | 64   |
| `num_layers`      | 2    |
| `num_heads`       | 4    |
| `num_queries`     | 2    |
| `epochs`          | 999  |
| `batch_size`      | 256  |
| `learning_rate`   | 1e-4 |
| `sparse_lr`       | 5e-2 |
| `pairwise_weight` | 0.0  |

## 快速运行

```bash
bash run.sh train --experiment config/baseline \
  --dataset-path /path/to/official_parquet_dir \
  --schema-path /path/to/official_parquet_dir/schema.json

bash run.sh val --experiment config/baseline \
  --dataset-path /path/to/official_parquet_dir \
  --schema-path /path/to/official_parquet_dir/schema.json
```

如果 `schema.json` 与 parquet 文件位于同一目录，可以省略 `--schema-path`。当前官方 baseline 管道要求本地 parquet 文件或 parquet 目录；缺少 schema 时会直接报错，避免静默落回旧格式。

## 输出目录

```
outputs/config/baseline/
```

## 来源

官方 Day0 baseline，仓库内实现位于 `config/baseline/`。
