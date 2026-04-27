---
icon: lucide/network
---

# Symbiosis

`config/symbiosis` 是本仓库维护的融合式 PCVR 实验包。它使用 `PCVRExperiment` 接入共享 runtime，并在包内 `model.py` 暴露 `PCVRSymbiosis`。

## 包结构

```text
config/symbiosis/
├── __init__.py
├── model.py
└── ns_groups.json
```

`__init__.py` 中的默认训练参数包括：

- `--ns_tokenizer_type rankmixer`
- `--user_ns_tokens 5`
- `--item_ns_tokens 2`
- `--ns_groups_json ns_groups.json`
- `--num_blocks 3`
- `--num_heads 4`
- `--use_rope`
- `--rope_base 1000000.0`
- `--hidden_mult 4`
- `--dropout_rate 0.02`

## 模型思路

`PCVRSymbiosis` 将非序列 token、dense token、动作 prompt 和多域序列 token 放到同一表示空间中，并融合本仓库几个实验方向的长处：

- `UserItemGraphBlock` 先做用户 token 与物品 token 的双向交互，吸收图式 user-item context 的思路。
- `UnifiedBlock` 使用 RMSNorm、RoPE 可选位置编码、FiLM 调制和 SwiGLU 前馈，把序列 token、非序列 token 与动作 prompt 统一建模。
- `FourierTimeEncoder` 在离散时间桶之外补充周期时间特征，对齐广告序列中常见的时间间隔信号。
- `ContextExchangeBlock` 让非序列上下文从各序列域读取信息，并通过门控融合。
- 多尺度摘要同时读取全局均值、后半段近期行为和最后行为，再和统一 token 表征做最终门控融合。

最终模型输出 logits；`predict()` 返回 logits 和融合后的 embedding，满足共享 PCVR 评估与推理契约。

## 训练

```bash
bash run.sh train --experiment config/symbiosis \
    --dataset-path /path/to/parquet_or_dataset_dir \
    --schema-path /path/to/schema.json \
    --num_epochs 1 \
    --batch_size 8 \
    --device cpu
```

## 评估

```bash
bash run.sh val --experiment config/symbiosis \
    --dataset-path /path/to/parquet_or_dataset_dir \
    --schema-path /path/to/schema.json \
    --run-dir outputs/config/symbiosis \
    --device cpu
```

## 线上打包

```bash
uv run taac-package-train --experiment config/symbiosis \
    --output-dir outputs/training_bundles/symbiosis_training_bundle \
    --force
```

打包产物会包含 `config/symbiosis/model.py` 与 `config/symbiosis/ns_groups.json`。