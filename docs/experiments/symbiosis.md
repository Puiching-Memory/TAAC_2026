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

`__init__.py` 中的 `train_defaults` 包括：

- `PCVRNSConfig(tokenizer_type="rankmixer", user_tokens=5, item_tokens=2, groups_json="ns_groups.json")`
- `PCVRModelConfig(num_blocks=3, num_heads=4, use_rope=True, rope_base=1000000.0, hidden_mult=4, dropout_rate=0.02)`
- `PCVRDataConfig(batch_size=128, num_workers=8)`
- `RuntimeExecutionConfig(amp=True, amp_dtype="bfloat16", compile=True)`
- `BinaryClassificationLossConfig(pairwise_auc_weight=0.05, pairwise_auc_temperature=1.0)`
- `PCVROptimizerConfig(dense_optimizer_type="orthogonal_adamw")`
- `PCVRSymbiosisConfig(...)` 默认启用候选条件序列解码、多分辨率记忆、attention sink、lane mixing、动作目标前移和 item semantic token。

## 模型思路

`PCVRSymbiosis` 将非序列 token、dense token、动作 prompt 和多域序列 token 放到同一表示空间中，并融合本仓库几个实验方向的长处：

- `UserItemGraphBlock` 先做用户 token 与物品 token 的双向交互，吸收图式 user-item context 的思路。
- `UnifiedBlock` 使用 RMSNorm、RoPE 可选位置编码、FiLM 调制和 SwiGLU 前馈，把序列 token、非序列 token 与动作 prompt 统一建模。
- `FourierTimeEncoder` 在离散时间桶之外补充周期时间特征，对齐广告序列中常见的时间间隔信号。
- `ContextExchangeBlock` 让非序列上下文从各序列域读取信息，并通过门控融合。
- 多尺度摘要同时读取全局均值、后半段近期行为和最后行为，再和统一 token 表征做最终门控融合。
- `CandidateConditionedSequenceDecoder` 用 user context、candidate item context、全局图上下文和 conversion action query 构造候选条件查询，逐域读取历史序列。
- 多分辨率序列记忆同时读取 recent memory、compressed block memory 和 global memory；默认每 16 个行为压缩一个 block，并选择 top-8 block 参与候选条件 attention。
- attention sink 作为 null evidence token 参与每个 domain 的序列读取，允许模型在候选与某个历史域弱相关时选择少读该域。
- 动作目标前移不猜测历史序列中哪个字段是 action，而是把 conversion action query 注入 prompt 和 candidate query，避免因 schema 无字段名而误用未来或无关字段。
- item semantic token 从 item 侧离散特征 token 聚合后投影得到，作为轻量 Semantic ID 替代路径；如果后续有离线 RQ-KMeans / RQ-VAE 码本，可继续通过 item 侧特征接入。
- `MultiLaneFusion` 对 unified context、context exchange、multi-scale summary、graph context 和 candidate sequence context 做非负归一化 lane mixing，替代简单无约束残差平均。

训练目标也已贴近 AUC 主指标：共享 binary loss 在 BCE/focal 之外支持 batch 内 positive-negative pairwise softplus 项。Symbiosis 默认 `pairwise_auc_weight=0.05`，因此仍保留概率校准约束，同时直接优化正负样本相对排序。稀疏 embedding 继续使用 Adagrad；dense 参数默认使用 `orthogonal_adamw`，在 AdamW step 前对二维以上 dense 梯度做 Muon 风格正交化，保持优化器分治而不引入新依赖。

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

## 消融开关

Symbiosis 支持一组实验开关，用于按计划验证现有模块是否真正贡献 AUC。默认值保持当前模型行为不变。

| 参数 | 默认 | 作用 |
| --- | --- | --- |
| `--symbiosis-use-user-item-graph` / `--no-symbiosis-use-user-item-graph` | 开启 | 启停 `UserItemGraphBlock` |
| `--symbiosis-use-fourier-time` / `--no-symbiosis-use-fourier-time` | 开启 | 启停额外 Fourier time encoder |
| `--symbiosis-use-context-exchange` / `--no-symbiosis-use-context-exchange` | 开启 | 启停 `ContextExchangeBlock` |
| `--symbiosis-use-multi-scale` / `--no-symbiosis-use-multi-scale` | 开启 | 启停 mean / recent / last 多尺度序列摘要 |
| `--symbiosis-use-domain-gate` / `--no-symbiosis-use-domain-gate` | 开启 | 在 `UnifiedBlock` 序列读取中用可学习 domain gate 替代简单 domain mean |
| `--symbiosis-use-candidate-decoder` / `--no-symbiosis-use-candidate-decoder` | 开启 | 启停 candidate-conditioned sequence decoder |
| `--symbiosis-use-action-conditioning` / `--no-symbiosis-use-action-conditioning` | 开启 | 启停 conversion action query 前移 |
| `--symbiosis-use-compressed-memory` / `--no-symbiosis-use-compressed-memory` | 开启 | 启停 compressed block memory |
| `--symbiosis-use-attention-sink` / `--no-symbiosis-use-attention-sink` | 开启 | 启停 null evidence sink token |
| `--symbiosis-use-lane-mixing` / `--no-symbiosis-use-lane-mixing` | 开启 | 启停非负归一化多 lane 融合 |
| `--symbiosis-use-semantic-id` / `--no-symbiosis-use-semantic-id` | 开启 | 启停 item semantic token |
| `--symbiosis-memory-block-size` | `16` | compressed memory 的行为块大小 |
| `--symbiosis-memory-top-k` | `8` | 每个 domain 选入 attention 的 compressed block 数量 |
| `--symbiosis-recent-tokens` | `64` | recent memory 保留的最近 token 数量 |

训练相关新增参数：

| 参数 | 默认 | 作用 |
| --- | --- | --- |
| `--pairwise-auc-weight` | `0.05` | BCE/focal 之外的 pairwise AUC loss 权重 |
| `--pairwise-auc-temperature` | `1.0` | pairwise margin 的 softplus 温度 |
| `--dense-optimizer-type` | `orthogonal_adamw` | dense 参数优化器，支持 `adamw` 和 `orthogonal_adamw` |

示例：

```bash
bash run.sh train --experiment config/symbiosis \
    --dataset-path /path/to/parquet_or_dataset_dir \
    --schema-path /path/to/schema.json \
    --no-symbiosis-use-user-item-graph \
    --no-symbiosis-use-context-exchange \
    --no-symbiosis-use-candidate-decoder \
    --pairwise-auc-weight 0.0 \
    --dense-optimizer-type adamw
```

## 评估输出

`taac-evaluate single` 会在 `evaluation.json` 中和 AUC 一起写出 `auc_ci`、`logloss`、`brier`、`score_diagnostics` 和顶层 `data_diagnostics`。AUC 仍是主指标，但本地或线上复盘时应同时检查置信区间、正负样本分数分布、score margin，以及数据切分是否 `is_l1_ready`。

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