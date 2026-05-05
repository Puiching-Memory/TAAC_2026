---
icon: lucide/activity
---

# Baseline

基准实验包，使用 HyFormer 模型和默认配置，作为所有实验的参照基线。

## 概述

Baseline 使用 `PCVRHyFormer` 模型类，Group NS Tokenizer，所有超参数取框架默认值。设计目标是提供一个干净的基准参考点。

## 模型架构

HyFormer 是一个多查询混合 Transformer，核心组件：

- **FeatureEmbeddingBank** -- 管理所有特征的 Embedding 表
- **GroupNSTokenizer** -- 按 NS Groups 将非序列特征分组，每组生成一个 token
- **SequenceTokenizer** -- 将序列特征编码为 token 序列
- **MultiSeqHyFormerBlock** -- 序列演化 + 查询解码 + 查询增强
- **CrossAttention** -- 查询向量与序列 token 的交叉注意力
- **RankMixerBlock** -- Token 混合 + FFN

## 默认配置

| 参数                 | 值             |
| -------------------- | -------------- |
| 模型类               | `PCVRHyFormer` |
| NS Tokenizer         | `group`        |
| `emb_skip_threshold` | 1,000,000      |
| `num_blocks`         | 2（默认）      |
| `num_heads`          | 4（默认）      |
| `dropout_rate`       | 0.01（默认）   |

## 快速运行

```bash
uv run taac-train \
  --experiment experiments/baseline
```

## 输出目录

训练产物保存在 `outputs/pcvr_baseline-<slug>/`：

```
outputs/pcvr_baseline-<slug>/
├── global_step<N>.{params}.best_model/
│   ├── model.safetensors
│   ├── schema.json
│   └── train_config.json
└── events.out.tfevents.*    # TensorBoard 日志
```

## 来源

- 模型源码：`experiments/baseline/model.py`
- NS Groups：`experiments/baseline/__init__.py` 中的 `USER_NS_GROUPS` / `ITEM_NS_GROUPS`
- 训练配置：`experiments/baseline/__init__.py`
- HyFormer 论文解读：[papers/hyformer.md](../papers/hyformer.md)
