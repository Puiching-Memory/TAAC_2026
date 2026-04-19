---
icon: lucide/folder-open
---

# 实验包总览

## 什么是实验包

实验包是 `config/<name>/` 下的一个 Python 目录包，包含数据管道、模型架构和训练工具的完整实现。每个包导出一个 `EXPERIMENT` 对象，框架通过它驱动训练、评估和搜索。

→ 契约详情见 [架构与概念](../architecture.md)

## 配置对比

### 模型配置

| 实验包         | Embedding Dim | Layers | Heads | Segment Count | Memory Slots | Queries | 特征交叉层 |
| -------------- | :-----------: | :----: | :---: | :-----------: | :----------: | :-----: | :--------: |
| Baseline       |      96       |   2    |   4   |       0       |      0       |    1    |     0      |
| CTR Baseline   |      96       |   2    |   4   |       0       |      0       |    1    |     0      |
| DeepContextNet |      128      |   4    |   8   |       0       |      0       |    1    |     0      |
| Grok           |      128      |   3    |   4   |       4       |      0       |    1    |     0      |
| HyFormer       |      128      |   4    |   4   |      13       |      0       |    1    |     0      |
| InterFormer    |      128      |   3    |   4   |       4       |      4       |    2    |     0      |
| OneTrans       |      128      |   4    |   4   |       8       |      0       |    0    |     0      |
| O_o            |      128      |   4    |   4   |       0       |      0       |    0    |     0      |
| UniRec         |      128      |   4    |   4   |       8       |      2       |    0    |     1      |
| UniScaleFormer |      128      |   3    |   4   |       8       |      6       |    4    |     1      |

### 训练配置

| 实验包         | Epochs | Batch Size | Learning Rate | Weight Decay | Pairwise Weight |
| -------------- | :----: | :--------: | :-----------: | :----------: | :-------------: |
| Baseline       |   5    |     64     |     5e-4      |     1e-4     |       0.0       |
| CTR Baseline   |   10   |     64     |     5e-4      |     1e-4     |       0.0       |
| DeepContextNet |   10   |     32     |     2e-4      |     1e-4     |       0.0       |
| Grok           |   10   |     64     |     3e-4      |     1e-4     |      0.15       |
| HyFormer       |   10   |     64     |     1e-3      |     1e-4     |       0.0       |
| InterFormer    |   10   |     64     |     1e-3      |     1e-4     |       0.0       |
| OneTrans       |   10   |     64     |     1e-3      |     1e-4     |       0.0       |
| O_o            |   10   |     64     |     1e-3      |     1e-4     |       0.0       |
| UniRec         |   10   |     64     |     1e-3      |     1e-4     |      0.25       |
| UniScaleFormer |   10   |     64     |     8e-4      |     0.02     |       0.0       |

### 共享默认值

所有实验包共享以下数据配置：

- `max_seq_len = 32`
- `max_feature_tokens = 16`
- `stream_batch_rows = 256`
- `label_action_type = 2`（转化作为正样本；`label_type`: 0=曝光, 1=点击, 2=转化）
- `dense_feature_dim = 16`
- `vocab_size = 131072`
- `sequence_names = ("domain_a", "domain_b", "domain_c", "domain_d")`（4 个行为域）

## 按架构分类

### 基础架构

- [**Baseline**](baseline.md) — 最小参考实现，2 层 Transformer，适合快速验证和二次开发
- [**CTR Baseline**](ctr-baseline.md) — DIN 风格注意力机制，同样轻量
- [**Grok**](grok.md) — 分段序列建模 + pairwise ranking loss

### 统一 Token 化方向

- [**OneTrans**](onetrans.md) — 统一 Tokenizer 将序列/非序列特征映射到同一 Transformer（WWW 2026）
- [**O_o**](oo.md) — 简化版统一设计

### 序列-特征交互方向

- [**InterFormer**](interformer.md) — 行为感知交互块，个性化 FFN（CIKM 2025）
- [**HyFormer**](hyformer.md) — 多序列分支 + Query Decode/Boost 架构
- [**DeepContextNet**](deepcontextnet.md) — 上下文感知深度建模，标准 Transformer 风格

### 多阶段融合方向

- [**UniRec**](unirec.md) — 特征交叉 + 序列 + 静态层的多阶段融合
- [**UniScaleFormer**](uniscaleformer.md) — 缩放序列建模 + 特征融合

## 运行任意实验包

```bash
# 训练
uv run taac-train --experiment config/<name>

# 评估
uv run taac-evaluate single --experiment config/<name>

# 搜索
uv run taac-search --experiment config/<name> --trials 20
```
