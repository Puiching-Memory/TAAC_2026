---
icon: lucide/lightbulb
---

# 从直播推荐实战看特征 Token 化、长序列注意力与辅助 Loss 设计

:material-calendar: 2026-04-12 · :material-tag: 特征交叉, 序列建模, 工业实践

## 原文章出处

- **标题**：推荐算法模型工作总结（暑期版）
- **作者**：比巴卜
- **链接**：<https://zhuanlan.zhihu.com/p/2025299319658493425>
- **日期**：2026-04-12
- **背景**：作者在某大厂推荐算法组实习期间的模型迭代工作总结，涵盖直播推荐场景下的多项模型优化实践。

<!-- more -->

## AI 解读

文章覆盖了 7 个完整的模型迭代方向，以下按与 TAAC 2026 赛题的相关度排序提炼：

### 高相关：特征交叉与 Token 化

**RankMixer 迭代**：用 RankMixer 替换 FM 做特征交叉，从暴力切割 Token → 语义切分 Token + MoE + mean pooling 逐步演进。核心收获：

- MoE 需要 balancing loss 防止 expert 坍缩，但系数 $\alpha$ 过大会抹杀 specialization
- 各预估目标 AUC 均有 0.1+ pp 提升
- 后续 TokenMixer 在 RankMixer 基础上加 CLS token + 辅助 loss 训练中间层

**分级类目生成式辅助学习**：在 TokenMixer 之后接 Decoder，利用类目的层级结构做 NTP（Next Token Prediction）辅助任务，只训练不推理。半因果 mask。思路是用生成式任务增强 Token 表征质量。

### 高相关：长序列建模

**STCA（Stacked Target Cross-Attention）机制**：

1. SWiGLU 替换 FFN 映射 Q，增强非线性
2. 多层堆叠 attention，每层输出经 SWiGLU 后拼接 Q 作为下一层输入
3. KV 复用（历史事实不变，变的是"怎么看"历史的 Q）

关键发现：STCA 对有时序关系的行为序列效果好，对共现关系构造的序列效果变差。

**行为序列融合**：将分行为序列（观看/点赞/关注/评论）按时间 diff 融合为统一序列，构造行为标签嵌入。行为标签在 QK 之前融入效果优于 QK 之后加 bias——前者让 attention score 能区分同一物品的不同行为。

### 中等相关：多任务与用户建模

**UIM（User-specific ID embedding per task）**：每个预估目标塔维护独立的 user_id embedding table，梯度独立更新。判断标准：各塔 user embedding 余弦相似度 > 0.95 则没必要加。

**观看时长预估**：对比了指数分布拟合、CREAD 分时段多头、YouTube WLR 三种方案。直播场景无完播概念，最终采用 CREAD。

### 较低相关：实时化

**模型实时化**三板斧：实时特征构造（user 侧重定向 + item 侧信号放大）、延迟样本回补（双窗口 + 补偿梯度 $-1$）、独立实时塔（logits 相加 = 乘性 odds）。这块与离线 batch 评估的 TAAC 赛制关系不大。

### 观测指标体系（值得借鉴）

| 场景     | 指标                                                       |
| -------- | ---------------------------------------------------------- |
| 特征交叉 | Expert 负载均衡度、Expert 输出相似度、NDCG                 |
| 序列建模 | 注意力分布熵（逐层递减）、注意力权重位置衰减曲线、截断消融 |
| 多任务   | 各塔 user embedding 余弦相似度、梯度幅度、分任务 PCOC      |
| 时长预估 | MAE、RMSE、NDCG、分时长桶预估偏差                          |
| 校准     | 分桶 LogLoss、ECE（加权 PCOC）                             |

### 对 TAAC 2026 的适用性分析

**高价值方向：**

- **RankMixer / TokenMixer 方向**：与当前 OneTrans / HyFormer 等统一 Token 化方向高度吻合。RankMixer 的语义切分 Token + MoE 可以理解为另一种特征交叉 Token 化策略，值得与现有 `feature_cross_layers` 机制对比
- **STCA 长序列**：当前实验包 `max_seq_len=32` 较短，但 STCA 的 KV 复用 + 多层 Q 堆叠思路在扩大序列长度时可直接应用。比起简单堆层，参数量更可控
- **行为序列融合**：当前数据管道按 `action_type` 区分行为，融合为统一时间序列的思路可在 `data.py` / tokenizer 层面实验
- **生成式辅助 loss**：类目层级 NTP 作为只训练不推理的辅助任务，不影响推理延迟约束（180s），可在 `build_loss_stack` 中加入

**风险点：**

- 文章基于直播推荐，强实时性需求与 TAAC 的离线 batch 评估不同，实时化部分不直接适用
- MoE 引入额外复杂度，在 3 GiB 参数量约束下需要谨慎控制 expert 数量
- UIM 需要 user_id 粒度的 embedding table，样例数据量（1000 条）太小无法验证效果，需等正式数据

**与现有架构的映射：**

- STCA 可作为 HyFormer `SequenceEncoderLayer` 或 InterFormer `StandardSelfAttention` 的替换实验
- 行为序列融合与 OneTrans 的 `UnifiedSequentialTokenizer` 思路相近，但 OneTrans 用分隔符而非行为标签嵌入
- 生成式辅助 loss 可直接通过 `LayerStack` 的 `Packet/Blackboard` 机制接入

## 我们的看法

*（待补充）*

## 实施清单

- [ ] **STCA 注意力机制**：在 baseline 或 interformer 的 `model.py` 中实现 STCA（SWiGLU-Q + 多层堆叠 + KV 复用），作为新实验包 `config/stca/`
- [ ] **行为序列融合 tokenizer**：在 `data.py` 中实现按时间排序的统一行为序列构造，将 `action_type` 作为嵌入特征融入 token 表示
- [ ] **生成式辅助 loss**：在 `utils.py` 的 `build_loss_stack` 中加入类目层级 NTP 辅助任务（半因果 mask，只训练不推理）
- [ ] **MoE 特征交叉**：在 RankMixer 思路下实现 MoE 版特征交叉层，替换现有 `feature_cross_layers`，注意加 balancing loss
- [ ] **观测指标扩充**：在 `metrics.py` 或 profiling 中增加注意力分布熵、Expert 负载均衡度等诊断指标
