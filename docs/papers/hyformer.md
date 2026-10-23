---
icon: lucide/book-marked
---

# HyFormer：联合序列建模与特征交互

原文：[HyFormer: Revisiting the Roles of Sequence Modeling and Feature Interaction in CTR Prediction，arXiv:2601.12681v2](https://arxiv.org/abs/2601.12681v2)。本页为中文阅读笔记，项目实现见 [Baseline](../experiments/baseline.md)。

## 问题与机制

先压缩长历史、再与静态字段交互，会让两部分难以共同更新。HyFormer 将序列读取和特征交互组织在同一混合主干中，使 query、非序列特征与多域行为在层间交换信息。设计关注表示能力与长序列成本的平衡。

![HyFormer 架构，来源为论文归档图](img/hyformer/hyformerv3-1.png)

## 阅读时关注的三个边界

1. query 从哪些上下文生成，是否依赖候选物品。
2. 序列何时压缩、保留多少信息，跨域怎样交换。
3. token mixing 对维度、分组与预算有哪些约束。

这些选择共同决定结构，不能只复制 block 名称。论文的效率和质量结果应连同其数据、规模与评测协议阅读。

## 在本仓库怎样验证

当前 Baseline 使用 PCVR raw parquet、RankMixer NS token、模型默认值和共享训练器。它与原文具有任务、数据及训练配方差异，不是论文表格的完整复现。

可先固定训练配方，改变 query 数、序列预算或混合层数；每次同时测 AUC/logloss、参数量、显存与 step time。更换 tokenizer 时，分组数量与输出 token 数也可能变化，应作为独立变量记录。

结构验证需覆盖全空序列、padding、forward/backward 与 sidecar 重建。论文图说明设计，不支持当前实现的速度或榜单结论；方法见[测试](../guide/testing.md)和[性能测量](../guide/performance-benchmarks.md)。
