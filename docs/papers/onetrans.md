---
icon: lucide/book-marked
---

# OneTrans：一个主干同时处理特征与序列

原文：[OneTrans: Unified Feature Interaction and Sequence Modeling with One Transformer in Industrial Recommender，arXiv:2510.26104v3](https://arxiv.org/abs/2510.26104v3)。当前实验是 OneTrans-inspired 实现，参见[实验页](../experiments/onetrans.md)。

## 核心问题

分别扩展特征交互网络和行为序列网络，难以获得统一的信息交换与计算预算。OneTrans 把静态特征和序列映射为 token，在同一 Transformer 主干中更新，并以金字塔式机制减少深层 token 成本。

![OneTrans 架构，来源为论文归档图](img/onetrans/onetrans_all-1.png)

## 统一之后仍有哪些选择

统一 token 形状不等于字段语义相同。阅读时仍需核对输入投影、位置与 mask、token 保留策略、读出和训练目标。缩短序列可省计算，也可能损失候选相关历史。

## 在项目中检验什么

当前 PCVR 包共享 parquet、配置和 checkpoint 机制，没有实现论文全部工业训练和系统条件。比较时记录每层实际 token 数，而不只记录原始序列上限。

一个可执行的实验是固定参数与数据预算，对比金字塔收缩、固定长度和滑动窗口；同时检查长历史分桶质量、显存、吞吐与推理延迟。与 [TokenFormer](../experiments/tokenformer.md) 比较时，位置编码、层数与增强也需要固定。

论文中的 scaling 观察不能由单个宽度或几步 smoke 重建。要讨论趋势，应跨多个预算、重复种子，并明确优化与数据是否成为新瓶颈。
