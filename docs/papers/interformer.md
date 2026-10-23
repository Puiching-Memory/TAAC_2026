---
icon: lucide/book-marked
---

# InterFormer：异构表示之间的双向交互

原文：[InterFormer: Effective Heterogeneous Interaction Learning for Click-Through Rate Prediction，arXiv:2411.09852v4](https://arxiv.org/abs/2411.09852v4)。本页解释方法动机，当前代码见[实验说明](../experiments/interformer.md)。

## 解决什么问题

静态画像和动态历史提供不同信息。过早汇总历史可能丢掉细节，而只让 query 读取序列会限制反向信息流。InterFormer 保留两类表示，并以交替方式交换信息；Cross Arch 控制选择性聚合，避免在早期一次性压缩全部信号。

![InterFormer 架构，来源为论文归档图](img/interformer/arch.png)

## 怎样阅读结构

分别追踪非序列表示、序列表示与跨表示交互，标出每一层谁提供 query、谁提供 key/value、何时压缩。双向交互并不意味着所有字段和所有事件始终完全互相可见。

论文报告的线上部署收益属于其工业环境；当前 PCVR 实验没有由此获得相同收益的证据。

## 本仓库的对照问题

实验包使用当前 schema、group tokenizer 和共享运行时。与 Baseline 比较时，先控制 token 预算、query 数、序列长度与训练配方，避免把 tokenizer 差异当作全部结构贡献。

可检查更深交互是否改善长序列/稀疏样本分桶、梯度是否到达所有分支、空域是否导致输出退化。若要判断“过早聚合”的代价，需保持参数与计算预算接近，再改变聚合位置。

模型存在、契约通过和部署效果是不同证据。当前实现范围见[实验页](../experiments/interformer.md)，运行方法见[快速开始](../getting-started.md)。
