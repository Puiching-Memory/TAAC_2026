---
icon: lucide/book-open
---

# 论文阅读：从问题到实现

本分区以中文笔记介绍论文要解决的问题、关键机制及与当前实验的关系。原文是理论和实验结果的权威来源；本仓库实现和测量范围在独立实验页说明。

## 按研究问题阅读

| 问题 | 材料 | 仓库关联 |
| --- | --- | --- |
| 序列建模与特征交互怎样共同演进 | [HyFormer](hyformer.md) | [Baseline](../experiments/baseline.md) |
| 如何避免静态与序列只有单向交互 | [InterFormer](interformer.md) | [InterFormer 实验](../experiments/interformer.md) |
| 是否能用一个主干统一特征与行为 | [OneTrans](onetrans.md) | [OneTrans 实验](../experiments/onetrans.md) |
| 长上下文如何控制计算与内存 | [DeepSeek-V4](deepseek-v4.md) | 系统设计参考，无完整模型复现 |
| 生成式召回与点式预测有何不同 | [TAAC 2025](taac2025.md) | 任务与方法对照，无完整召回流水线 |

## 阅读顺序

先确认任务是候选评分还是大候选池召回，再看 tokenization、信息交换、序列预算、损失与评估。若直接从架构名称推断能力，容易把数据特征、训练预算和评测方式差异忽略。

HyFormer、InterFormer 和 OneTrans 分别提供融合方式的不同参照，它们不构成严格的单线继承关系。Symbiosis、TokenFormer 是仓库实验组合，QueryFormer 来源为公开方案说明；其身份与差异见[实验索引](../experiments/index.md)。

## 用论文指导实验

记录原文版本、输入、训练目标、关键结构与对照预算，再提出一个可证伪假设。接入代码只说明该结构可执行；测试、GPU 测量和独立验证分别建立不同证据。

本文档保留已有论文图片资源和来源链接，正文是阅读笔记，不再作为逐页转换的论文全文。需要公式证明、完整结果表与作者列表时查看原文。研究设想另见[笔记区](../ideas/index.md)。
