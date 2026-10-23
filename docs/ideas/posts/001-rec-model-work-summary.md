---
icon: lucide/notebook-pen
---

# 特征 Token 化、长历史与辅助 Loss：实验拆解

阅读来源：[直播推荐实战文章](https://zhuanlan.zhihu.com/p/2025299319658493425)。本文保留其主题启发，重点整理本项目可执行的研究问题；来源中的业务效果未在当前 PCVR 数据上复现。

## 假设一：字段边界影响信息保留

group tokenizer、固定预算 RankMixer 和 field-level token 在压缩程度与交互成本上不同。先固定模型、训练和切分，比较 tokenization，再记录实际 token 数与参数量。若只增加 token 数，质量变化不能全部归因于“更合理的分组”。

项目参照为 Baseline、InterFormer 与 TokenFormer，但三者默认结构和训练配方不同，不能直接把默认结果当作 tokenizer 消融。

## 假设二：长历史收益取决于选择方式

比较 tail、recent+memory、候选感知读取与完整历史，需要同时固定原始可见范围和进入 backbone 的预算。按长/短历史、空域与数据时间分桶，检查额外成本是否集中帮助某类样本。

先用[数据管道](../../guide/pcvr-data-pipeline.md)确认时间过滤与切分，避免未来事件或重叠用户导致乐观结果。

## 假设三：辅助目标带来不同学习信号

BCE 主要优化概率，pairwise 目标关注正负排序。DualQ 当前提供 BCE+pairwise AUC 配方；这不证明在其他模型上也会涨分。引入额外 loss 时记录权重、有效 pair 数、空 pair batch 和梯度规模，避免单纯扩大梯度被误认为目标贡献。

## 最小可复核实验

1. 固定一个模型、一份独立验证数据和多个 seed，保存 BCE 基线。
2. 分别改变 tokenizer、序列选择、loss，其他参数保持一致。
3. 同时报告 AUC/logloss、分桶质量、峰值内存、吞吐和训练时长。
4. 若收益只在单 seed 或微小 demo 上出现，继续复核，不发布泛化结论。

已有入口见[实验索引](../../experiments/index.md)，性能与预测诊断见[Benchmark](../../benchmark/index.md)。
