---
icon: lucide/notebook-pen
---

# Learning Mechanics：把训练过程变成证据

主要来源：[On neural scaling and the quanta hypothesis](https://learningmechanics.pub/quanta/) 与 [Deep linear networks are a surprisingly useful toy model of weight-space dynamics](https://learningmechanics.pub/deep-linear-nets/)。本文区分理论启发与本项目分析方案。

## 来源观点

Quanta hypothesis 尝试用多个局部学习过程解释总体平滑的缩放行为；深度线性网络用可分析的简化模型研究权重空间和逐阶段模式学习。toy model 给出机制线索，并不自动解释带非线性、稀疏特征和随机采样的 PCVR 模型。

## 先观察哪些量

| 层次 | 观测 | 要回答的问题 |
| --- | --- | --- |
| 整体 | loss、AUC、校准、梯度范数 | 是否在稳定学习 |
| 样本/分桶 | loss 轨迹、序列长度、频次、时间段 | 哪些样本先改善 |
| 表征 | 范数、有效秩、奇异值谱 | 表征是否退化或仅改变尺度 |
| 权重 | delta W、更新范数、谱 | 不同 optimizer 的更新如何不同 |
| 预算 | 参数、数据、step、时间 | 等成本下质量如何变化 |

有效秩降低本身不是失败，可能是有用的压缩；需同时看任务质量和对照。指标上的跳变也可能来自阈值或样本噪声。

## 当前可用入口

项目已有训练/评估 telemetry、逐样本预测、诊断图以及 `taac-analysis-learning-trace` CLI。Symbiosis 还记录部分 token 与表征诊断。这些能力不等于已实现完整 quanta 发现、梯度谱聚类或缩放律拟合。

```bash
uv run --with torch==2.13.0 taac-analysis-learning-trace --help
```

先通过 help 核对实际输入参数，按[PCVR 诊断](../../benchmark/pcvr-diagnostics.md)收集匹配产物，不对不存在的全量轨迹作占位分析。

## 最小研究设计

固定验证样本与切分，选择多个真实训练阶段的 checkpoint，记录相同样本的概率/loss。按时间、用户/物品频次与序列长度分桶，检查局部改善是否跨 seed 稳定，再考虑更昂贵的权重谱或单样本梯度。

梯度相似性分析可先限制到小样本和少量层，明确采样、投影与聚类随机性。不要在日志里保留可识别用户信息。

## 如何讨论 Scaling

分别改变参数 N、样本 D 和 step S，记录 `L(N,D,S)` 与实际时间/算力。拟合前固定数据来源、验证集和收敛口径，报告残差和不确定性；少数点或合成重复样本不足以建立长期缩放规律。

若大模型只在更多训练预算下获益，需要同时报告等 step 和等成本比较。理论标签不能替代受控实验证据。
