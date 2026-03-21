# TAAC_2026

> [!INFO]  
> 这是TAAC其中一个参赛队伍的代码仓库, 不代表官方文档

https://algo.qq.com/#intro

## Introduction
**推向统一序列建模与特征交互的大规模推荐系统**

推荐系统为大规模内容平台（信息流、短视频等）和数字广告（点击率/转化率预测等）提供核心支撑，直接影响用户体验、用户参与度和平台收入。在巨大的流量压力和严格的延迟限制下，这些系统每天需要进行数十亿次实时决策，并支撑着价值数千亿美元的数字广告市场。在过去二十年间，推荐系统研究沿着两条主要路径发展：特征交互模型，专注于对高维多字段分类特征和上下文特征进行建模；以及序列模型，通过基于嵌入的检索系统和Transformer风格的排序模型来捕捉用户行为的时间动态。尽管这两种范式都取得了显著成功，但它们在很大程度上是独立发展的。这种分离在工业系统中造成了结构性瓶颈：跨范式交互浅层化、优化目标不一致、扩展性有限，以及硬件和工程复杂性不断增加。随着序列长度和模型规模的持续增长，这些碎片化的架构变得越来越低效。

近年来，一些研究开始尝试弥合这两个历史上分离的分支 [1–3]。为了进一步加速这一方向的进展，我们提出了“推向统一序列建模与特征交互的大规模推荐系统”挑战。我们鼓励参赛者开发一种统一的标记化方案和一个同质化、可堆叠的骨干网络，在单一架构内同时建模用户序列行为和非序列的多字段特征，用于转化率预测。提交结果将根据单一的ROC曲线下面积（AUC） 指标进行排名。除了排行榜外，我们还将设立两项创新奖——统一模块创新奖（45,000美元） 和缩放规律创新奖（45,000美元），分别表彰在统一架构和系统性缩放规律探索方面的突出进展。这些奖项与排行榜排名无关，而研讨会论文评审将着重于这两个方向的新颖性和洞察力，而非仅仅关注AUC指标。

## Timeline
Global Registration Mar.19 — Apr.23 23:59:59 AOE

## Eligibility
Academic Track

## Dataset&Task

https://huggingface.co/datasets/TAAC2026/data_sample_1000

本次比赛发布的数据集经过完全匿名化处理，不反映腾讯广告平台的实际生产特性。

我们的数据集是一个基于真实广告日志构建的大规模工业级数据集，包含两个主要组成部分：(1) 用户行为序列 和 (2) 非序列多字段特征。

用户行为序列 包含用户与物品之间的交互事件（如曝光、点击、转化），每个事件都附带时间戳和行为类型等附加信息。多字段特征 包括用户属性、物品属性、上下文信号以及交叉特征。

为确保公平性和保护隐私，所有稀疏特征均以匿名整数ID表示，稠密特征则以固定长度的浮点向量提供。不发布任何原始内容（如文本、图像、URL）或个人身份信息。

此外，我们提供了一些示例样本供参考：

当前示例样本以JSON格式提供，但正式比赛所用数据可能基于此初步版本进行调整，包括格式和实际内容的可能变更。

**Sequential Data (e.g. one user behavior sequence)**
```json
{"user_id": "1", "seq": [{"item_id": 16612, "action_type": 1, "timestamp": 1770564000}, {"item_id": 49638, "action_type": 1, "timestamp": 1770564000}, ..., {"item_id": 173346, "action_type": 3, "timestamp": 1766960100}, ..., {"item_id": 49495, "action_type": 2, "timestamp": 1766576760}, ..., {"item_id": 1753, "action_type": 4, "timestamp": 1766399880}], ...}
```

**User Features (e.g. one specific user)**
```json
[{"feature_id": 10, "feature_value_type": "int_array", "int_array": [1]},      // Marital Status
 {"feature_id": 8, "feature_value_type": "int_value", "int_value": 1},       // Gender
 {"feature_id": 7, "feature_value_type": "int_value", "int_value": 44}, ...] // Age
```

**Item Features (e.g. one specific item)**
```json
[{"feature_id": 70, "feature_value_type": "int_value", "int_value": 2},      // Type
 {"feature_id": 60, "feature_value_type": "int_value", "int_value": 3},      // Category
 {"feature_id": 72, "feature_value_type": "int_value", "int_value": 2}, ...] // Advertiser Type
```

**Context Features (e.g. one specific session)**
```json
[{"feature_id": 17, "feature_value_type": "int_value", "int_value": 3},      // Device Brand
 {"feature_id": 21, "feature_value_type": "int_value", "int_value": 3}, ...] // OS Type
```

**Cross Features**
```json
[{"feature_id": 25, "feature_value_type": "float_array", "float_array": [0.111, 0.057, 0.121, 0.043, -0.066, 0.081, 0.038, 0.105, -0.026, ...]}, ...] // User Embedding
```

## Our Work
TODO

## References
[1] InterFormer: Effective Heterogeneous Interaction Learning for Click-Through Rate Prediction. CIKM, 2025.  
[2] OneTrans: Unified Feature Interaction and Sequence Modeling with One Transformer in Industrial Recommender. arXiv preprint, 2025.  
[3] HyFormer: Revisiting the Roles of Sequence Modeling and Feature Interaction in CTR Prediction. arXiv preprint, 2026.  