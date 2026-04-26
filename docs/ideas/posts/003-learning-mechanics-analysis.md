---
icon: lucide/lightbulb
---

# Learning Mechanics 分析方法：从缩放律、学习量子到低秩动力学

:material-calendar: 2026-04-26 · :material-tag: 学习力学, Scaling Law, Quanta, 低秩更新, 谱分析

## 原文章出处

- **标题**：终于，学界找到了深度学习的「牛顿定律」
- **来源**：机器之心公众号
- **链接**：<https://mp.weixin.qq.com/s/v3XujOLco3fMJuEzQKer0w>
- **说明**：对论文 *There Will Be a Scientific Theory of Deep Learning* 的中文综述，提炼了 Learning Mechanics 的五条主线。

- **标题**：On neural scaling and the quanta hypothesis
- **作者**：Eric J. Michaud
- **链接**：<https://learningmechanics.pub/quanta/>
- **日期**：2026-04-23

- **标题**：Deep linear networks are a surprisingly useful toy model of weight-space dynamics
- **作者**：Mark Rhee, Dhruva Karkada, Jamie Simon
- **链接**：<https://learningmechanics.pub/deep-linear-nets/>
- **日期**：2026-04-23

<!-- more -->

## AI 解读

这组文章关心的不是再发明一个更复杂的模型，而是建立一套解释和诊断训练过程的方法。对 TAAC 2026 来说，价值在于把“模型为什么涨分、什么时候涨分、靠哪些样本或特征涨分”变成可观测对象，服务统一模块创新奖和 Scaling Law 创新奖。

### 方法 1：用可解 toy model 解释真实训练现象

Learning Mechanics 主张先找能精确分析的简化系统，再把其中的机制迁移回真实模型。深度线性网络是典型例子：虽然端到端函数仍是线性的，但参数空间优化是非凸的，能展示分阶段学习、低秩更新、鞍点到鞍点的轨迹和隐式低秩偏置。

可借鉴的分析动作：

- 对模型权重、embedding 表、MLP/attention 投影矩阵做 SVD，观察训练中奇异值是否按顺序增长。
- 记录每个 checkpoint 的权重增量 `delta W` 的有效秩、top singular value 占比、谱熵。
- 比较小初始化、默认初始化、不同深度下 loss 曲线是否更接近“台阶式”或“平滑式”。
- 把复杂模型中出现的低秩学习现象先在一个小型矩阵分解或线性化推荐任务上复现。

### 方法 2：用谱分解定位“先学什么、后学什么”

深度线性网络文章的核心分析路径是：写出梯度流方程，白化输入，旋转到输入输出协方差的 SVD 基，假设权重先快速对齐，再把矩阵动力学解耦成逐个奇异模式的标量动力学。结论是大奇异值模式更早学到，小奇异值模式更晚学到。

迁移到本项目时，不需要完整解析解，也可以做经验版谱分析：

- 对非序列特征、序列特征、用户侧 dense 特征、物品侧特征分别构造统计矩阵，估计主方向。
- 检查模型 embedding 或中间表征是否先对齐高方差、高频或高互信息方向。
- 在训练中跟踪不同特征组的梯度范数、激活范数、表示漂移量。
- 对比 baseline、hyformer、symbiosis 等实验包是否学习相同主方向，只是速度不同。

### 方法 3：把平滑缩放律拆成很多离散学习单元

Quanta hypothesis 试图解释一个张力：总体 loss 随参数、数据、step 呈平滑幂律下降，但很多能力在局部看像突然出现。它的解释是，模型学习了大量离散或近似离散的“学习量子”，每个量子只影响一小部分样本，整体 loss 把这些小相变平均掉了。

可借鉴的分析动作：

- 不只看整体 AUC/logloss，还记录 per-sample、per-user-segment、per-feature-group、per-sequence-domain 的学习曲线。
- 统计哪些样本在某个训练阶段 loss 突然下降，哪些样本始终缓慢改善，哪些样本出现 inverse scaling。
- 把样本按 loss 曲线形状聚类，识别“早学会”“中期学会”“晚学会”“不稳定”的样本族。
- 估计样本族频率是否长尾分布，并观察模型规模、训练步数、数据量增加时是否按频率顺序解锁。

### 方法 4：用梯度相似性发现潜在机制或样本簇

Quanta 文章中，一个具体实验是对模型预测正确且低 loss 的 token 计算 loss 关于参数的梯度，再用梯度余弦相似度和谱聚类寻找相似机制。直觉是：如果模型在不同样本上调用了类似机制，这些样本的梯度方向也会相似。

迁移到 PCVR 推荐任务，可以把 token 换成样本或样本中的目标 item：

- 从 validation 或 held-out batch 中抽样，计算单样本 loss gradient。
- 只保留最后几层、embedding projection 或特定模块的梯度，控制显存和计算量。
- 对梯度向量做随机投影或 PCA 降维，再计算 cosine similarity。
- 谱聚类后分析每个簇的 schema 特征、序列长度、行为域、label_type、item/user 高频程度。
- 输出簇级别 AUC/logloss 和训练阶段变化，判断模型是否存在“技能簇”。

### 方法 5：把 emergence 和 metric artifact 分开看

文章提醒，很多“涌现能力”可能是指标造成的视觉跳变，例如 accuracy 会把概率分布的微小变化变成 0/1 翻转。推荐任务中也有类似问题：AUC、top-k 命中、logloss、校准误差对模型变化的敏感性不同。

可借鉴的分析动作：

- 同时记录 AUC、logloss、分桶 calibration、top-k/rank proxy，避免只看单一指标。
- 对相同 checkpoint 输出概率分布，观察是整体置信度提高，还是正负样本排序真正拉开。
- 对看似“突然变好”的模型配置，检查 per-sample loss 是否早已有隐藏进展。
- 对关键实验跑多 seed，区分真实相变和随机种子噪声。

### 方法 6：研究参数、数据、step 的联合缩放

Quanta 文章讨论了参数缩放、数据缩放、step 缩放和联合缩放之间的差异，并指出真实模型里“大模型学习效率更高”会让简单的瓶颈模型失效。本项目正好有 Scaling Law 创新奖背景，适合把这条路线变成可复核实验。

可借鉴的分析动作：

- 以 `N` 表示模型参数量，`D` 表示训练样本量或数据比例，`S` 表示训练 step，记录 `L(N,D,S)`。
- 分别扫模型宽度、层数、embedding 维度、训练数据比例、训练步数。
- 拟合 `L = E + A N^-alpha_N`、`L = E + B D^-alpha_D`、`L = E + C S^-alpha_S`。
- 进一步比较 Chinchilla 风格可加形式和 Quanta 风格全局指数形式哪一个更贴近本赛题。
- 记录等 loss 曲线，观察增加模型规模是否能减少达到同等 loss 所需 step。

### 方法 7：比较不同架构的表征收敛

Learning Mechanics 的另一个主线是普适行为：不同架构和数据集可能学到相似表征。对本项目来说，可以用它判断多个实验包是不是在学同一类结构，只是表达方式不同。

可借鉴的分析动作：

- 在相同 validation slice 上抽取 baseline、symbiosis、hyformer、interformer 等模型的中间表征。
- 用 CKA、RSA、线性 probe 或 nearest-neighbor overlap 比较表示相似度。
- 对序列域、非序列域、用户 dense、物品特征分别比较表征收敛程度。
- 如果某个模型 AUC 高但表征与其他模型差异大，优先分析它学到的新增结构。

## 我们的看法

*（待补充）*

## 实施清单

- [ ] 先实现 `analysis trace`：从训练 checkpoint 和 validation slice 导出结构化轨迹。
- [ ] 再实现 `analysis spectra`：权重和权重增量 SVD，输出低秩指标。
- [ ] 再实现 `analysis scaling-fit`：读取多个 run summary，拟合简单幂律。
- [ ] 再实现 `analysis sample-curves`：固定样本跨 checkpoint 的 loss 曲线。
- [ ] 再实现 `analysis gradient-clusters`：小样本单样本梯度相似性聚类。
- [ ] 最后实现 `analysis representation-similarity`：跨实验包表征收敛分析。