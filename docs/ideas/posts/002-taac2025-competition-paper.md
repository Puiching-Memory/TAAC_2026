---
icon: lucide/lightbulb
---

# 腾讯广告算法大赛 2025 官方论文解读：全模态生成式推荐

:material-calendar: 2026-04-14 · :material-tag: 生成式推荐, 多模态, Semantic ID, InfoNCE, Scaling Law

## 原文章出处

- **标题**：The Tencent Advertising Algorithm Challenge 2025: All-Modality Generative Recommendation
- **作者**：Junwei Pan, Wei Xue, Chao Zhou 等（腾讯 + 港中文）
- **链接**：<https://arxiv.org/abs/2604.04976>
- **日期**：2026-04-04
- **背景**：2025 届腾讯广告算法大赛官方赛题论文，系统介绍了 TencentGR-1M / TencentGR-10M 数据集构建、Baseline 模型、评估协议及 Top 方案总结。该赛事吸引了超过 8,440 名参赛者，冠军奖金 200 万 RMB。

<!-- more -->

## AI 解读

### 赛题定义

核心任务是 **多模态广告交互序列上的 next-item 预测**。每个用户有一条时间有序的行为序列 $S_u = \{x_u, x_{u,1}, \ldots, x_{u,T_u}\}$，其中：

- $x_u$：用户画像 token（静态特征）
- $x_{u,t}$：广告交互 token = 分类特征 + 行为类型 $f_\text{act}$ + 多模态嵌入 $f_\text{mm}$

目标是从大规模候选池（初赛 66 万、决赛 364 万）中检索出用户最可能交互的下一个广告。

### 两轮赛制差异

| 维度       | 初赛 TencentGR-1M                               | 决赛 TencentGR-10M                                    |
| ---------- | ----------------------------------------------- | ----------------------------------------------------- |
| 用户数     | 100 万                                          | 1,014 万                                              |
| 广告数     | 478 万                                          | 1,749 万                                              |
| 行为类型   | 曝光 + 点击                                     | 曝光 + 点击 + **转化**                                |
| 候选池     | 66 万                                           | 364 万                                                |
| 评估       | 标准 HR@10 + NDCG@10                            | **加权** w-HR@10 + w-NDCG@10（转化权重 $\alpha=2.5$） |
| 排行榜分数 | $0.31 \cdot \text{HR} + 0.69 \cdot \text{NDCG}$ | 同结构，加权指标                                      |

关键变化：决赛将**转化事件**引入序列和预测目标，且评估指标对转化赋予 2.5 倍权重。

### 多模态特征

论文提供 6 种预提取嵌入（非原始素材）：

| Emb ID | 模型                               | 模态 | 维度      |
| ------ | ---------------------------------- | ---- | --------- |
| 81     | Bert-finetune（协同对比学习微调）  | 文本 | 32        |
| 82     | Conan-embedding-v1                 | 文本 | 1,024     |
| 83     | gte-Qwen2-7B-instruct              | 文本 | 3,584     |
| 84     | Hunyuan-mm-7B-finetune（协同微调） | 图像 | 4,096→323 |
| 85     | QQMM-embed-v1                      | 图像 | 3,584     |
| 86     | UniME-LLaVA-OneVision-7B           | 图像 | 3,584     |

注意 ID 81 和 84 经过**协同数据对比学习微调**，融合了协同信号，其余为原始预训练模型直出。

### Baseline 模型架构

官方 Baseline 采用经典的 **因果 Transformer + InfoNCE + ANN 检索** 范式：

1. **特征编码**：稀疏 ID → embedding lookup → concat → MLP 投影；多模态嵌入直接拼接
2. **Backbone**：1 层 Transformer（$d=32$，1 头，dropout=0.2），因果 mask
3. **训练目标**：InfoNCE loss，每正样本采 1 个负样本
4. **推理**：用户序列 → Transformer → 末位 hidden state 作 user embedding → Faiss ANN 检索 top-K

配置极简（单层、$d=32$），有巨大的扩展空间。

### Top 方案核心思路

#### 🥇 第一名：Qwen Backbone + Action Conditioning + Semantic ID

- **骨干**：基于 dense Qwen 的多模态自回归模型
- **Action Conditioning**：逐位置行为调制——gated fusion + FiLM + attention bias，使模型区分曝光/点击/转化语义
- **时间特征工程**：绝对时间戳 + 相对间隔 + 会话结构 + 多频 Fourier 编码
- **Semantic ID**：对多模态嵌入做 RQ-KMeans 残差量化生成语义 ID + random-$k$ 正则化
- **优化器**：Muon + AdamW 混合；GPU 友好的静态 shape InfoNCE + 大规模负样本池
- **推理**：端到端生成 user vector → ANN 检索

#### 🥈 第二名：Encoder-Decoder + GNN + Semantic ID

- **Encoder**：多 gated MLP 分别编码用户 / 物品 / 交互序列 + GNN 在用户-物品交互图上做邻域聚合
- **Decoder**：改进 SASRec Transformer（$d=2048$，8 层，8 头）生成 "next embedding"
- **Semantic ID**：SVD + RQ-KMeans
- **Action Conditioning**：参考 PinRec，编码下一步行为类型做条件生成
- **两阶段训练**：先在曝光数据预训练，再在点击/转化数据微调
- **后处理**：过滤已交互物品

#### 🥉 第三名：Decoder-only Transformer + Scaling Law 研究

- **架构**：Decoder-only Transformer + PinRec 风格 next action type conditioning
- **训练**：InfoNCE + AMP 混合精度 + 静态图编译
- **Scaling Law 系统研究**：
    - 负样本数扩至 **380K**，性能持续增长
    - 模型容量（深度 × 宽度）与 ID embedding 维度的扩展规律
- **核心结论**：对生成式推荐，**规模比精巧设计更重要**

#### 🏆 技术创新奖：联合检索+排序的生成模型

- **联合建模**：同一模型同时生成下一个 item 的 Semantic ID 和预测 action type
- **训练目标**：Semantic ID generation loss + action prediction loss 联合优化
- **Semantic ID 创新**：
    - 专用 Decoder-only Transformer + InfoNCE 提取协同嵌入
    - 二级码碰撞解决机制（自动搜索最近簇中心替代）
- **架构组件**：FlashAttention + SwiGLU + RMSNorm + RoPE + DeepSeek-V3 MoE
- **特征**：稀疏特征 + 多模态 + 多时间窗口 item 热度统计 + 时间特征
- **优化**：混合精度、稀疏/稠密分优化器、grouped GEMM、KV cache 加速

### 跨方案共性总结

| 技术点                             | 冠军  | 亚军  |   季军    | 创新奖 |
| ---------------------------------- | :---: | :---: | :-------: | :----: |
| Causal Transformer                 |   ✓   |   ✓   |     ✓     |   ✓    |
| InfoNCE loss                       |   ✓   |   ✓   |     ✓     |   ✓    |
| Action Conditioning（PinRec 风格） |   ✓   |   ✓   |     ✓     |   ✓    |
| RQ-KMeans Semantic ID              |   ✓   |   ✓   |     -     |   ✓    |
| 大规模负样本                       |   ✓   |   -   | ✓（380K） |   -    |
| 多模态嵌入利用                     |   ✓   |   ✓   |     -     |   ✓    |
| 时间特征工程                       |   ✓   |   -   |     ✓     |   ✓    |
| MoE                                |   -   |   -   |     -     |   ✓    |
| GNN                                |   -   |   ✓   |     -     |   -    |

**4/4 方案均使用**：因果 Transformer + InfoNCE + Action Conditioning。这三者构成了赛题的"基本盘"。

### 对 TAAC 2026 的适用性分析

**1. Action Conditioning 是必选项**

所有 Top 方案均引入了行为类型条件化机制（参考 PinRec）。当前实验包以单一 `label_type` 做二分类（BCE loss），没有在序列建模层面区分不同行为语义。如果 2026 届赛题继续包含多行为类型，需要优先在 token 表示中编码 action type，并在生成阶段做条件化。

**2. Semantic ID 是多模态利用的主流路径**

Top 3 中有 3 支队伍使用了 RQ-KMeans 将多模态嵌入量化为离散 Semantic ID。这比直接拼接连续嵌入更适配生成式范式（离散 token 空间）。当前实验包直接使用连续嵌入，后续应探索 RQ-KMeans / RQ-VAE 离散化方案。

**3. 规模扩展 > 精巧设计（在一定范围内）**

第三名的 Scaling Law 实验表明，负样本数扩至 380K 时性能仍在增长。这与传统判别式推荐的调参思路不同——在生成式推荐中，加大负样本库、增大模型容量可能是性价比最高的提分手段。当前 baseline 只采 1 个负样本，有巨大空间。

**4. InfoNCE 是实质上的标准 loss**

所有 Top 方案都使用了 InfoNCE（对比学习 loss）而非 BCE / BPR。这与检索式评估（HR@K / NDCG@K）天然对齐。当前部分实验包使用 BCE loss 做点击率预估，需要考虑切换到 InfoNCE + ANN 检索范式。

**5. 赛题评估方式决定了模型范式**

HR@10 / NDCG@10 本质是 retrieval 指标，要求模型从数十万候选中精准检索。这与工业界的 CTR 排序模型（逐样本打分）有本质区别。模型必须产出可做 ANN 检索的 dense embedding，而非逐对 logit。

### 与现有实验包的差距映射

| 论文技术              | 当前实验包对应                       | 差距                                          |
| --------------------- | ------------------------------------ | --------------------------------------------- |
| 因果 Transformer      | baseline / onetrans / hyformer / interformer | 已有，但层数和维度远小于 Top 方案      |
| InfoNCE loss          | 无                                   | **当前使用 BCE，需新增 InfoNCE + 负采样**     |
| Action Conditioning   | 无                                   | **当前 action_type 仅作 label，未编码入序列** |
| RQ-KMeans Semantic ID | 无                                   | **需新建离散化模块**                          |
| 大规模负样本          | 无                                   | 当前 BCE 逐样本，无负样本池概念               |
| ANN 检索推理          | 无                                   | 当前逐样本 logit，需切换到 embedding + Faiss  |
| 时间特征工程          | 部分（timestamp 列存在但未充分利用） | 需增加相对间隔、Fourier 编码等                |

### 风险与局限

- 论文是 2025 届赛题总结，2026 届数据 schema 已发生变化（flat column layout, 4 domain 序列），具体技术需适配新格式
- 冠军方案使用 Qwen 骨干（参数量大），在 2026 届资源约束下能否适用需评估
- GNN 方案（亚军）在离线 batch 评估中有效，但构建交互图开销大，需权衡收益
- RQ-KMeans Semantic ID 需要额外的离线量化流程和码本维护

## 我们的看法

*（待补充）*

## 实施清单

- [ ] **InfoNCE loss + 负采样**：在训练损失模块中实现 InfoNCE（温度参数 + in-batch negatives），替代 BCE 作为核心训练目标
- [ ] **Action Conditioning**：在 token embedding 层引入 action type embedding（gated fusion 或 FiLM），实现 PinRec 风格的行为条件化生成
- [ ] **RQ-KMeans Semantic ID**：实现残差量化 k-means 模块，将多模态嵌入离散化为语义 ID 序列，作为新的 item 表示方案
- [ ] **扩大负样本规模**：实现全局负样本池 / cross-batch memory bank，逐步对标第三名 380K 负样本规模
- [ ] **ANN 检索推理**：在评估阶段集成 Faiss ANN 索引，从逐样本打分切换到 embedding 检索范式
- [ ] **时间特征增强**：在 data pipeline 中计算相对时间间隔、会话边界，在 embedding 层加多频率 Fourier 时间编码
- [ ] **模型 Scaling 实验**：系统测试 Transformer 层数 / 隐藏维度 / embedding 维度的 scaling 效果，建立本赛题的 scaling 参考曲线
