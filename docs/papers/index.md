---
icon: lucide/book-open
---

# 搜广推技术发展时间线

本页梳理近年来搜索、广告、推荐（搜广推）领域的主要技术成果，帮助快速了解从经典方法到前沿架构的演进脉络。

## 技术演进关系图

> 实线表示同一技术分支内的演进，虚线表示跨分支影响，金色边框表示本仓库已实现的方法。
>
> 将鼠标悬停于节点可高亮其相关方法，拖拽画布可平移视图。

---

## 经典时代（2015 及以前）

| 年份 | 里程碑                      | 说明                                                    |
| ---- | --------------------------- | ------------------------------------------------------- |
| 2001 | Item-based CF               | Amazon 提出基于物品的协同过滤，奠定工业推荐基础         |
| 2007 | Matrix Factorization        | Netflix Prize 推动矩阵分解方法成为主流                  |
| 2010 | FM (Factorization Machines) | Rendle 提出因子分解机，统一多种特征交叉方法             |
| 2013 | DSSM                        | Huang et al. 提出深度结构化语义模型，将深度学习引入检索 |
| 2015 | GRU4Rec                     | 首次将 RNN 用于会话推荐（按 arXiv 预印本时间）          |

## 深度学习推荐崛起（2016–2019）

| 年份 | 里程碑                      | 说明                                                       |
| ---- | --------------------------- | ---------------------------------------------------------- |
| 2016 | YouTube DNN                 | Covington et al. 提出召回+排序两阶段深度架构，成为工业标准 |
| 2016 | Wide & Deep                 | Google 融合记忆与泛化能力                                  |
| 2017 | DeepFM                      | 将 FM 与 DNN 端到端结合                                    |
| 2018 | DIN (Deep Interest Network) | 阿里引入 Target Attention 机制建模用户兴趣                 |
| 2018 | SASRec                      | 将 Self-Attention 应用于序列推荐                           |
| 2019 | BERT4Rec                    | 借鉴 BERT 双向编码做序列推荐                               |
| 2019 | DIEN                        | 兴趣演化网络，用 AUGRU 捕捉动态兴趣                        |
| 2019 | DLRM                        | Meta 开源深度学习推荐模型，成为行业基准                    |

## 长序列与特征交叉（2020–2023）

| 年份 | 里程碑     | 说明                                     |
| ---- | ---------- | ---------------------------------------- |
| 2020 | MIMN / SIM | 长序列兴趣建模，突破用户行为序列长度限制 |
| 2021 | DCNv2      | Google 改进交叉网络，提升特征交叉效率    |
| 2022 | DHEN       | Meta 提出异构专家网络                    |
| 2023 | ETA / SDIM | 基于哈希的长序列高效检索方案             |
| 2024 | Wukong     | 字节提出大规模排序模型                   |
| 2024 | LONGER     | 字节的长序列 Transformer 压缩方案        |
| 2024 | RankMixer  | Token-Mixing 特征交叉架构                |

## PCVR 建模：样本偏差与延迟反馈（2018–2026）

Post-Click Conversion Rate（PCVR）预估是广告与电商推荐的核心任务，但在实践中面临两大特有挑战：**样本选择偏差**（Sample Selection Bias, SSB）和**数据稀疏**（Data Sparsity, DS）。CVR 训练样本仅来自点击后子集，而推理需覆盖全部曝光空间；转化行为远少于点击行为，正样本极度稀缺。此外，**延迟反馈**（Delayed Feedback）使转化标签随时间逐步到达，加剧了建模难度。

### 核心挑战

| 挑战               | 说明                                                             |
| ------------------ | ---------------------------------------------------------------- |
| 样本选择偏差 (SSB) | CVR 仅在点击空间训练，但需在曝光空间推理，训练/推理分布不一致    |
| 数据稀疏 (DS)      | 转化行为极为稀少（通常 <1% 点击率），正样本严重不足              |
| 延迟反馈 (DF)      | 转化可能发生在点击数小时甚至数天后，训练时标签不完整             |
| 行为路径异构       | 用户从曝光→点击→转化之间经历多步中间行为，如何利用中间信号是关键 |

### 全空间多任务建模演进

从 ESMM 开创全空间建模范式以来，后续工作沿"行为路径分解 → 因果推断 → 表示增强"三个方向不断演进。

> 下表按时间顺序梳理 PCVR 全空间多任务建模的关键里程碑。

| 年份 | 里程碑                    | 说明                                                                                                      | 方法类型     |
| ---- | ------------------------- | --------------------------------------------------------------------------------------------------------- | ------------ |
| 2018 | **ESMM**                  | 阿里提出全空间多任务模型，利用 CTR×CVR=CTCVR 恒等式消除 SSB，特征表示迁移缓解 DS (SIGIR 2018)             | 全空间多任务 |
| 2019 | **ESM²**                  | 阿里提出点击后行为分解，在 click→conversion 之间插入 DAction/OAction 并行路径，进一步缓解 DS (SIGIR 2020) | 行为路径分解 |
| 2020 | **NCS4CVR**               | 腾讯提出神经元连接共享，通过结构共享将 CTR 网络知识迁移到 CVR 网络                                        | 结构共享     |
| 2020 | **DESMM**                 | 在 ESMM 基础上引入延迟反馈建模，处理转化标签延迟到达问题                                                  | 延迟反馈     |
| 2021 | **HM³**                   | 阿里提出微观/宏观行为分层建模，利用详情页组件级交互补充细粒度 CVR 信号 (SIGIR 2021)                       | 行为路径分解 |
| 2021 | **Follow the Prophet**    | 提出基于 Prophet 的在线延迟反馈校正方法 (SIGIR 2021)                                                      | 延迟反馈     |
| 2022 | **MSEN**                  | 多尺度用户行为网络，捕获不同时间粒度的行为信号用于全空间多任务学习 (CIKM 2022)                            | 行为路径分解 |
| 2023 | **DCMT**                  | 提出直接全空间因果多任务框架，反事实机制消除选择偏差并解决 NMAR 问题 (ICDE 2023)                          | 因果推断     |
| 2023 | **ESMC**                  | ESM² 的参数约束改进版，通过参数约束增强全空间 CVR 建模的稳定性                                            | 全空间多任务 |
| 2023 | **CST**                   | 点击感知结构迁移 + 样本权重分配，利用点击侧结构信息辅助 CVR 估计                                          | 结构迁移     |
| 2025 | **ChorusCVR**             | 快手提出合唱监督，区分模糊负样本与事实负样本，实现去偏全空间 CVR 学习                                     | 全空间多任务 |
| 2025 | **Counterfactual CVR**    | 基于反事实推断的点击后转化率预估，消除选择偏差 (ICDM 2025)                                                | 因果推断     |
| 2026 | **EKTM**                  | 华为提出多任务推荐有效知识迁移，Router + Transmitter + Enhancer 跨任务传递 CVR 知识，eCPM +3.93%          | 知识迁移     |
| 2026 | **RankUp**                | 腾讯提出高秩表示架构，通过随机置换分割 + 多嵌入 + 全局 token + 预训练嵌入交叉 + 任务 token 解耦提升 CVR   | 表示增强     |
| 2026 | **Counterfactual MTL-DF** | 反事实多任务学习应对电商大促延迟转化建模 (SIGIR 2026)                                                     | 因果+延迟    |

### RankUp：PCVR 表示增强的最新进展

RankUp（Chen et al., 2026, arXiv 2604.17878）是当前 PCVR 表示增强方向最具代表性的工作，已部署于微信视频号、公众号和朋友圈广告，GMV 分别提升 3.41%、4.81% 和 2.21%。其核心发现是：**MetaFormer 架构中 token 表示的有效秩随深度呈阻尼振荡轨迹，深层无法单调增长甚至退化**——即参数增长 ≠ 表示容量增长。为此提出五项机制：

1. **随机置换分割 (RPS)** — 相比语义分组减少 token 间相关性和共线性
2. **多嵌入表示 (ME)** — 扩展潜在空间基础自由度
3. **全局 token 集成 (GTI)** — token mixing 时交互全局上下文
4. **预训练嵌入交叉 (CPE)** — 引入外部领域/场景知识丰富潜在空间
5. **任务 token 解耦 (TSD)** — 缓解多目标梯度干扰

### PCVR 开放挑战

1. **统一架构下的 PCVR**：当前竞赛（TAAC 2026）以单一 Transformer 主干统一序列建模与特征交互完成 PCVR，如何在此框架下引入全空间/因果/延迟反馈机制是开放问题
2. **表示坍缩与有效秩**：深层推荐模型表示趋向低秩子空间集中，限制用户/物品区分能力，需更系统的高秩保持机制
3. **跨任务知识迁移**：CTR→CVR 的知识迁移仍是工业界关键需求，如何在 MoE / 多任务框架下高效迁移有待探索
4. **延迟反馈在线学习**：大促等场景下延迟转化建模的实时性要求与 ESMM 类全空间方法的兼容

---

## 统一建模与 Scaling Law（2024–2026）

这一阶段的核心趋势是将序列建模与特征交叉统一到单一 Transformer 主干中，并验证推荐系统中的 Scaling Law。

> 注：本表年份与仓库内技术图谱及 Semantic Scholar 缓存使用的 year 字段保持一致，以避免与图中 x 轴年份冲突。

| 年份 | 里程碑                | 说明                                                                                           | 本仓库                                                           |
| ---- | --------------------- | ---------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| 2024 | **HSTU / GR**         | Meta 提出 Hierarchical Sequential Transducer Units，1.5 万亿参数生成式推荐器，验证 Scaling Law | —                                                                |
| 2024 | **InterFormer**       | UIUC & Meta 提出交错式异构交互学习，双向信息流 + Cross Arch                                    | [论文](interformer.md) · [实验包](../experiments/interformer.md) |
| 2025 | **OneTrans**          | NTU & 字节提出单 Transformer 主干 + 统一 Tokenizer + KV 缓存                                   | [论文](onetrans.md) · [实验包](../experiments/onetrans.md)       |
| 2025 | **GPSD**              | 阿里巴巴提出生成式预训练初始化判别式推荐，13K→0.3B 参数遵循 Power Law (KDD'25)                 | —                                                                |
| 2025 | **Foundation-Expert** | Meta 提出 Foundation-Expert 范式，首次大规模部署推荐基础模型，日服务数百亿请求                 | —                                                                |
| 2025 | **HoMer**             | 美团提出统一 Encoder-Decoder，序列 + 集合建模消除三重异构性，节省 27% GPU                      | —                                                                |
| 2025 | **MTmixAtt**          | 美团提出 MoE + AutoToken + Multi-Mix Attention，缩放至 1B 参数实现跨场景统一排序               | —                                                                |
| 2026 | **HyFormer**          | 字节提出混合 Transformer，统一长序列建模与特征交叉                                             | [论文](hyformer.md) · [Baseline](../experiments/baseline.md)     |

## 生成式推荐（2023–2026）

生成式推荐将推荐任务从"检索+排序"转变为"序列生成"，利用 LLM 的自回归能力预测下一个物品。

### 关键思路

- **语义 ID**：用有语义含义的 token 序列代替传统物品 ID（如 IDGenRec），让模型理解物品本质
- **RAG 架构**：轻量召回 + LLM 精排，兼顾效率与质量（LlamaRec、PALR）
- **多模态生成**：同时处理文本、图片、视频的统一推荐（UniMP, MMGRec, Molar）
- **可控生成**：Speculative Decoding 加速推理（SpecGR），指令微调实现约束推荐

### 代表工作

| 年份 | 工作                  | 说明                                                                                               |
| ---- | --------------------- | -------------------------------------------------------------------------------------------------- |
| 2024 | GenRec                | 基于 LLM 的序列推荐，Masked Item Prediction                                                        |
| 2024 | IDGenRec              | 训练 ID 生成器将元数据转化为语义 ID                                                                |
| 2024 | RecGPT                | 生成式预训练文本推荐 (ACL 2024)                                                                    |
| 2024 | UniMP                 | 统一多模态个性化框架 (ICLR 2024)                                                                   |
| 2024 | MMGRec                | 多模态生成推荐 + 层级量化 (CIKM 2024)                                                              |
| 2024 | SpecGR                | 基于 Speculative Decoding 的归纳式生成推荐                                                         |
| 2024 | Molar                 | 多模态 LLM + 协同过滤对齐                                                                          |
| 2025 | SessionRec            | 下一会话预测范式 (NSPP)，解决传统 NIPP 与真实用户行为的不一致                                      |
| 2025 | COBRA                 | 级联稀疏-稠密表示的统一生成推荐，2 亿日活广告平台部署 (腾讯)                                       |
| 2025 | LLaDA-Rec             | 离散扩散替代自回归，并行生成语义 ID                                                                |
| 2025 | RecGPT-V2             | 层级多 Agent + 约束 RL，淘宝部署 CTR +2.98% (阿里)                                                 |
| 2025 | xGR                   | 面向生成式推荐的高效服务系统，3.49x 吞吐提升                                                       |
| 2025 | OxygenREC             | 快慢思考 + 指令跟随生成推荐，统一多场景训练一次全场景部署                                          |
| 2026 | TencentGR / TAAC 2025 | 腾讯广告算法大赛官方论文，发布全模态广告生成式推荐数据集、基线与 Top 方案总结；[论文](taac2025.md) |
| 2026 | HiGR                  | 层级规划 + 多目标偏好对齐的生成式 Slate 推荐                                                       |

### 开放挑战

1. **推理成本**：LLM 解码多 token 延迟高，需量化/蒸馏/Speculative Decoding 加速
2. **评估标准**：传统 NDCG/HR 不够，需多样性、新颖性、可解释性等新维度
3. **长尾推荐**：热门物品易推、长尾物品难推，需多样性奖励和重采样
4. **冷启动**：语义 ID 理论上缓解冷启动，但实际效果仍需验证

---

## 参考文献

- Zhai et al. *Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations.* ICML 2024
- Zeng et al. *InterFormer: Effective Heterogeneous Interaction Learning for CTR Prediction.* arXiv 2411.09852
- Huang et al. *HyFormer: Revisiting the Roles of Sequence Modeling and Feature Interaction in CTR Prediction.* arXiv 2601.12681
- Zhang et al. *OneTrans: Unified Feature Interaction and Sequence Modeling with One Transformer.* WWW 2026
- Wang et al. *Scaling Transformers for Discriminative Recommendation via Generative Pretraining.* KDD 2025. arXiv 2506.03699
- Li et al. *Realizing Scaling Laws in Recommender Systems: A Foundation-Expert Paradigm for Hyperscale Model Deployment.* arXiv 2508.02929
- Chen et al. *HoMer: Addressing Heterogeneities by Modeling Sequential and Set-wise Contexts for CTR Prediction.* arXiv 2510.11100
- Qi et al. *MTmixAtt: Integrating Mixture-of-Experts with Multi-Mix Attention for Large-Scale Recommendation.* arXiv 2510.15286
- Cao & Lio. *GenRec: Generative Sequential Recommendation with Large Language Models.* ECIR 2024
- Ji et al. *IDGenRec: LLM-RecSys Alignment with Textual ID Learning.* 2024
- Liu et al. *Multi-Behavior Generative Recommendation.* CIKM 2024
- Wei et al. *UniMP: Towards Unified Multi-modal Personalization.* ICLR 2024
- Zhao et al. *Recommender Systems in the Era of Large Language Models.* IEEE TKDE 2024
- Huang et al. *SessionRec: Next Session Prediction Paradigm for Generative Sequential Recommendation.* arXiv 2502.10157
- Yang et al. *Sparse Meets Dense: Unified Generative Recommendations with Cascaded Sparse-Dense Representations.* arXiv 2503.02453
- Shi et al. *LLaDA-Rec: Discrete Diffusion for Parallel Semantic ID Generation in Generative Recommendation.* arXiv 2511.06254
- Yi et al. *RecGPT-V2 Technical Report.* arXiv 2512.14503
- Sun et al. *xGR: Efficient Generative Recommendation Serving at Scale.* arXiv 2512.11529
- Hao et al. *OxygenREC: An Instruction-Following Generative Framework for E-commerce Recommendation.* arXiv 2512.22386
- Pang et al. *HiGR: Efficient Generative Slate Recommendation via Hierarchical Planning and Multi-Objective Preference Alignment.* arXiv 2512.24787
- Pan et al. *The Tencent Advertising Algorithm Challenge 2025: All-Modality Generative Recommendation.* arXiv 2604.04976

### PCVR 参考文献

- Ma et al. *Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate.* SIGIR 2018. arXiv 1804.07931
- Wen et al. *Entire Space Multi-Task Modeling via Post-Click Behavior Decomposition for Conversion Rate Prediction.* SIGIR 2020. arXiv 1910.07099
- Xiao et al. *NCS4CVR: Neuron-Connection Sharing for Multi-Task Learning in Video Conversion Rate Prediction.* 2020. arXiv 2008.09872
- Wang et al. *Delayed Feedback Modeling for the Entire Space Conversion Rate Prediction.* 2020. arXiv 2011.11826
- Wen et al. *Hierarchically Modeling Micro and Macro Behaviors via Multi-Task Learning for Conversion Rate Prediction.* SIGIR 2021. arXiv 2104.09713
- Li et al. *Follow the Prophet: Accurate Online Conversion Rate Prediction in the Face of Delayed Feedback.* SIGIR 2021. arXiv 2108.06167
- Jin et al. *Multi-Scale User Behavior Network for Entire Space Multi-Task Learning.* CIKM 2022. arXiv 2208.01889
- Zhu et al. *DCMT: A Direct Entire-Space Causal Multi-Task Framework for Post-Click Conversion Estimation.* ICDE 2023. arXiv 2302.06141
- Jiang et al. *ESMC: Entire Space Multi-Task Model for Post-Click Conversion Rate via Parameter Constraint.* 2023. arXiv 2307.09193
- Ouyang et al. *Click-aware Structure Transfer with Sample Weight Assignment for Post-Click Conversion Rate Estimation.* 2023. arXiv 2304.01169
- Cheng et al. *ChorusCVR: Chorus Supervision for Entire Space Post-Click Conversion Rate Modeling.* 2025. arXiv 2502.08277
- Ahn & Lee. *On Predicting Post-Click Conversion Rate via Counterfactual Inference.* ICDM 2025. arXiv 2510.04816
- Jia et al. *No One Left Behind: How to Exploit the Incomplete and Skewed Multi-Label Data for Conversion Rate Prediction.* 2025. arXiv 2512.13300
- Cai et al. *Effective Knowledge Transfer for Multi-Task Recommendation Models.* 2026. arXiv 2605.05730
- Chen et al. *RankUp: Towards High-rank Representations for Large Scale Advertising Recommender Systems.* 2026. arXiv 2604.17878
- Song et al. *Counterfactual Multi-task Learning for Delayed Conversion Modeling in E-commerce Sales Pre-Promotion.* SIGIR 2026. arXiv 2604.21675
