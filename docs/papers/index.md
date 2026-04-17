# 搜广推技术发展时间线

本页梳理近年来搜索、广告、推荐（搜广推）领域的主要技术成果，帮助快速了解从经典方法到前沿架构的演进脉络。

## 技术演进关系图

> 实线表示同一技术分支内的演进，虚线表示跨分支影响，金色边框表示本仓库已实现的方法。
>
> 将鼠标悬停于节点可高亮其相关方法，拖拽画布可平移视图。

<div class="echarts" data-src="assets/figures/papers/tech-timeline.echarts.json"></div>

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

## 统一建模与 Scaling Law（2024–2026）

这一阶段的核心趋势是将序列建模与特征交叉统一到单一 Transformer 主干中，并验证推荐系统中的 Scaling Law。

> 注：本表年份与仓库内技术图谱及 Semantic Scholar 缓存使用的 year 字段保持一致，以避免与图中 x 轴年份冲突。

| 年份 | 里程碑              | 说明                                                                                           | 本仓库                                                           |
| ---- | ------------------- | ---------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| 2024 | **HSTU / GR**       | Meta 提出 Hierarchical Sequential Transducer Units，1.5 万亿参数生成式推荐器，验证 Scaling Law | —                                                                |
| 2024 | **InterFormer**     | UIUC & Meta 提出交错式异构交互学习，双向信息流 + Cross Arch                                    | [论文](interformer.md) · [实验包](../experiments/interformer.md) |
| 2025 | **OneTrans**        | NTU & 字节提出单 Transformer 主干 + 统一 Tokenizer + KV 缓存                                   | [论文](onetrans.md) · [实验包](../experiments/onetrans.md)       |
| 2025 | **GPSD**            | 阿里巴巴提出生成式预训练初始化判别式推荐，13K→0.3B 参数遵循 Power Law (KDD'25)                 | —                                                                |
| 2025 | **Foundation-Expert** | Meta 提出 Foundation-Expert 范式，首次大规模部署推荐基础模型，日服务数百亿请求                  | —                                                                |
| 2025 | **HoMer**           | 美团提出统一 Encoder-Decoder，序列 + 集合建模消除三重异构性，节省 27% GPU                      | —                                                                |
| 2025 | **MTmixAtt**        | 美团提出 MoE + AutoToken + Multi-Mix Attention，缩放至 1B 参数实现跨场景统一排序               | —                                                                |
| 2026 | **HyFormer**        | 字节提出混合 Transformer，统一长序列建模与特征交叉                                             | [论文](hyformer.md) · [实验包](../experiments/hyformer.md)       |

## 生成式推荐（2023–2026）

生成式推荐将推荐任务从"检索+排序"转变为"序列生成"，利用 LLM 的自回归能力预测下一个物品。

### 关键思路

- **语义 ID**：用有语义含义的 token 序列代替传统物品 ID（如 IDGenRec），让模型理解物品本质
- **RAG 架构**：轻量召回 + LLM 精排，兼顾效率与质量（LlamaRec、PALR）
- **多模态生成**：同时处理文本、图片、视频的统一推荐（UniMP, MMGRec, Molar）
- **可控生成**：Speculative Decoding 加速推理（SpecGR），指令微调实现约束推荐

### 代表工作

| 年份 | 工作       | 说明                                                                              |
| ---- | ---------- | --------------------------------------------------------------------------------- |
| 2024 | GenRec     | 基于 LLM 的序列推荐，Masked Item Prediction                                      |
| 2024 | IDGenRec   | 训练 ID 生成器将元数据转化为语义 ID                                               |
| 2024 | RecGPT     | 生成式预训练文本推荐 (ACL 2024)                                                   |
| 2024 | UniMP      | 统一多模态个性化框架 (ICLR 2024)                                                  |
| 2024 | MMGRec     | 多模态生成推荐 + 层级量化 (CIKM 2024)                                             |
| 2024 | SpecGR     | 基于 Speculative Decoding 的归纳式生成推荐                                        |
| 2024 | Molar      | 多模态 LLM + 协同过滤对齐                                                         |
| 2025 | SessionRec | 下一会话预测范式 (NSPP)，解决传统 NIPP 与真实用户行为的不一致                      |
| 2025 | COBRA      | 级联稀疏-稠密表示的统一生成推荐，2 亿日活广告平台部署 (腾讯)                       |
| 2025 | LLaDA-Rec  | 离散扩散替代自回归，并行生成语义 ID                                                |
| 2025 | RecGPT-V2  | 层级多 Agent + 约束 RL，淘宝部署 CTR +2.98% (阿里)                                |
| 2025 | xGR        | 面向生成式推荐的高效服务系统，3.49x 吞吐提升                                      |
| 2025 | OxygenREC  | 快慢思考 + 指令跟随生成推荐，统一多场景训练一次全场景部署                          |
| 2026 | HiGR       | 层级规划 + 多目标偏好对齐的生成式 Slate 推荐                                      |

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
