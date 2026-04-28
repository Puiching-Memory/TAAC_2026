<h1 align="center">TAAC 2026 Experiment Workspace</h1>

<p align="center">
  <strong>迈向统一序列建模与特征交互的大规模推荐系统</strong>
</p>

<p align="center">
  <a href="https://github.com/Puiching-Memory/TAAC_2026/actions/workflows/ci.yml"><img src="https://github.com/Puiching-Memory/TAAC_2026/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI Status (main)"></a>
  <a href="https://puiching-memory.github.io/TAAC_2026/"><img src="https://img.shields.io/website?label=Docs&up_message=Online&down_message=Offline&up_color=0A7B83&url=https%3A%2F%2Fpuiching-memory.github.io%2FTAAC_2026%2F" alt="Online Docs Status"></a>
  <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.7.1-EE4C2C.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/Task-Recommendation-brightgreen.svg" alt="Task">
  <img src="https://img.shields.io/badge/Track-TAAC%202026-orange.svg" alt="Track">
  <img src="https://img.shields.io/badge/Status-Research-yellow.svg" alt="Status">
</p>

<p align="center">
  <a href="https://algo.qq.com/#intro">Competition</a> ·
  <a href="docs/getting-started.md">Quick Start</a> ·
  <a href="docs/experiments/index.md">Experiments</a> ·
  <a href="https://puiching-memory.github.io/TAAC_2026/">Online Docs</a> ·
  <a href="#交流讨论">QQ 群</a>
</p>

<p align="center">
  <img src="figures/taac2026_promo_hero.png" alt="TAAC 2026 宣传首图">
</p>

> [!NOTE]
> 这是 TAAC 2026 其中一个参赛队伍的代码仓库，不代表官方文档。  
> 我们的目标是提供一个开箱即用、便于扩展和回归验证的实验工作区，
> 以促进社区在统一序列建模与特征交互方向上的研究和创新。

> [!IMPORTANT]
> 感谢各位的支持, 本项目会继续维护，但是需要提前说明：
> 1. 我们无法保证 API 长期稳定。
> 2. 各子模型的研究与复现状态并不等于 100% 官方还原。
>
> 当前仓库的主要开发方向是：
> 1. 提供开箱可用的训练与评估框架。
> 2. 支持大算力场景下的超参数搜索和实验管理。
> 3. 持续同步最新论文、公开方案和可复核实验包。
>
> 当前仓库仅支持 Linux 运行时；Windows 与 WSL 不在支持范围内。

这是一个完全面向 TAAC 2026 大赛的实验工作区。设计目标是共享训练底座、目录式实验包、统一输出产物和回归测试放进同一套工程里，让新实验可以更快接入、训练、评估和复核。

## 比赛简介

推荐系统作为大规模内容平台（信息流、短视频等）与数字广告（点击率/转化率预估等）的核心引擎，直接决定了用户体验、参与度及平台商业收益。面对海量并发请求与严苛的实时响应约束，现代推荐系统每日需完成数十亿次在线决策，支撑起规模庞大的数字广告生态。  

过去二十年间，推荐技术主要沿两条路径演进：一是**特征交互模型**，专注于高维稀疏多域特征与上下文信号的深度交叉；二是**序列模型**，借助 Embedding 检索与 Transformer 架构捕捉用户行为的时序动态。尽管两条路线各自成果丰硕，但长期以来的割裂发展导致工业界系统面临结构性瓶颈：跨范式交互浅层化、优化目标不一致、扩展能力受限，以及日益攀升的硬件与工程复杂度。随着序列长度与模型参数的持续增长，这种碎片化架构的效率瓶颈愈发凸显。

近年来，学界与工业界开始探索融合这两大传统分支的统一建模范式 [1~3]。为加速该方向的突破，我们发起"**迈向统一序列建模与特征交互的大规模推荐系统**"挑战赛。我们鼓励参赛者设计统一的 Tokenization 方案与同质化、可堆叠的骨干网络，在单一架构内同时建模用户行为序列与非序列多域特征，完成转化率预估任务。  

参赛队伍将依据 ROC 曲线下面积（AUC）进行统一排名。除排行榜外，本次大赛特设两项创新奖——**统一模块创新奖**（45,000 美元）与**Scaling Law 创新奖**（45,000 美元），分别表彰在统一架构设计与系统性缩放规律探索方面的杰出工作。创新奖与排行榜名次独立评审，研讨会论文录用将重点考察方法在上述两个方向的新颖性与洞察力，而非单纯追求 AUC 指标。

------

## 我们的工作

![Model Performance VS Size](figures/model_performance_vs_size.svg)

![Model Performance VS Compute](figures/model_performance_vs_compute.svg)

## 快速开始

```bash
uv python install 3.10.20
uv sync --locked --extra cuda126

# 训练baseline
bash run.sh train --experiment config/baseline \
  --dataset-path /path/to/dataset_dir \
  --schema-path /path/to/dataset_dir/schema.json

# 评估默认输出目录中的 best.pt；single 模式始终只评估一个实验/一个 checkpoint
bash run.sh val --experiment config/baseline \
  --dataset-path /path/to/dataset_dir \
  --schema-path /path/to/dataset_dir/schema.json
```

```bash
# 生成线上训练上传文件
uv run taac-package-train --experiment config/baseline

# 生成线上推理上传文件
uv run taac-package-infer --experiment config/baseline

# 跑完整训练栈回归
uv run pytest tests -q
```

## 当前支持实验包

| 实验包         | 目录                                           | 公开来源                                                                                                                                      |
| -------------- | ---------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| Baseline       | [config/baseline](config/baseline)             | 官方 DHyFormer baseline                                                                                                                       |
| Symbiosis      | [config/symbiosis](config/symbiosis)           | 本仓库维护的比赛用融合实验模型                                                                                                                |
| CTR Baseline   | [config/ctr_baseline](config/ctr_baseline)     | [creatorwyx/TAAC2026-CTR-Baseline](https://github.com/creatorwyx/TAAC2026-CTR-Baseline)                                                       |
| DeepContextNet | [config/deepcontextnet](config/deepcontextnet) | [suyanli220/TAAC-2026-Baseline-Tencent-Advertisement-Contest](https://github.com/suyanli220/TAAC-2026-Baseline-Tencent-Advertisement-Contest) |
| InterFormer    | [config/interformer](config/interformer)       | [InterFormer paper](https://arxiv.org/abs/2411.09852)                                                                                         |
| OneTrans       | [config/onetrans](config/onetrans)             | [OneTrans paper](https://arxiv.org/abs/2510.26104)                                                                                            |
| HyFormer       | [config/hyformer](config/hyformer)             | [HyFormer paper](https://arxiv.org/abs/2601.12681)                                                                                            |
| UniRec         | [config/unirec](config/unirec)                 | [hojiahao/TAAC2026](https://github.com/hojiahao/TAAC2026)                                                                                     |
| UniScaleFormer | [config/uniscaleformer](config/uniscaleformer) | [twx145/Unirec](https://github.com/twx145/Unirec)                                                                                             |

更详细的训练命令、线上训练/推理打包说明和各实验包说明，可以看 [docs/getting-started.md](docs/getting-started.md)、[docs/guide/online-training-bundle.md](docs/guide/online-training-bundle.md)、[docs/guide/official-competition-docs.md](docs/guide/official-competition-docs.md)、[docs/experiments/index.md](docs/experiments/index.md) 和 [docs/architecture.md](docs/architecture.md)。

------

## Timeline
1. Competition Begins - Mar.15, 2026 - 23:59:59 AOE - Releasing demo dataset
2. Global Registration - Mar.19 ~ Apr.23 - 23:59:59 AOE
3. First-round Competition - Apr.24 ~ May 23 - 23:59:59 AOE
4. Second-round Competition - May 25 ~ Jun.24 - 23:59:59 AOE
5. Winners Announcement - Jul.15, 2026 Winner Notification - Aug. 9, 2026 - Winner Public Announcement

## Our Eligibility
Academic Track

## Dataset&Task

> [!NOTE]
> 本次比赛发布的数据集经过完全匿名化处理，不反映腾讯广告平台的实际生产特性。  
> 所有稀疏特征均以匿名整数 ID 表示，稠密特征以固定长度浮点向量提供；官方不发布原始文本、图像、URL 或任何个人身份信息。

> [!IMPORTANT]
> **Update [2026.04.10]**: 示例数据集已更新为扁平列布局格式，特征名已重命名，新增序列特征。请参考最新的 `demo_1000.parquet` 和 HuggingFace 上的 README 获取最新 schema 详情。  
> 
> 本项目已经同步更新最新的数据格式

下载链接: https://huggingface.co/datasets/TAAC2026/data_sample_1000

官网披露的初赛数据集是一个基于真实广告日志构建的大规模工业级数据集，包含约 2 亿条用户序列。数据由两类核心信息组成：一类是用户与物品之间的行为序列，例如曝光、点击和转化，并附带时间戳、动作类型等上下文信息；另一类是非序列多字段特征，覆盖用户属性、物品属性、上下文信号和交叉特征。

当前样例数据采用扁平列布局（flat column layout）：所有特征都作为独立的顶级列存储在 Parquet 文件中，而不是嵌套结构。样例文件共 120 列，官网摘要如下：

| 特征分组       | 列数 | 数据形态                | 说明                                           |
| -------------- | ---- | ----------------------- | ---------------------------------------------- |
| ID 与标签      | 5    | `int64` / `int32`       | 核心标识、监督标签和时间戳                     |
| 用户整型特征   | 46   | `int64` / `list<int64>` | 单值或多值离散用户特征，描述用户属性与偏好     |
| 用户稠密特征   | 10   | `list<float>`           | 连续值用户特征，包含 embedding 与对齐统计信号  |
| 物品整型特征   | 14   | `int64` / `list<int64>` | 离散物品特征，包含类目、类型、基础信息与多标签 |
| 域行为序列特征 | 45   | `list<int64>`           | 来自 4 个行为域的用户行为序列特征              |

### 详细字段结构

**ID 与标签列（5 列）**

这 5 列均无空值：

| 字段 | `user_id` | `item_id` | `label_type` | `label_time` | `timestamp` |
| ---- | --------- | --------- | ------------ | ------------ | ----------- |
| 类型 | `int64`   | `int64`   | `int32`      | `int64`      | `int64`     |

**用户稠密特征（10 列）**

- `user_dense_feats_{61, 87}`：共 2 列，表示用户 embedding 特征（SUM、LMF4Ads）。
- `user_dense_feats_{62-66, 89-91}`：共 8 列，与 `user_int_feats_{62-66, 89-91}` 一一对应，数组长度保持一致；例如 `user_int_feats_62: [1, 2, 3]` 与 `user_dense_feats_62: [10.5, 20, 15.5]` 按元素对齐。

**物品整型特征（14 列）**

- `item_int_feats_{5-10, 12-13, 16, 81, 83-85}`：共 13 列，标量 `int64`。
- `item_int_feats_11`：共 1 列，数组 `list<int64>`。

**域行为序列特征（45 列）**

- `domain_a_seq_{38-46}`：9 列。
- `domain_b_seq_{67-79, 88}`：14 列。
- `domain_c_seq_{27-37, 47}`：12 列。
- `domain_d_seq_{17-26}`：10 列。

可以用示例样本快速查看当前字段：

```python
import pandas as pd
df = pd.read_parquet("demo_1000.parquet")
print(df.shape)       # (1000, 120)
print(df.columns)     # ['user_id', 'item_id', 'label_type', ...]
```

如果你按仓库当前文档做本地 smoke，推荐目录布局如下：

```text
data/sample_1000_raw/
├── demo_1000.parquet
└── schema.json
```

补充说明：官方 `demo_1000.parquet` 当前只有 1 个 Row Group。本仓库已经兼容这种样例文件，在 smoke 训练时会复用同一个 Row Group 做 train/valid 切分，仅用于通路验证，不代表有统计意义的离线验证。

## Evaluation
我们将使用单一的ROC曲线下面积（AUC）指标对所有团队进行排名（越高越好）。为确保实用性，每次提交还必须在官方评估环境和协议下满足特定于赛道和轮次的推理延迟限制；超出延迟预算的提交将被视为无效，因此不予排名，无论AUC分数如何。

为鼓励与我们主题一致的创新——构建一个统一模块，弥合序列建模与多字段特征交互之间的鸿沟，并探索推荐系统的缩放规律——我们还将提供两项创新奖：统一模块创新奖（45,000美元）和缩放规律创新奖（45,000美元）。这些奖项与排行榜排名无关。最终获奖决定将由委员会根据提交的技术报告、代码以及所提方法的新颖性和洞察力进行综合评审，特别是围绕本次比赛强调的两个方向，而非仅关注最终AUC分数。

## Rules
**评分标准**
比赛设有两条平行赛道，分别拥有独立的排行榜。  
学术赛道仅限团队成员全部隶属于大学或学院的队伍参加（如本科生、硕士生或博士生；需提供学术 affiliation 证明）。工业赛道则无资格限制，向所有参与者开放。为更好地反映部署约束，工业赛道将执行更严格的推理延迟限制。  
为强调方法论的清晰性并实现公平比较，我们禁止在整个比赛中使用模型集成。

比赛采用两阶段评估框架，逐步强调预测准确性、可扩展性、效率和可复现性。在第一轮（开放初赛阶段），所有团队将在隐藏测试集上根据官方评估指标进行排名，同时实施严格的防过拟合控制（如提交限制和延迟反馈）。如有必要，将实施容量感知滚动准入机制（支持多达5,000支并发团队），以确保公平的资源访问。第一轮结束时，排行榜将被冻结，前50名学术团队和前20名工业团队将仅根据官方指标表现晋级第二轮。
第二轮在约10倍更大规模的数据集上评估模型的鲁棒性和大规模建模能力，同时设置严格的推理延迟限制，以鼓励采用GPU高效统一架构。每支决赛团队将获得相当的计算资源，且所有提交必须通过官方环境中的可复现性和规则合规性验证。

## 社区

欢迎加入 TAAC2026(民间群) 交流训练、复现、实验管理和线上提交经验。QQ群：1098676137。

![Alt](https://repobeats.axiom.co/api/embed/fa09d4072a7dc3cf21ed2a9fcccdd2847ce9c2f7.svg "Repobeats analytics image")

<a href="https://www.star-history.com/?repos=Puiching-Memory%2FTAAC_2026&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=Puiching-Memory/TAAC_2026&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=Puiching-Memory/TAAC_2026&type=date&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=Puiching-Memory/TAAC_2026&type=date&legend=top-left" />
 </picture>
</a>

## 相关工作
以下按公开可访问资料整理，优先保留能直接借鉴代码、EDA、方法说明和赛事资料的链接，持续补充。
调查时间: 2026-04-24

**2025届：官方 / 公开代码**  
1. [TencentAdvertisingAlgorithmCompetition/baseline_2025](https://github.com/TencentAdvertisingAlgorithmCompetition/baseline_2025) 官方 parquet baseline，主体为 SASRec，并附带 faiss-based-ann 检索与 RQ-VAE 扩展入口。  
2. [zcyeee/TAAC](https://github.com/zcyeee/TAAC) 决赛方案公开仓库，README 给出生成式 next-item 推荐框架、训练流程与 Top-K 推理脚本。  
3. [salmon1802/O_o](https://github.com/salmon1802/O_o) O_o 队伍公开代码，仓库说明标注为 2025 初赛第十四名 / 初赛 Top 1%。  
4. [mx-Liu123/OmniGenRec-TAAC2025](https://github.com/mx-Liu123/OmniGenRec-TAAC2025) 复现 OmniGenRec 两个关键组件，README 给出 HR@10 / NDCG@10 的提升记录。  

**2025届：博客 / 新闻 / 资料**  
1. [TAAC七日游](https://pd-ch.github.io/blog/2025-07-31-taac-participate-record/) 一份较完整的个人复盘，覆盖论文补课、RQ-VAE/HSTU 学习、实验记录和比赛期资料整理。  
2. [从算法大赛千名开外到鹅厂技术骨干，他们亲授“逆袭秘籍”｜学长深度访谈直播实录](https://mp.weixin.qq.com/s/mAVOICmMOay_Axcr0IN4PA) 官方公众号文章，偏组队、工程化、提交策略和竞赛节奏。  
3. [一文读懂算法大赛前沿赛题｜赛前必看攻略第7期](https://mp.weixin.qq.com/s/xz0kb-xjCOy_A0k_gYwKeg) 官方赛前攻略，梳理赛题重点、baseline 思路和优化方向。  
4. [Angel平台&GPU虚拟化技术全解析｜赛期进阶攻略第1期](https://mp.weixin.qq.com/s/yzqPYYm0Ybf8_6A-IlIYBQ) 官方平台资料，偏训练环境、GPU 虚拟化和赛期工程细节。  

**2026届：公开仓库 / 方案**  
1. [creatorwyx/TAAC2026-CTR-Baseline](https://github.com/creatorwyx/TAAC2026-CTR-Baseline) DIN baseline，侧重流式清洗、地址簿随机读取与单机训练工程化。  
2. [suyanli220/TAAC-2026-Baseline-Tencent-Advertisement-Contest](https://github.com/suyanli220/TAAC-2026-Baseline-Tencent-Advertisement-Contest) DeepContextNet baseline，显式走 HSTU 风格序列建模与 Muon 优化器路线。  
3. [hojiahao/TAAC2026](https://github.com/hojiahao/TAAC2026) UniRec 方案，强调 unified tokenization、混合 attention mask、scaling law 和 2 卡 DDP。  
4. [twx145/Unirec](https://github.com/twx145/Unirec) UniScaleFormer 模板，内置 InterFormer / OneTrans / HyFormer / base 配置对比与 scaling law 脚本。  
5. [XiaolongWang-c/tencent-ad](https://github.com/XiaolongWang-c/tencent-ad) 轻量级 TAAC 2026 备赛工程脚手架，强调统一 Sample 抽象、显式标签映射入口与验证预测产物，便于快速替换 baseline 与特征工程。  

**2026届：Kaggle / Notebook**  
1. [galegale05/TAAC2026 Baseline v3 - Final](https://www.kaggle.com/code/galegale05/taac2026-baseline-v3-final/notebook) Kaggle 上公开的 HSTU 风格时间特征 baseline notebook，可作为时间 bucket、session 切分和轻量级序列建模的补充参考。  

**2026届：EDA / 资料入口**  
1. [hun9008/TAAC_DI_Lab_EDA](https://github.com/hun9008/TAAC_DI_Lab_EDA) 对公开 sample parquet 做了较完整的 EDA，包含 label 分布、序列长度、feature 密度和建模建议。  
2. [https://huggingface.co/datasets/TAAC2026/data_sample_1000](https://huggingface.co/datasets/TAAC2026/data_sample_1000) 官方样例数据页面。  
3. [https://algo.qq.com/#intro](https://algo.qq.com/#intro) 大赛主页。  

**通用开源框架 / Benchmark**  
1. [reczoo/FuxiCTR](https://github.com/reczoo/FuxiCTR) CTR 预测开源底座，长处是可配置、可调参与可复现实验，适合快速对照经典 ranking 模型与数据管线。
2. [meta-recsys/generative-recommenders](https://github.com/meta-recsys/generative-recommenders) Meta 官方 HSTU / Generative Recommenders 代码，包含训练、推理与公开实验脚本，是统一生成式路线的重要工程参考。
3. [snap-research/GRID](https://github.com/snap-research/GRID) Semantic ID 生成式推荐框架，串起文本 embedding、RQ 式语义 ID 学习与 Transformer 解码，适合后续探索 item-side semantic tokenization。
4. [datawhalechina/torch-rechub](https://github.com/datawhalechina/torch-rechub) 轻量级 PyTorch 推荐框架，覆盖 matching、ranking、multi-task 与 generative 等多类模型，并提供统一训练流程、ONNX 导出与工程化示例，适合作为经典推荐建模与部署链路的对照参考。

## References
```bibtex
@misc{interformer2025,
  author = {Zhichen Zeng and Xiaolong Liu and Mengyue Hang and Xiaoyi Liu and Qinghai Zhou and Chaofei Yang and Yiqun Liu and Yichen Ruan and Laming Chen and Yuxin Chen and Yujia Hao and Jiaqi Xu and Jade Nie and Xi Liu and Buyun Zhang and Wei Wen and Siyang Yuan and Hang Yin and Xin Zhang and Kai Wang and Wen-Yen Chen and Yiping Han and Huayu Li and Chunzhi Yang and Bo Long and Philip S. Yu and Hanghang Tong and Jiyan Yang},
  title = {InterFormer: Effective Heterogeneous Interaction Learning for Click-Through Rate Prediction},
  year = {2025},
  eprint = {2411.09852},
  archivePrefix = {arXiv},
  note = {CIKM 2025},
  doi = {10.48550/arXiv.2411.09852},
  url = {https://arxiv.org/abs/2411.09852},
}

@misc{onetrans2025,
  author = {Zhaoqi Zhang and Haolei Pei and Jun Guo and Tianyu Wang and Yufei Feng and Hui Sun and Shaowei Liu and Aixin Sun},
  title = {OneTrans: Unified Feature Interaction and Sequence Modeling with One Transformer in Industrial Recommender},
  year = {2025},
  eprint = {2510.26104},
  archivePrefix = {arXiv},
  note = {Accepted at The Web Conference 2026 (WWW 2026)},
  doi = {10.48550/arXiv.2510.26104},
  url = {https://arxiv.org/abs/2510.26104},
}

@misc{hyformer2026,
  author = {Yunwen Huang and Shiyong Hong and Xijun Xiao and Jinqiu Jin and Xuanyuan Luo and Zhe Wang and Zheng Chai and Shikang Wu and Yuchao Zheng and Jingjian Lin},
  title = {HyFormer: Revisiting the Roles of Sequence Modeling and Feature Interaction in CTR Prediction},
  year = {2026},
  eprint = {2601.12681},
  archivePrefix = {arXiv},
  note = {arXiv preprint},
  doi = {10.48550/arXiv.2601.12681},
  url = {https://arxiv.org/abs/2601.12681},
}
```
