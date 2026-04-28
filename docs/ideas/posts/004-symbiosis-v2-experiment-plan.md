---
icon: lucide/clipboard-check
---

# Symbiosis V2 改造与实验验证计划

:material-calendar: 2026-04-28 · :material-tag: Symbiosis, AUC, 统一模块, 长序列, 消融实验, Scaling Law

## 背景

TAAC 2026 的排行榜主指标是 AUC，线上推理契约要求输出 `user_id -> probability`。因此 Symbiosis 的下一代改造不能把主线改成 HR@K / NDCG 式检索，也不能把 `predictions.json` 改成 top-k item。所有方案都必须服务于：

```text
P(label_action_type = 2 | user, candidate item, context, multi-domain history)
```

本文把前面讨论过的想法整理成可验证计划。这里的每个改造点都只是待检验假设，不预设一定有效。实验必须以当前 `config/symbiosis` 为对照，记录 AUC、logloss、推理延迟、参数量、训练稳定性和分群指标，再决定是否保留。

## 对当前 Symbiosis 的客观判断

当前 Symbiosis 不是一无是处。它已经具备几个值得保留或至少认真消融的资产：

| 现有设计                         | 可能价值                                                         | 主要风险                                                            | 验证方式                                                   |
| -------------------------------- | ---------------------------------------------------------------- | ------------------------------------------------------------------- | ---------------------------------------------------------- |
| UserItemGraphBlock               | 在非序列 token 层提前做 user-item 双向交互，可能提高候选相关性   | 交互太早可能放大噪声或增加重复建模                                  | 与移除 graph block 的版本比较 AUC、logloss、延迟           |
| UnifiedBlock                     | 将 prompt、非序列 token 和序列读取放进统一空间，符合统一模块主题 | 所有 domain 的序列更新被简单平均，候选相关性可能不够强              | 比较统一 block、domain-wise decoder、无统一 block 三种结构 |
| FourierTimeEncoder               | 显式利用时间桶和周期特征，可能改善近期意图建模                   | 时间桶离散化已包含部分信息，额外 Fourier 投影可能过拟合             | 开关 time encoder，按序列长度和近期行为桶分群看 AUC        |
| ContextExchangeBlock             | 让全局 context 从序列域读取信息，补充非序列表征                  | 单个 context query 可能过粗，无法区分 candidate 相关历史            | 与 candidate-conditioned query 比较                        |
| 多尺度 mean / recent / last 摘要 | 低成本保留长期、近期、末次行为信号                               | `tokens.shape[1] // 2` 的 recent 定义偏粗，和真实时间近期不一定一致 | 与 top-k 最近有效 token、压缩块记忆比较                    |
| ActionConditionedHead            | 已经有 action prototype 的雏形，兼容多 action 输出               | action 只在 head 里出现，可能太晚                                   | 与 action token 前移版本比较                               |

现阶段不应该直接推翻当前 Symbiosis。更合理的做法是先做组件消融，确认哪些模块确实贡献 AUC，哪些只是增加复杂度。

## 总体实验原则

1. 主指标只看 validation AUC，logloss 和校准作为副指标。
2. 任一架构改动都必须报告推理延迟；超出线上预算的 AUC 增益不计为有效收益。
3. 每个方案先跑小样本 smoke，再跑固定 validation slice，最后才扩大到正式训练。
4. 单次 AUC 提升小于随机波动时，不写成有效结论；关键方案至少跑 3 个 seed。
5. 不把辅助目标、Semantic ID、compressed memory 的收益提前当真；它们只在消融中证明自己。
6. 保留当前 PCVR package contract：`forward()` 返回 logits，`predict()` 返回 `(logits, embeddings)`，不新增 package-local trainer 或 run.sh。

## 本地测试数据不足的处理

当前本地样本量太小，baseline 轻松达到 0.95 AUC。这类结果不能直接作为模型选择依据，因为它通常意味着验证集过容易、方差过大，或者存在与线上分布不一致的采样偏差。小样本本地评估应定位为 smoke test：验证代码能跑通、loss 不崩、输出概率合法、推理延迟大致可控，而不是判断一个架构改动是否真的有效。

后续实验需要把验证可信度分成三层：

| 层级                                 | 数据来源                           | 主要用途                                  | 结论可信度                 |
| ------------------------------------ | ---------------------------------- | ----------------------------------------- | -------------------------- |
| L0 smoke                             | `sample_1000` 或同等小样本         | 检查 forward/backward、打包、推理、无 NaN | 只能证明没坏，不能证明有效 |
| L1 internal validation               | 从可用训练数据中切出的固定大验证集 | 比较消融、loss、架构改动                  | 可做阶段性模型选择         |
| L2 platform / hidden-like validation | 官方平台或尽量贴近线上分布的大样本 | 最终保留或淘汰方案                        | 才能作为真实收益判断       |

在 L0/L1 数据有限时，必须增加不确定性估计：

- 对 validation 预测做 bootstrap，报告 AUC 的 95% 置信区间。
- 对关键方案至少跑 3 个 seed，报告 mean、std 和最差 seed。
- 对同一批预测同时看 AUC、logloss、score margin、正负样本分数分布，避免只看一个饱和 AUC。
- 对序列长度、item 频率、用户活跃度、domain 缺失模式做切片；如果整体 AUC 饱和，切片 AUC 往往更能暴露差异。
- 如果一个改动只在 `sample_1000` 上提升，但 bootstrap 置信区间高度重叠，则视为无结论。
- 如果一个改动在小样本上 AUC 不升但延迟、稳定性或分群指标明显变好，只能标为“值得上大样本复核”，不能直接保留。

实操上，任何小于 0.001 的 AUC 差异都不应在小样本上解读为有效收益；即使大于 0.001，也必须通过 bootstrap 和多 seed 确认。若 baseline 已经接近 0.95，应优先寻找更难的验证切片，例如长序列用户、低频 item、domain 缺失样本、近期行为弱相关样本，而不是继续在整体 AUC 上做细微比较。

后续 L1 消融应直接使用 `taac-evaluate single`。评估输出的 `metrics` 会和 `auc` 一起写出 `auc_ci`、`logloss`、`brier`、`score_diagnostics` 和 `sample_count`；顶层 `data_diagnostics` 会同时记录 parquet Row Group 数量、默认 train/valid 切分是否复用、是否适合 L1 比较。只有当 `data_diagnostics.row_group_split.is_l1_ready=true`、`reuse_train_for_valid=false`、`is_disjoint=true` 时，才把这份数据用于模型优劣比较。若像 `sample_1000` 一样只有单个 Row Group，评估报告会标为 L0-only，只能做链路 smoke。

不同 run 之间不再依赖单独的分析 CLI。比较时先看每个 run 自带的 `auc_ci` 是否与 baseline 明显分离；若置信区间重叠，结论应标为“无结论”或“需要更大验证集 / 多 seed”。

## 实验基线

| 基线                       | 目的                                                | 预期结果                                              | 判定标准                              |
| -------------------------- | --------------------------------------------------- | ----------------------------------------------------- | ------------------------------------- |
| B0: baseline 官方 HyFormer | 官方参考下限                                        | AUC 稳定、延迟较低                                    | 所有方案必须至少与它比较              |
| B1: 当前 Symbiosis         | 当前融合式实现                                      | 可能高于 baseline，也可能因复杂度和训练不充分表现不稳 | 作为所有 Symbiosis V2 方案的主对照    |
| B2: Symbiosis-lite         | 保留 tokenizer、time、multi-scale，去掉较重统一交互 | AUC 可能略降但延迟更低，用来估计复杂模块 ROI          | 若 B2 接近 B1，说明当前重模块贡献不足 |
| B3: Symbiosis-full-log     | 不改模型，只增加诊断日志和分群评估                  | AUC 不变，但能解释误差来源                            | 为后续方案提供可比较诊断              |

## 方案 A：当前组件消融

### 假设

当前 Symbiosis 的一部分模块已经有效，但贡献大小未知。先消融再重构，避免误删有效结构。

### 实验

- A1: 移除 `UserItemGraphBlock`。
- A2: 移除 `FourierTimeEncoder`，只保留原始 time bucket embedding。
- A3: 移除 `ContextExchangeBlock`。
- A4: 移除 `_multi_scale_context`。
- A5: 将 `UnifiedBlock._attend_sequences()` 的 domain mean 改为 gated domain weighting。
- A6: 将 `num_blocks` 从 3 扫到 1、2、4。

### 预期结果

- 如果当前 Symbiosis 的融合思路有效，A1-A4 至少有一个会带来可复现 AUC 下降。
- 如果多个消融几乎不影响 AUC，说明当前复杂度存在冗余，应优先简化而不是继续堆模块。
- A5 若有效，预期 AUC 小幅提升，同时 domain 权重能解释哪些序列域有贡献。

### 风险

小样本上消融结论可能受随机性影响；正式结论需要固定 validation slice 和多 seed。

## 方案 B：AUC-aware 训练目标

### 假设

AUC 只关心正负样本相对排序。BCE 隐式优化 AUC，但加入 batch 内 pairwise ranking loss 可能直接改善排序质量。

### 设计

保留 BCE 或 focal 作为主概率约束，增加 pairwise 项：

```text
loss = BCEWithLogits + lambda_pairwise * softplus(-(pos_logit - neg_logit) / temperature)
```

负样本先从同 batch 采样，再尝试 hard negative，即当前模型打分偏高的负样本。

### 预期结果

- AUC 可能提升 0.0005 到 0.003，尤其是在 hard negative 较多的 validation slice 上。
- logloss 可能持平或略差，因为 pairwise loss 更关注排序而不是概率校准。
- 若 lambda 过大，概率分布可能变尖，logloss 和稳定性变差。

### 判定标准

- AUC 提升必须超过 seed 波动，并且 logloss 不能明显劣化。
- 若 AUC 升但 logloss 明显坏，需要只作为后期微调策略，而不是全程训练默认。

## 方案 C：Candidate-conditioned 序列解码

### 假设

当前序列读取偏全局摘要，无法充分回答“这个候选 item 与用户哪段历史相关”。用 candidate token 和 conversion query 去读取各 domain 序列，可能更贴近 AUC 排序。

### 设计

每个 domain sequence 增加轻量解码器：

```text
query = candidate_context + conversion_query + user_context
key/value = domain sequence tokens
output = candidate_specific_domain_intent
```

输出保留 domain 维度，不再简单平均。最终由 domain gate 融合。

### 预期结果

- 若候选相关历史是主要信号，AUC 可能提升 0.001 到 0.004。
- 对长序列用户、候选与历史强相关样本，分群 AUC 应更明显提升。
- 推理延迟可能增加 10% 到 30%，需要用 top-k 或压缩记忆控制。

### 风险

- 若当前 item token 已经通过 UnifiedBlock 充分参与交互，新增 decoder 可能重复建模。
- 若序列噪声大，candidate query 可能过拟合偶然共现。

## 方案 D：Action conditioning 前移

### 假设

26 届离线任务接近点击样本中的转化预测。只在 head 里做 action conditioning 太晚，历史行为 token 应该显式携带 action 语义。

### 设计

如果 schema 中的历史序列包含 action 类字段，则加入：

```text
event_token = id_embedding + action_embedding + domain_embedding + time_embedding
target_token = conversion_query
```

如果当前序列没有可用 action 字段，则先不做伪 action，避免泄漏或制造假信号。

### 预期结果

- 若历史 action 可用，AUC 可能提升 0.0005 到 0.002。
- 转化稀疏或长序列用户上的提升可能更明显。
- 若 action 与 label 构造强相关，必须检查是否存在 target leakage。

### 判定标准

只允许使用历史事件 action，不允许使用当前样本 label 或未来信息。任何无法证明无泄漏的结果不计入有效收益。

## 方案 E：DeepSeek-V4 式多分辨率序列记忆

### 假设

长序列既需要近期细粒度行为，也需要远期压缩兴趣。将 CSA/HCA 思想改造成推荐序列记忆，可能在不显著增加延迟的情况下提升长历史利用率。

### 设计

每个 domain 构造三类 memory：

```text
recent memory      最近 N 个有效行为，保留原始 token
compressed memory  每 m 个行为压缩成一个 block
global memory      更大压缩率的长期摘要
```

candidate query 先选择 top-k compressed blocks，再与 recent memory 联合 attention。加入 learnable sink token，允许模型判断某个 domain 对当前候选无贡献。

### 预期结果

- 长序列分群 AUC 可能提升 0.001 到 0.003。
- 全局 AUC 未必显著提升，因为短序列或弱序列用户占比可能稀释收益。
- 推理延迟应低于 full sequence attention；若高于当前 Symbiosis 30% 以上，需要降级为 optional path。

### 风险

压缩 block 的学习可能不稳定；top-k 选择若不可微或实现复杂，第一版应采用 soft top-k 或固定窗口近似。

## 方案 F：Attention sink / null evidence token

### 假设

并非每个候选 item 都能从每条历史序列中找到相关证据。强迫 attention 分配到历史 token 可能引入噪声。sink token 允许模型选择“不读取该 domain”。

### 预期结果

- 对噪声序列、弱相关 domain，AUC 和 logloss 可能小幅改善。
- domain attention 分布应更稀疏，sink 权重可作为诊断信号。
- 如果 sink 权重长期过高，说明序列模块没有被有效利用。

### 判定标准

只有在 AUC 不降且 sink 使用率与分群结果有合理相关性时保留。

## 方案 G：Multi-lane constrained residual mixing

### 假设

Symbiosis 统一模块混合了 user、item、prompt、recent、long、domain 等多源信号。普通残差可能让某一类信号支配其他信号。借鉴 mHC 的稳定性思想，可用轻量 lane mixing 保护多源信息流。

### 设计

维护多个 lane：

```text
candidate lane
user lane
recent sequence lane
long sequence lane
domain interaction lane
```

每层使用非负归一化 mixing 权重，而不是无限制线性混合。第一版不实现完整 Sinkhorn，只做 row-stochastic gate。

### 预期结果

- 深层配置如 `num_blocks >= 4` 的训练稳定性可能改善。
- AUC 提升未必来自浅层模型，主要看深度 scaling 是否更顺滑。
- 参数和延迟增加应很小。

### 风险

约束过强可能限制表达能力；如果浅层模型效果不升，不应强行保留。

## 方案 H：Semantic ID 表征增强

### 假设

Semantic ID 不应作为 26 届主输出，但可以作为 item/candidate 的离散语义补充，帮助长尾 item 和多模态/稀疏特征泛化。

### 设计

先离线生成 item semantic code，再作为 item 侧额外 token 或 embedding 输入。主输出仍是 probability。

### 预期结果

- 长尾 item、低频 item 分群 AUC 可能提升。
- 全局 AUC 可能只有小幅变化，甚至因码本质量差而下降。
- 离线码本生成成本较高，不应作为第一阶段改造。

### 判定标准

只有当低频 item 分群提升且整体 AUC 不降时进入正式方案。

## 方案 I：优化器与 scaling 实验

### 假设

DeepSeek-V4 的 Muon 启发在于优化器分治，而不是盲目替换。推荐模型应保持稀疏 embedding 和 dense interaction block 分开优化。

### 实验

- 稀疏 embedding 继续用现有稀疏优化策略。
- dense attention / mixer / projection 可试 AdamW vs Muon 类优化器。
- norm、bias、gate、head 保守使用 AdamW。

### 预期结果

- 可能提升收敛速度，而不一定提升最终 AUC。
- 若最终 AUC 相同但达到同等 AUC 的 step 更少，可作为 scaling law 方向证据。

### 风险

引入新优化器会增加环境与复现成本；只有在现有依赖和线上环境可控时才推进。

## 实验矩阵

| 阶段 | 实验                          | 目标                            | 成功标准                         | 退出条件                            |
| ---- | ----------------------------- | ------------------------------- | -------------------------------- | ----------------------------------- |
| S0   | B0/B1/B2/B3 基线              | 建立可信对照和诊断日志          | 指标稳定，能复现当前结果         | 基线自身不稳定，先修数据/训练流程   |
| S1   | 当前组件消融                  | 判断现有 Symbiosis 哪些模块有效 | 找到至少一个可解释的正贡献模块   | 所有模块无贡献，转向简化版          |
| S2   | AUC-aware loss                | 直接优化排序                    | AUC 提升且 logloss 可接受        | 多 seed 不稳定或校准明显恶化        |
| S3   | Candidate-conditioned decoder | 强化候选相关历史读取            | 长序列/强历史分群 AUC 提升       | 延迟过高或全局 AUC 下降             |
| S4   | Action 前移                   | 区分点击兴趣和转化意图          | 无泄漏前提下 AUC 提升            | schema 不支持历史 action 或疑似泄漏 |
| S5   | 多分辨率 memory + sink        | 提升长序列效率和抗噪            | 长序列分群提升，延迟可控         | 复杂度高但收益不可复现              |
| S6   | Lane mixing                   | 验证深层统一模块稳定性          | 深度 scaling 更平滑              | 浅层和深层均无收益                  |
| S7   | Semantic ID                   | 增强候选表征                    | 长尾 item 分群提升               | 码本成本高且整体 AUC 不升           |
| S8   | 优化器/scaling                | 研究计算效率和缩放规律          | 同等 AUC 所需 step 或 FLOPs 降低 | 环境不可复现或收益只出现在单 seed   |

## 指标记录模板

每个 run 至少记录：

```text
run_id
git_commit
experiment_package
model_variant
seed
train_rows
valid_rows
num_parameters
train_steps
best_auc
best_logloss
final_auc
final_logloss
inference_latency_ms_per_batch
positive_score_mean
negative_score_mean
score_margin_mean
nan_count
```

推荐增加分群：

- 序列长度桶：空、短、中、长、超长。
- item 频率桶：低频、中频、高频。
- 用户活跃度桶。
- domain 缺失模式。
- 最近行为时间桶。

## 初始优先级

第一阶段不要直接写一个巨大的 Symbiosis V2。建议顺序是：

1. 先做 B3 诊断日志和 A 系列消融。
2. 再做 B 系列 AUC-aware loss，因为改动小、贴指标。
3. 再做 C 系列 candidate-conditioned decoder，这是最核心的架构假设。
4. 然后根据 C 的收益决定是否进入 E/F 的压缩记忆和 sink token。
5. G/H/I 作为第二阶段，服务深层 scaling、长尾泛化和训练效率。

## 已启动工作

- 2026-04-28：启动 B3 诊断日志第一步，在共享 metrics、PCVR trainer 和离线 evaluation 中加入 `score_diagnostics`，记录正负样本数量、正负样本平均分、平均分差、分数标准差、分位数和 invalid count。该改动不改变模型结构和训练目标，只增强后续消融实验的观测能力。
- 2026-04-28：启动 A 系列组件消融基础设施，在共享训练 CLI、PCVR model builder 和 `config/symbiosis` 中加入可复现开关：`symbiosis_use_user_item_graph`、`symbiosis_use_fourier_time`、`symbiosis_use_context_exchange`、`symbiosis_use_multi_scale`、`symbiosis_use_domain_gate`。默认保持当前 Symbiosis 行为不变，后续可逐项关闭或启用 domain gate 做 A1-A5 实验。
- 2026-04-28：完成首轮 L0 smoke。使用 `data/sample_1000_raw/demo_1000.parquet`、CPU、1 epoch、`batch_size=64`、`num_workers=0`、关闭 AMP/compile，跑通 B1 默认 Symbiosis 和 A1-A5 组件消融。由于该 parquet 只有单个 Row Group，训练和验证复用同 1000 行，结果只证明链路可运行、开关可落盘、诊断指标可读，不作为模型优劣结论。
- 2026-04-28：将 AUC 解释性指标内聚到 `taac-evaluate single`。每次 eval/test 输出 AUC 时，会自动一起计算 `auc_ci`、`score_diagnostics` 和 `data_diagnostics`，避免为 AUC 表达力不足再维护独立分析 CLI。
- 2026-04-28：对 B1/A1-A5 L0 smoke checkpoint 重新运行 `taac-evaluate single`，生成各 run 的 `validation_predictions.jsonl` 和增强后的 `evaluation.json`。六个 run 的 bootstrap 95% CI 全部与 B1 baseline 重叠，当前只说明评估链路可用，不说明任何组件优劣。

| L0 run   | 开关                                  | 参数量      | valid AUC | logloss  | pos mean | neg mean | margin   | 用时 |
| -------- | ------------------------------------- | ----------- | --------- | -------- | -------- | -------- | -------- | ---- |
| B1 smoke | 默认 Symbiosis                        | 160,824,772 | 0.975586  | 0.227011 | 0.331396 | 0.068890 | 0.262506 | 43s  |
| A1 smoke | `--no-symbiosis-use-user-item-graph`  | 160,376,260 | 0.978329  | 0.237869 | 0.295058 | 0.078232 | 0.216826 | 44s  |
| A2 smoke | `--no-symbiosis-use-fourier-time`     | 160,824,772 | 0.976727  | 0.223777 | 0.337535 | 0.069100 | 0.268436 | 46s  |
| A3 smoke | `--no-symbiosis-use-context-exchange` | 160,600,324 | 0.969795  | 0.231776 | 0.351352 | 0.088133 | 0.263219 | 24s  |
| A4 smoke | `--no-symbiosis-use-multi-scale`      | 160,824,772 | 0.972732  | 0.231006 | 0.328535 | 0.068924 | 0.259612 | 68s  |
| A5 smoke | `--symbiosis-use-domain-gate`         | 160,824,772 | 0.976064  | 0.228063 | 0.329349 | 0.068870 | 0.260479 | 71s  |

这些 L0 数字有两个直接用途：第一，确认消融配置能进入 `train_config.json` 并完成 checkpoint 保存；第二，暴露初步工程成本，例如 A3 明显更快、A5 在 CPU smoke 上更慢。补跑增强后的 `taac-evaluate single` 后，所有 run 的 95% CI 都与 B1 baseline 重叠，因此 AUC 差异不解读。下一步必须换成 `data_diagnostics` 判定为 L1-ready 的固定验证切片，再做正式比较。

## 当前结论

当前 Symbiosis 有价值，但价值还没有被严格拆解。它已经提供了一个融合 user、item、序列、时间和 action prompt 的统一实验平台。下一步不是否定它，而是把它变成一套可被证伪的实验体系：哪些模块贡献 AUC，哪些模块只是复杂，哪些 DeepSeek-V4 或 TAAC 2025 启发能在 2026 AUC 契约下真正成立，都必须由实验结果决定。
