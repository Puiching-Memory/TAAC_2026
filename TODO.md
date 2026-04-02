# TAAC 2026 路线图与待办

## 总目标

围绕“统一序列建模与特征交互”这一主题，构建一个满足时延约束、具备清晰技术叙事、并且能持续提升 AUC 的单模型系统。

当前仓库现在有两层工作面：

1. 主线模型仍是 Grok 风格 unified transformer baseline。
2. 工程上已经重新接入多套公开方案复现版本，用于统一对比和吸收归纳偏置。

## 当前判断

1. 同口径正式对比已经完成：样本集当前最强是 creatorwyx_din_adapter，AUC 0.7769。
2. retrieval-style 方案整体位于第二梯队，说明 pairwise / ranking bias 在当前样本集上有效。
3. 统一建模路线里 UniRec 系当前最强，但新增的 `unirec_din_readout` 只把 AUC 从 0.6761 推到 0.6780，同时显著拉高了时延，说明简单叠加 post-transformer readout 性价比很低。
4. grouped DIN 没有超过单路 DIN，说明“先分组再融合”不是充分条件。
5. Grok unified backbone 上新增的 grok_din_readout 已将 AUC 从 0.6155 提升到 0.6457，说明 target-aware readout 确实补中了主缺口，但当前只是第一步。
6. 现在更关键的问题已经不是“要不要加 readout”，而是“把 target-aware bias 放在 transformer 前、后，还是只保留其中一种”。
7. 小样本结果只能作为方向参考，因此不能因为 E002 最强就直接放弃 unified 路线；更合理的是继续比较 readout 与 unified backbone 的耦合方式。
8. 多指标回填后，creatorwyx_grouped_din_adapter 的 PR-AUC 略高于单路 DIN，但 AUC 和时延仍差，说明 grouped branch 目前更像 head positive trick，而不是更强主线。
9. 当前 sample 验证集里 user 几乎全唯一，GAUC 全部退化成 0.5 且 valid_group_count 为 0，因此当前 GAUC 只能算接口就绪，不能算结论。
10. 在 grok_din_readout 上做完 128/256/384 的三 seed truncation sweep 后，128 取得最佳均值 AUC / PR-AUC 和最低延迟，说明更长历史当前没有稳定收益。

## 已完成基础设施

1. 训练入口已经支持 baseline、DIN、SASRec/retrieval-style、DeepContextNet、UniRec、UniScaleFormer 多种配置。
2. 当前模型实现已扩展为 `taac2026/models/` 下的多文件模型仓。
3. 特征分析逻辑已经从单脚本迁移到 `taac2026/feature_engineering/` 子模块。
4. schema 分析现在默认同时导出摘要和 feature_id 字典。
5. 已完成 10 个配置的同口径 smoke test 与正式训练对比，结果已写入 EXPERIMENTS.md。
6. 已完成 Grok unified backbone 的 post-transformer DIN readout 变体 `grok_din_readout`，并完成首轮正式训练验证。
7. 已完成 UniRec 上的 stacked target-aware 变体 `unirec_din_readout`，并验证“interest token + post-transformer readout”简单叠加收益有限。
8. 训练评估现在会同时写出 AUC、PR-AUC、Brier、logloss，并补充按序列长度、行为密度、时间窗口、user/item 频次分桶的验证结果。
9. 评估现在已支持 bootstrap AUC/PR-AUC 置信区间与 user-level GAUC 统计。
10. 已新增批量回填脚本 `taac2026.batch_evaluate`，可为当前 12 个活跃实验统一写出 evaluation.json 和总表。
11. 已完成 `grok_din_readout` 的 128/256/384 三 seed truncation sweep，并得到 sample 上 `max_seq_len=128` 的最佳 Pareto 结果。
12. 已新增独立的数据分析 CLI，可输出序列长度、截断率、时间漂移和 user/item 冷热分布统计。
13. 已新增基于 matplotlib 的统一可视化 CLI，可直接消费 evaluation、batch report、summary、truncation sweep 和 dataset profile 产物并导出 PNG。

## 模型主线待办

按优先级排序：

1. 在 Grok unified 主线上比较 post-transformer DIN readout 与 pre-transformer interest token 两种结合方式。
2. 在 UniRec 主线上比较 interest token only、post-transformer readout only、stacked both 三种 target-aware 耦合方式。
3. 给高基数实体补多哈希 typed embedding，优先覆盖 user、target item、history item。
4. 把当前 `max_seq_len=128` 的结论扩展到 pre-transformer interest token 和 UniRec 主线，而不是只在 `grok_din_readout` 上成立。
5. 试验 dense feature 注入方式，比较 dense token、candidate fusion 和低秩投影。
6. 把现有 user/item 热度、GAUC 和多 seed CI 搬到更大样本或正式数据上，因为当前 sample 支撑度不足。
7. 如果继续保留 retrieval 方向，对 zcyeee 与 O_o 做真正的 scheme-specific 适配，而不是共用同一核心结构。

## 特征工程主线待办

按优先级排序：

1. 扩展 `feature_dictionary.json`，增加字段分组建议、疑似时间字段、长数组字段和高频字段清单。
2. 输出 user_feature、item_feature、sequence 三部分的字段覆盖率报告，识别稀有字段和稳定字段。
3. 针对 sequence 中不同 group 的 feature_id 做共现统计，为 typed embedding 和 token 保留策略提供证据。
4. 设计“字段进入模型前检查表”，新模型改输入表示前必须先完成一次分析。
5. 当正式数据到位后，补齐更大样本上的字段分布与时间漂移分析。

## 实验方法约束

1. 尽量把“输入表示变化”和“骨干结构变化”拆开验证。
2. 每次实验都记录配置、指标、时延、结论和是否可直接比较。
3. 如果小样本结果和结构直觉冲突，先回到字段分析确认，不要直接过拟合小样本。
4. 如果缺乏新想法，优先补数据分析和查阅相关公开工作，而不是盲目堆模块。

## 近期执行顺序

1. 以 creatorwyx_din_adapter 作为样本集 accuracy 上界对照，不再重复做无目标的横向大扫表。
2. 在 Grok unified 主线上继续比较 post-transformer DIN readout 与 pre-transformer interest token。
3. 在 UniRec 主线上比较 interest token only、post-transformer readout only、stacked both 三种 target-aware 耦合方式。
4. 以 `max_seq_len=128` 作为当前 unified + readout 默认长度，再进入 typed embedding 消融。

## 结论

当前最可靠的推进方式不是继续扩大模型复杂度，而是让“字段理解 → 输入表示 → 主干建模”三者形成稳定闭环。模型主线和特征工程主线必须并行推进，缺一不可。

重点优化方向：

1. target-aware readout
2. typed embedding
3. token pruning
4. sequence truncation policy
5. embedding table 规模控制
6. block 数与 hidden size 联合搜索

## 模型搜索优先级

### 优先做

1. 候选感知 attention 结构
2. 事件表示中的时间差编码
3. 长序列压缩与截断策略
4. user/context/item feature 的统一编码接口
5. 与时延相关的 token 数控制

### 暂缓做

1. 复杂 MoE
2. 大规模蒸馏链路
3. 多模型融合
4. 纯论文导向但部署复杂的 exotic 模块

原因是这些方向工程成本高，且不一定适配比赛规则。

## 消融实验清单

每轮实验至少记录以下维度：

1. AUC
2. 参数量
3. token 总数
4. 单样本推理延迟
5. 吞吐量
6. 显存占用
7. 训练稳定性

优先消融：

1. 历史长度 64、128、256、384
2. 是否加入 time gap encoding
3. 是否加入 typed embedding
4. transformer 层数
5. static prefix 的 token 预算
6. 不同 feature group 的注入方式

## 指标体系

公开主指标是 AUC，但内部必须维护多维指标。

建议看板：

1. 主验证 AUC
2. 不同行为频次用户分桶 AUC
3. 不同序列长度分桶 AUC
4. 不同时间窗口分桶 AUC
5. 推理 P50、P95 延迟
6. 每秒样本吞吐

如果没有分桶评估，很容易出现整体 AUC 提升但关键长序列场景退化的问题。

## 风险点

### 风险 1：样例数据太小导致结构判断失真

公开样例只有 1000 条，不能据此对复杂架构做强结论。

应对：

1. 先把框架搭好
2. 小样本只做功能验证
3. 正式数据到位后再做结构优选

### 风险 2：统一建模带来 token 爆炸

如果把所有 feature 都变成 token，复杂度可能不可接受。

应对：

1. 区分核心 token 与辅助 token
2. 对低价值字段做 pooling 或压缩
3. 对 float_array 类特征先投影后注入

### 风险 3：延迟限制后期卡死

如果一开始忽视延迟，后期往往要推翻重来。

应对：

1. 训练阶段就记录延迟
2. 每个版本同步评估参数量和 token 数
3. 主干设计优先选择结构上可裁剪的模块

### 风险 4：实验空间过大，试验无序

应对：

1. 只围绕三个主假设推进
2. 采用固定模板记录实验
3. 每周只保留少数重点分支

## 冠军级实施节奏

### 第 1 周

1. 完成数据解析与 schema 抽象
2. 完成最小训练闭环
3. 搭建时间切分验证
4. 产出第一个 unified baseline

### 第 2 周

1. 跑第一轮 unified + target-aware readout 关键消融
2. 确定事件表示方案
3. 确定历史长度基线
4. 确定 typed embedding 方案

### 第 3 周

1. 引入长序列压缩模块或更强截断策略
2. 建立延迟与 AUC 联合评估表
3. 固化当前最优主线

### 第 4 周及以后

1. 围绕主线结构做精细搜索
2. 开始写技术报告骨架
3. 同步整理 scaling law 实验
4. 为创新奖准备方法论叙事

## 创新奖叙事建议

如果目标不仅是冲榜，还要冲创新奖，建议围绕两个主张展开。

### 统一模块创新奖

主张：

1. 用统一 token 化和统一骨干，将序列建模与多字段特征交互合并到一个 stack 内。
2. 用 candidate token 读 prefix 的方式替代传统晚融合或外接 readout 设计。

### 缩放规律创新奖

主张：

1. 分析模型容量、历史长度、memory token 数、特征 token 数与 AUC 的关系。
2. 给出时延约束下的最优 scaling 区间，而不是只报单点最好结果。

## 近期立即执行事项

按优先级排序：

1. 扩展 schema 分析脚本，导出 feature dictionary，标记各 feature_id 的类型、频次、数组长度和疑似时间字段。
2. 在 Grok unified 主线上补一版 pre-transformer interest token，与 `grok_din_readout` 做同口径比较。
3. 在 UniRec 主线上补一版 post-transformer readout only 变体，与 `unirec` 和 `unirec_din_readout` 做三路对照。
4. 把现有 user/item 热度、GAUC 和 seed 置信区间迁移到更大样本或正式数据上验证。
5. 在 `max_seq_len=128` 的 unified 主线上继续做 typed embedding 和 dense 注入方式对比。
6. 在保持当前主线可训练的前提下，继续跟踪 P50、P95 延迟和参数量变化。

## 结论

主线不应是“做一个尽可能大的模型”，而应是“做一个单模型、统一、候选感知、长序列可压缩、时延友好”的系统。

如果这条路线走通，这个方案既有冲排行榜的现实可能，也有写出有辨识度技术报告的空间。
