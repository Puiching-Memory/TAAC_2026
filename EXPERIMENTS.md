# 实验记录

## 用途

本文件用于统一记录本地实验，确保模型改动、指标和结论可以持续对比，而不是散落在终端日志或临时备注里。

## 记录规则

1. 每个配置运行单独占一行。
2. 至少记录模型名、核心结构改动、最佳验证 AUC 与时延。
3. 每组实验后补一条简短结论。
4. 如果实验之间不可直接比较，需要说明原因。

## 当前主线

当前主线配置：configs/grok_din_readout.yaml

当前主线模型：grok_din_readout

当前状态：已完成 12 个结构方案的多指标统一回填，并在 unified + target-aware 主线上新增一组 max_seq_len 128/256/384 的三 seed truncation 消融。

本轮直接可比说明：

1. E001-E012 全部使用同一个 sample_data.parquet、相同时间排序切分、相同训练轮数与同一训练入口，且均为单 seed 42，可直接做结构横向比较。
2. E013-E015 全部基于 configs/grok_din_readout.yaml，仅改变 max_seq_len，并统一使用 seeds=42/43/44 做聚合，三者之间可直接比较，但不应与 E001-E012 的单 seed 排名表混成一个“谁最好”的结论。

## 活跃实验表

| 编号 | 配置                                        | 模型                           | 核心改动                                                                    |   参数量 | 验证 AUC | 验证 PR-AUC |  Brier | 平均时延（毫秒/样本） | P95 时延（毫秒/样本） | 结论                                                           |
| ---- | ------------------------------------------- | ------------------------------ | --------------------------------------------------------------------------- | -------: | -------: | ----------: | -----: | --------------------: | --------------------: | -------------------------------------------------------------- |
| E001 | configs/baseline.yaml                       | grok_baseline                  | Grok 风格 unified transformer + candidate isolation mask + 分解历史事件编码 | 21061345 |   0.6155 |      0.2078 | 0.2708 |                0.3017 |                0.7562 | unified 主线可训练，但排序与校准都明显弱于 target-aware 系方案 |
| E002 | configs/creatorwyx_din_adapter.yaml         | creatorwyx_din_adapter         | 单路 DIN 目标注意力读 history                                               | 19501731 |   0.7769 |      0.4858 | 0.1104 |                0.1572 |                0.5240 | 当前样本集最强，且延迟也最有竞争力                             |
| E003 | configs/creatorwyx_grouped_din_adapter.yaml | creatorwyx_grouped_din_adapter | action/content/item 三路分组 DIN 后 gating 融合                             | 19668586 |   0.7446 |      0.4904 | 0.1111 |                0.1816 |                0.5718 | PR-AUC 略高于单路 DIN，但 AUC 和时延仍更差                     |
| E004 | configs/tencent_sasrec_adapter.yaml         | tencent_sasrec_adapter         | SASRec 风格 causal history encoder + candidate-aware pooling                | 19672993 |   0.6934 |      0.3450 | 0.3090 |                0.1631 |                0.4276 | 纯序列编码是合格强基线，但校准偏弱                             |
| E005 | configs/zcyeee_retrieval_adapter.yaml       | zcyeee_retrieval_adapter       | retrieval-style 多摘要读出 + BCE/pairwise 组合损失                          | 19969452 |   0.7504 |      0.3925 | 0.1277 |                0.1781 |                0.4499 | retrieval / ranking 混合归纳偏置很强，且校准优于 OmniGenRec    |
| E006 | configs/oo_retrieval_adapter.yaml           | o_o_retrieval_adapter          | retrieval-style 多摘要读出 + BCE/pairwise 组合损失                          | 19969452 |   0.7504 |      0.3925 | 0.1277 |                0.2652 |                0.7388 | 与 E005 同核同分，当前差异主要体现在推理时延                   |
| E007 | configs/omnigenrec_adapter.yaml             | omnigenrec_adapter             | retrieval-style 结构 + Muon/AdamW 混合优化 + combined AUC loss              | 20264940 |   0.7185 |      0.3907 | 0.3086 |                0.2668 |                0.7434 | AUC 尚可，但 logloss/Brier 明显差，校准不如更简单的 retrieval  |
| E008 | configs/deep_context_net.yaml               | deep_context_net               | CLS/global context unified stack                                            | 21135649 |   0.5825 |      0.2007 | 0.2605 |                0.2788 |                0.7178 | 当前 pure global context 适配在样本集上最弱                    |
| E009 | configs/unirec.yaml                         | unirec                         | unified tokenizer + feature cross + interest token + unified stack          | 21802275 |   0.6761 |      0.2948 | 0.1367 |                0.3094 |                0.5836 | unified 方案里当前最强，但距离 DIN / retrieval 仍有明显差距    |
| E010 | configs/uniscaleformer.yaml                 | uniscaleformer                 | memory-compressed history + candidate cross-attention                       | 20912929 |   0.6286 |      0.2067 | 0.2619 |                0.1773 |                0.4675 | 压缩思路有时延优势，但排序与校准都没有补齐                     |
| E011 | configs/grok_din_readout.yaml               | grok_din_readout               | Grok unified backbone + post-transformer DIN-style target-aware readout     | 21431523 |   0.6457 |      0.2299 | 0.2293 |                0.3445 |                0.8183 | unified 主线得到明确增益，但 readout 位置仍未达到最优          |
| E012 | configs/unirec_din_readout.yaml             | unirec_din_readout             | UniRec feature cross + interest token + post-transformer DIN readout        | 22098341 |   0.6780 |      0.2539 | 0.2321 |                0.3085 |                0.5492 | AUC 略高于 E009，但 PR-AUC 下降，简单叠加性价比不高            |

## 多 seed 截断消融表

| 编号 | 基础配置                      | max_seq_len | seeds    |  AUC mean/std | AUC 95% CI       | PR-AUC mean/std | 平均时延 mean | P95 时延 mean | 结论                                                 |
| ---- | ----------------------------- | ----------: | -------- | ------------: | ---------------- | --------------: | ------------: | ------------: | ---------------------------------------------------- |
| E013 | configs/grok_din_readout.yaml |         128 | 42,43,44 | 0.6453/0.0330 | [0.6080, 0.6827] |   0.2401/0.0224 |        0.2585 |        0.6874 | 当前 sample 上最优 Pareto 点，AUC 均值最高且时延最低 |
| E014 | configs/grok_din_readout.yaml |         256 | 42,43,44 | 0.6348/0.0420 | [0.5874, 0.6823] |   0.2273/0.0332 |        0.3090 |        0.5972 | 比 128 更慢且均值更低，没有看到稳定增益              |
| E015 | configs/grok_din_readout.yaml |         384 | 42,43,44 | 0.6407/0.0417 | [0.5935, 0.6879] |   0.2357/0.0331 |        0.3923 |        0.6608 | 单次最高分可见，但均值不优于 128，时延代价显著更大   |

## 当前结论

1. 在当前 sample_data.parquet 上，E002 仍是最强方案，不仅 AUC 最高，Brier 和 logloss 也明显优于大多数对照，说明单路 target-aware history readout 仍是当前最强归纳偏置。
2. E003 的 PR-AUC 略高于 E002，但 AUC、平均时延和 P95 时延都更差，说明 grouped history 可能对头部正样本有帮助，但现有融合不足以转化成整体排序收益。
3. E005/E006 继续稳居第二梯队，且 E005 的校准明显好于 E007，说明 retrieval / ranking bias 有用，但更复杂的优化器与组合损失并没有自动带来更好的 calibrated probability。
4. E011 相比 E001 的 AUC 从 0.6155 提升到 0.6457，PR-AUC 从 0.2078 提升到 0.2299，说明 Grok unified backbone 确实缺 target-aware readout；但时延也同步恶化，readout 放置方式仍需继续优化。
5. E012 只比 E009 提升了 0.0019 AUC，但 PR-AUC 从 0.2948 降到 0.2539，说明在已有 interest token 的 UniRec 上继续叠加后置 readout 并不是稳定的正收益，stacking 不应默认保留。
6. 这轮多指标回填显示，单看最终 AUC 会漏掉一些关键信息，例如 E003 对 E002 的 PR-AUC 反超，以及 E007 相比 E005/E006 的明显校准劣化。
7. 当前 sample 验证集上的 user 几乎全唯一，GAUC 统计里 valid_group_count 为 0，因此所有模型的 GAUC 都退化成 0.5；这说明 GAUC 代码已接好，但样例集本身并不能支持 user-level 归因分析。
8. item 热度分桶已经可用，但当前 sample 验证集里 item 仍以 one-off 为主，例如 E002 的验证集中 92.5% 样本落在 item_frequency=1 桶，因此热点分层结论目前仍只能作方向性参考。
9. 在 unified + target-aware 主线的三 seed truncation sweep 中，E013 的 max_seq_len=128 拿到最高 AUC 均值 0.6453 和最高 PR-AUC 均值 0.2401，同时平均时延最低 0.2585 ms/样本，是当前样例集上的最佳 Pareto 点。
10. E014 和 E015 都没有相对 E013 展现稳定收益；其中 E015 虽然出现过单次最高 AUC 0.6790，但三 seed 均值仅 0.6407，且平均时延升到 0.3923 ms/样本，说明更长历史当前主要在放大方差和时延成本，而不是稳定增益。
11. 因此，当前关键问题已经从“要不要加 readout”和“要不要拉更长历史”转成“target-aware bias 应该放在 transformer 前还是后”和“128 长度下如何提升输入表示质量”。
12. 小样本结果仍然只能作为方向参考，因此不能因为 E002 最强就直接放弃 unified 路线；更合理的推进方式是继续比较耦合位置、typed embedding 和更强输入表示，而不是继续无差别堆长序列或堆模块。

## 历史归档结论

这些结论对应的代码分支已经从当前仓库移除，但结论本身仍然保留：

1. 最早的 candidate-aware pooling baseline 能跑通完整 parquet 训练闭环，说明数据与评估流程本身没有问题。
2. UCASIM 风格的多层 candidate-aware 交互块在样本数据上明显不稳，复杂骨干不应优先于输入表示与长历史处理。
3. 分解事件编码曾显著优于扁平历史 token，说明 history 的结构化表示是有效方向。
4. DIN 风格目标注意力曾达到样本集上的最好结果，说明 target-aware history readout 是强归纳偏置。
5. 分组历史本身不是充分条件，history 分支如何融合比“是否分组”更关键。

## 详细记录

### E001

- 配置：configs/baseline.yaml
- 输出目录：outputs/baseline/summary.json
- 核心改动：将当前主线收敛成单一 Grok 风格 ranker，使用 static prefix、causal history 和 candidate isolation mask。
- 最佳验证 AUC：0.6155
- 平均时延（毫秒/样本）：0.3017
- P95 时延（毫秒/样本）：0.7562
- 观察：多指标回填后，PR-AUC 只有 0.2078，Brier 0.2708，说明 baseline 不只是排序弱，概率质量也偏弱。
- 结论：这版 baseline 适合作为统一主线代码继续迭代，但不能把样本集结果误判为最终结构优胜。

### E002

- 配置：configs/creatorwyx_din_adapter.yaml
- 输出目录：outputs/creatorwyx_din_adapter/summary.json
- 核心改动：单路 DIN 目标注意力从 history 中做 target-aware readout。
- 最佳验证 AUC：0.7769
- 平均时延（毫秒/样本）：0.1572
- P95 时延（毫秒/样本）：0.5240
- 观察：本轮最强 AUC，PR-AUC 也达到 0.4858，且 Brier 只有 0.1104，说明 target-aware readout 在样本集上不仅排序强，校准也明显更好。
- 结论：当前样本集上的 accuracy 上界参考应切换为 creatorwyx_din_adapter。

### E003

- 配置：configs/creatorwyx_grouped_din_adapter.yaml
- 输出目录：outputs/creatorwyx_grouped_din_adapter/summary.json
- 核心改动：action/content/item 三个 group 分别做 DIN 汇聚后再 gating 融合。
- 最佳验证 AUC：0.7446
- 平均时延（毫秒/样本）：0.1816
- P95 时延（毫秒/样本）：0.5718
- 观察：虽然比单路 DIN 低 0.0323 AUC，但 PR-AUC 反而略高，说明 grouped route 可能更偏向 head positives，而不是更好的全局排序。
- 结论：先保留单路 DIN 作为更稳的强对照，分组历史需要更强融合才值得继续。

### E004

- 配置：configs/tencent_sasrec_adapter.yaml
- 输出目录：outputs/tencent_sasrec_adapter/summary.json
- 核心改动：SASRec 风格 causal history encoder 配合 candidate-aware pooling。
- 最佳验证 AUC：0.6934
- 平均时延（毫秒/样本）：0.1631
- P95 时延（毫秒/样本）：0.4276
- 观察：在较低时延下拿到中上水平结果，但 Brier 和 logloss 偏高，说明纯序列编码虽能排序，但概率刻画不够稳。
- 结论：没有显式 target-aware readout 的 causal encoder 还不够强。

### E005

- 配置：configs/zcyeee_retrieval_adapter.yaml
- 输出目录：outputs/zcyeee_retrieval_adapter/summary.json
- 核心改动：retrieval-style 多摘要读出 + BCE/pairwise 组合损失。
- 最佳验证 AUC：0.7504
- 平均时延（毫秒/样本）：0.1781
- P95 时延（毫秒/样本）：0.4499
- 观察：pairwise / ranking bias 很有效，整体仅次于 DIN 系方案，且 Brier 0.1277、logloss 0.4256，校准明显优于 OmniGenRec。
- 结论：如果目标是样本集冲分，retrieval-style 是当前第二梯队主力。

### E006

- 配置：configs/oo_retrieval_adapter.yaml
- 输出目录：outputs/oo_retrieval_adapter/summary.json
- 核心改动：retrieval-style 多摘要读出 + BCE/pairwise 组合损失。
- 最佳验证 AUC：0.7504
- 平均时延（毫秒/样本）：0.2652
- P95 时延（毫秒/样本）：0.7388
- 观察：当前实现与 E005 共用同一核心结构，因此 AUC 与 PR-AUC 完全一致，当前主要差异仍体现在推理时延。
- 结论：如果要更忠实地区分 zcyeee 与 O_o 方案，需要继续做 scheme-specific 适配，而不只是 lineage 命名区分。

### E007

- 配置：configs/omnigenrec_adapter.yaml
- 输出目录：outputs/omnigenrec_adapter/summary.json
- 核心改动：retrieval-style 主体 + Muon/AdamW 混合优化 + combined AUC loss。
- 最佳验证 AUC：0.7185
- 平均时延（毫秒/样本）：0.2668
- P95 时延（毫秒/样本）：0.7434
- 观察：相比 baseline/unified 明显更强，但 Brier 和 logloss 明显差于 E005/E006，说明 loss trick 没有自然带来更好的 calibrated probability。
- 结论：优化器 / loss trick 可以增强已有结构，但不能替代结构主假设本身。

### E008

- 配置：configs/deep_context_net.yaml
- 输出目录：outputs/deep_context_net/summary.json
- 核心改动：CLS/global context unified stack。
- 最佳验证 AUC：0.5825
- 平均时延（毫秒/样本）：0.2788
- P95 时延（毫秒/样本）：0.7178
- 观察：这是本轮最弱方案之一，尾延迟也偏高，说明当前 pure global context 适配没有学到足够强的 candidate-history 关系。
- 结论：DeepContextNet 这版适配不宜作为当前主线继续推进。

### E009

- 配置：configs/unirec.yaml
- 输出目录：outputs/unirec/summary.json
- 核心改动：feature cross + interest token + unified stack。
- 最佳验证 AUC：0.6761
- 平均时延（毫秒/样本）：0.3094
- P95 时延（毫秒/样本）：0.5836
- 观察：这是当前 unified 系方案里最好的一个，但参数量和时延都不小，AUC 仍显著落后于 DIN / retrieval-style。
- 结论：unified 路线仍需显式补 target-aware readout，不能只靠 feature cross 和堆叠骨干。

### E010

- 配置：configs/uniscaleformer.yaml
- 输出目录：outputs/uniscaleformer/summary.json
- 核心改动：memory-compressed history + candidate cross-attention。
- 最佳验证 AUC：0.6286
- 平均时延（毫秒/样本）：0.1773
- P95 时延（毫秒/样本）：0.4675
- 观察：memory 压缩确实带来较好时延，但 AUC、PR-AUC 和 Brier 都没有进入第一梯队，说明单靠压缩不足以补齐 candidate-aware 建模缺口。
- 结论：压缩策略应作为 unified 主线的配角，而不是当前阶段的主角。

### E011

- 配置：configs/grok_din_readout.yaml
- 输出目录：outputs/grok_din_readout/summary.json
- 核心改动：保持 E008 的 Grok unified backbone 不变，只在输出端额外加入一个 post-transformer DIN-style target-aware history readout。
- 最佳验证 AUC：0.6457
- 平均时延（毫秒/样本）：0.3445
- P95 时延（毫秒/样本）：0.8183
- 观察：相比 E001 提升 0.0302 AUC、0.0221 PR-AUC，说明 unified backbone 的确缺 target-aware readout；但延迟恶化也很明显，说明仅靠后置 readout 还不够。
- 结论：这次实验验证了方向，但也限定了上界。下一步不应重复证明“readout 有用”，而应继续比较 readout 与 unified backbone 的耦合方式。

### E012

- 配置：configs/unirec_din_readout.yaml
- 输出目录：outputs/unirec_din_readout/summary.json
- 核心改动：在 E009 的 UniRec 上保留 feature cross 与 pre-transformer interest token，再额外加入一个 post-transformer DIN-style target-aware history readout。
- 最佳验证 AUC：0.6780
- 平均时延（毫秒/样本）：0.3085
- P95 时延（毫秒/样本）：0.5492
- 观察：相比 E009 只提升了 0.0019 AUC，但 PR-AUC 下降 0.0409，说明在已有 interest token 的 UniRec 上继续叠加显式 readout，并不是稳定正收益。
- 结论：UniRec 路线上，“interest token + post-transformer readout”简单堆叠的性价比不高。下一步应比较 replacement 和 stacking，而不是默认两个都保留。

### E013

- 配置：configs/grok_din_readout.yaml（override: max_seq_len=128，seeds=42/43/44）
- 输出目录：outputs/truncation_sweep/grok_din_readout/len_128/
- 核心改动：在 Grok unified + post-transformer DIN readout 主线上，将 history truncation 控制在 128。
- AUC mean/std：0.6453 / 0.0330
- AUC 95% CI：[0.6080, 0.6827]
- PR-AUC mean/std：0.2401 / 0.0224
- 平均时延 mean（毫秒/样本）：0.2585
- P95 时延 mean（毫秒/样本）：0.6874
- 观察：三 seed 下的 AUC 和 PR-AUC 均值都是这组里最优，且时延最低，说明在当前样例集上 128 已经足够承载 target-aware readout 的主要收益。
- 结论：当前 sample 上，max_seq_len=128 是这条 unified 主线最好的 Pareto 点。

### E014

- 配置：configs/grok_din_readout.yaml（override: max_seq_len=256，seeds=42/43/44）
- 输出目录：outputs/truncation_sweep/grok_din_readout/len_256/
- 核心改动：保持同一主线，只把 history truncation 放宽到 256。
- AUC mean/std：0.6348 / 0.0420
- AUC 95% CI：[0.5874, 0.6823]
- PR-AUC mean/std：0.2273 / 0.0332
- 平均时延 mean（毫秒/样本）：0.3090
- P95 时延 mean（毫秒/样本）：0.5972
- 观察：虽然单 seed 里仍会出现不错分数，但三 seed 均值不如 128，且时延更高，说明继续拉长历史并没有转化成稳定收益。
- 结论：当前样例集不支持把 max_seq_len 从 128 拉到 256 作为默认选择。

### E015

- 配置：configs/grok_din_readout.yaml（override: max_seq_len=384，seeds=42/43/44）
- 输出目录：outputs/truncation_sweep/grok_din_readout/len_384/
- 核心改动：保持同一主线，只把 history truncation 放宽到 384。
- AUC mean/std：0.6407 / 0.0417
- AUC 95% CI：[0.5935, 0.6879]
- PR-AUC mean/std：0.2357 / 0.0331
- 平均时延 mean（毫秒/样本）：0.3923
- P95 时延 mean（毫秒/样本）：0.6608
- 观察：会出现单次最优 AUC，但三 seed 均值仍略低于 128，且平均时延进一步恶化，说明更长历史主要在放大方差和延迟成本。
- 结论：如果没有更强的 history selection 或压缩策略，不应把 384 作为当前 unified 主线默认长度。

## 下一步实验

1. 在 Grok unified 主线上补一版 pre-transformer interest token，并与 E011 / E013 做同口径三路比较。
2. 在 UniRec 主线上补一版 post-transformer readout only 变体，与 E009 / E012 做 replacement vs stacking 对照。
3. 以 E013 的 max_seq_len=128 为统一主线默认长度，继续补多哈希 typed embedding，优先覆盖 user、target item 和 history item。
4. 当更大样本或正式数据到位后，重新评估 GAUC 与 item 热度分桶，因为当前 sample 的 user/item 支持度不足以支撑强结论。

## 模板

后续实验请按以下格式追加：

### EXXX

- 配置：
- 输出目录：
- 核心改动：
- 最佳验证 AUC：
- 平均时延（毫秒/样本）：
- P95 时延（毫秒/样本）：
- 观察：
- 结论：
