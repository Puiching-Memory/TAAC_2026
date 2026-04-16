# 数据集 EDA 报告

> 本报告由 `uv run taac-dataset-eda` 自动生成的 ECharts JSON 驱动。
> 如需更新，重新运行 CLI 即可刷新 `docs/assets/figures/eda/` 下的图表数据。

---

## 1. 数据集概况

基于 `TAAC2026/data_sample_1000`（1000 条采样数据）的分析结果。下文各节通过 ECharts 交互图表逐项展开。

!!! warning "采样局限性"
    以下统计基于 **1000 行采样**，仅适用于 schema 验证和初步画像。涉及分布形态、长尾特征、域优先级等结论在全量数据上可能发生变化，标注 ⚠️ 的结论需在更大样本上复核。

---

## 2. 列布局概览

<div class="echarts" data-src="assets/figures/eda/column_layout.echarts.json"></div>

共 **120** 列：标量 5、用户整型 46、用户稠密 10、物品整型 14、域序列共 45（分布于 4 个行为域）。

- **格式**：扁平 120 列 parquet
- **行为域**：`domain_a/b/c/d` 四个独立域（45 列）
- **用户特征**：共 56 个（46 整型 + 10 稠密）

---

## 3. 行为类型分布

<div class="echarts" data-src="assets/figures/eda/label_distribution.echarts.json"></div>

**关键发现**：

- 当前采样数据中只有 `label_type=1`（点击）和 `label_type=2`（转化）两种类型
- 点击占约 87.6%，转化约 12.4%
- 样本仅包含正向行为（点击/转化），不含曝光（`label_type=0`）
- **任务口径**：训练以 `label_action_type=2`（转化）为正样本，其余为负样本，因此离线任务本质是**点击样本中的转化预测（CVR）**，而非曝光级 CTR

---

## 4. 特征缺失率

<div class="echarts" data-src="assets/figures/eda/null_rates.echarts.json"></div>

**关键发现**：

- `user_int_feats_100`–`103`、`109` 缺失率超过 **84%**，这些高编号用户特征可能是稀缺标签
- `item_int_feats_83/84/85` 缺失率约 **83%**，可能对应多模态编码特征
- 约 30 个特征列缺失率 > 10%，需要专门的缺失值嵌入（learned missing token）策略

---

## 5. 稀疏特征基数

<div class="echarts" data-src="assets/figures/eda/cardinality.echarts.json"></div>

**关键发现**：

- 最高基数：`item_int_feats_11`（924）和 `item_int_feats_16`（662），但在 1000 行采样中基数有限，完整数据预计会显著增长
- 用户特征中 `user_int_feats_66`（533）和 `user_int_feats_54`（462）基数较高
- ⚠️ 分 user / item 两组的基数分布在采样中相对均匀，全量数据的长尾分布待确认
- 基数 < 10 的特征可直接嵌入；基数极高的特征需要哈希或使用 Semantic ID

---

## 6. 特征覆盖率热力图

<div class="echarts" data-src="assets/figures/eda/coverage_heatmap.echarts.json"></div>

**关键发现**：

- 大部分物品特征覆盖率接近 **100%**（深绿色区域），除了 `item_int_feats_83/84/85`
- 用户特征覆盖率方差较大：核心特征（ID < 60）普遍 > 90%，高编号特征（> 90）覆盖率骤降
- 建议：对覆盖率 < 50% 的特征使用独立的**缺失值嵌入向量**，而非零填充

---

## 7. 序列长度分布

<div class="echarts" data-src="assets/figures/eda/sequence_lengths.echarts.json"></div>

**关键发现**：

- ⚠️ **domain_d** 序列在采样中最长（均值 1099，P95=2451），初步判断为主行为域，需全量验证
- **domain_c** 序列最短（均值 449），但几乎无空序列（0.2%）
- **domain_d** 有 8% 空序列率，需要在模型中处理缺失域
- 所有域均呈**右偏分布**：少量用户有极长序列（> 2000），多数用户在 200–800 范围
- ⚠️ domain_d 最长域已超 1000，**长序列建模**可能是核心挑战（待全量确认）

<div class="echarts" data-src="assets/figures/eda/seq_length_summary.echarts.json"></div>

### 架构启示

1. ⚠️ **domain_d 序列极长**（采样均值 > 1000）：若全量数据确认此趋势，直接全量自注意力的复杂度不可控，需要分桶、滑动窗口或稀疏注意力。
2. ⚠️ **域间长度差异大**：统一截断长度可能不合理，建议在全量数据上验证后按域设置独立的 `max_seq_len`。

---

## 8. 用户维度分析

### 8.1 用户活跃度分布

<div class="echarts" data-src="assets/figures/eda/user_activity.echarts.json"></div>

**关键发现**：

- 用户活跃度（每用户行为条数）的分布是否符合 power law？
- 冷启动用户（仅 1 条行为）占比直接影响 GAUC coverage
- ⚠️ 采样数据量有限，全量验证后才能判定高/中/低活跃度的实际分桶阈值

### 8.2 跨域用户重叠

<div class="echarts" data-src="assets/figures/eda/cross_domain_overlap.echarts.json"></div>

**关键发现**：

- 各 domain 之间用户重叠率决定了多域融合策略的可行性
- 若某域用户几乎是另一域的子集，可考虑域间迁移学习
- 活跃在所有 4 个域的用户占比 → 决定是否值得做全域联合建模

---

## 9. 特征有效性分析（Label-Conditional）

### 9.1 单特征 AUC 排名

<div class="echarts" data-src="assets/figures/eda/feature_auc.echarts.json"></div>

**关键发现**：

- 单特征 AUC > 0.55 的特征是建模的**核心信号**，应优先保障 embedding 质量
- AUC ≈ 0.5 的特征为噪声，可考虑在 lightweight 模型中丢弃以降低参数量
- ⚠️ 采样数据上的单特征 AUC 可能不稳定，全量数据上需复核

### 9.2 正负样本缺失率对比

<div class="echarts" data-src="assets/figures/eda/null_rate_by_label.echarts.json"></div>

**关键发现**：

- 若某特征在正样本（转化）中缺失率显著低于负样本 → 可能存在**标签泄露**，需排查
- 缺失率差异大的特征本身具有预测能力，learned missing token 可捕获这类信号
- 差异小的特征缺失为随机缺失 (MCAR)，填充策略影响不大

---

## 10. 缺失值模式分析

### 10.1 特征共缺失模式

<div class="echarts" data-src="assets/figures/eda/co_missing.echarts.json"></div>

**关键发现**：

- 高频共缺失对暗示这些特征来自**同一数据源**，可作为一组统一处理
- 共缺失率极高（> 0.8）的特征组可合并为单个 "是否缺失" flag 特征
- 共缺失模式有助于诊断是 MCAR 还是 MNAR（结构性缺失）

---

## 11. 稠密特征分布

<div class="echarts" data-src="assets/figures/eda/dense_distributions.echarts.json"></div>

**关键发现**：

- `user_dense_feats_*` 的分布形态决定预处理策略：
    - 若均值接近 0、标准差接近 1 → 可能是预提取 embedding，**不要再做标准化**
    - 若分布高度偏斜 → 需要 log 变换或分桶离散化
    - 若大量零值 → 可能是稀疏表示，考虑用 embedding 替代 dense 输入

---

## 12. 特征基数区间分布

<div class="echarts" data-src="assets/figures/eda/cardinality_bins.echarts.json"></div>

**关键发现**：

- 各基数区间的特征数量指导 embedding table 设计：
    - 基数 1-10：直接嵌入，低维（4-8 维）
    - 基数 11-1K：标准嵌入（16-64 维）
    - 基数 1K+：需哈希或 Semantic ID，否则 embedding table 过大
- ⚠️ 采样中基数被严重低估，全量数据上需重新评估

---

## 13. 序列行为模式

### 13.1 序列内物品重复率

<div class="echarts" data-src="assets/figures/eda/seq_repeat_rate.echarts.json"></div>

**关键发现**：

- 高重复率（> 0.3）意味着用户反复与同类物品交互 → 序列去重或 position-aware attention 可能有效
- 低重复率表明行为多样性高 → 序列完整保留的信息增益更大
- 各域重复率差异 → 可能需要域特定的序列处理策略

---

## 14. 多模态嵌入分析

本届 `item_int_feats_83/84/85` 的覆盖率约 17%，暗示多模态数据仍为稀疏覆盖。

### 待深入分析项

- [ ] 确认 `item_int_feats_83/84/85` 是否为多模态嵌入 ID
- [ ] 分析 `user_dense_feats_*` 的分布形态（是否为预提取嵌入向量）
- [ ] 如有嵌入向量，计算跨模态相关性矩阵

---

## 15. 暂无法自动化的分析项（待全量数据）

以下分析需要全量数据或外部工具支撑，在采样数据上不具备可靠性：

### 15.1 时序分析（需 timestamp + 全量数据）

- [ ] **训练/验证集时间切分点**：按 `timestamp` 排序后确定 split point，对比切分前后的标签率、用户/物品 overlap
- [ ] **日内 CVR 曲线**：按小时分桶统计转化率，识别 day-parting 模式
- [ ] **冷启动率**：验证集中首次出现的 user_id / item_id 占比
- [ ] **特征覆盖率时间漂移**：前 N 天 vs 后 N 天的特征缺失率变化趋势
- [ ] **行为时间衰减**：序列中最近 N 天的行为对预测的增益曲线（指导截断策略）

### 15.2 特征交互分析（需全量数据 + 计算资源）

- [ ] **特征交叉 AUC**：两两特征组合后的 AUC 提升（发现强交互对）
- [ ] **特征冗余检测**：高相关特征对识别（Cramér's V / mutual information）
- [ ] **嵌入维度经验公式验证**：按基数分组，验证 $d = \min(50, \lceil c^{0.25} \rceil)$ 的合理性

### 15.3 序列深度分析（需全量序列数据）

- [ ] **序列内行为类型转移矩阵**：点击→转化 vs 点击→流失的转移概率
- [ ] **序列时间间隔分布**：用于 session 切分决策
- [ ] **序列尾部 N 步 vs 全序列的 AUC 差异**：定量确定截断长度
- [ ] **序列长度 vs CVR 关系曲线**：判断长序列是否确实带来增益

### 15.4 模型 Error Analysis（需已训练模型）

- [ ] **分群 AUC 切片**：按用户活跃度、物品热度、序列长度、缺失率分桶，输出各桶 AUC/PR-AUC/GAUC
- [ ] **Brier vs AUC scatter**：判断瓶颈是排序能力还是校准能力
- [ ] **GAUC coverage 分析**：哪些用户群被模型放弃？
- [ ] **Domain 贡献度消融**：逐域 mask 后的 AUC 差异

### 15.5 竞赛策略定量验证

- [ ] **上届方案适用性验证**：全量数据上验证序列长度分布后，定量评估全序列 attention 的复杂度拐点
- [ ] **负样本构造 ROI**：当前无曝光数据，需评估 random negative 的采样比对 AUC 的影响
- [ ] **Semantic ID 覆盖率 ROI**：`item_int_feats_83/84/85` 17% 覆盖率下 Semantic ID 的投入产出比

---

## 16. 重新生成报告

```bash
# 完整重跑 EDA（自动下载 sample 数据集）
uv run taac-dataset-eda

# 指定数据集路径
uv run taac-dataset-eda --dataset data/my_dataset.parquet

# 限制扫描行数加速
uv run taac-dataset-eda --max-rows 5000

# 同时输出 JSON 格式统计（供其他工具消费）
uv run taac-dataset-eda --json-path docs/assets/figures/eda/stats.json
```

输出产物位于 `docs/assets/figures/eda/`（ECharts JSON 文件）。
