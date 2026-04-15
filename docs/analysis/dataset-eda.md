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

## 8. 多模态嵌入分析

本届 `item_int_feats_83/84/85` 的覆盖率约 17%，暗示多模态数据仍为稀疏覆盖。

### 待深入分析项

- [ ] 确认 `item_int_feats_83/84/85` 是否为多模态嵌入 ID
- [ ] 分析 `user_dense_feats_*` 的分布形态（是否为预提取嵌入向量）
- [ ] 如有嵌入向量，计算跨模态相关性矩阵

---

## 9. 重新生成报告

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
