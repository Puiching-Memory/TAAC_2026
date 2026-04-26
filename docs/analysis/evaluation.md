---
icon: lucide/clipboard-list
---

# 评估指标分析

本文档解读 TAAC 2026 竞赛评估协议，并结合数据集实测分布提出优化方向。

---

## 1. 竞赛评估指标：AUC

本届竞赛使用单一的 **ROC 曲线下面积（AUC）** 指标对所有团队排名（越高越好）。

$$\text{AUC} = \frac{\sum_{i \in \mathcal{P}} \sum_{j \in \mathcal{N}} \mathbb{1}[\hat{y}_i > \hat{y}_j]}{|\mathcal{P}| \cdot |\mathcal{N}|}$$

其中 $\mathcal{P}$ 为正样本集合，$\mathcal{N}$ 为负样本集合，$\hat{y}$ 为模型预测分数。

### 1.1 AUC 的直觉含义

- AUC = 随机抽取一个正样本和一个负样本，模型给正样本打出更高分的概率
- AUC = 0.5 等价于随机猜测；AUC = 1.0 为完美排序
- AUC 对预测分数的**绝对值不敏感**，只关心正负样本之间的**相对排序**

### 1.2 延迟约束

每次提交还必须在官方评估环境下满足特定于赛道和轮次的**推理延迟限制**——超出延迟预算的提交视为无效。工业赛道的延迟限制比学术赛道更严格。

---

## 2. 数据分布对 AUC 的影响

基于 `taac-dataset-eda` 对 TAAC 2026 sample_1000 的分析：

### 关键发现

1. **本届样本不含曝光**：`label_type` 只出现 1（点击）和 2（转化），说明采样策略已过滤纯曝光行为。离线任务等价于**点击样本中的转化预测（CVR）**。
2. **转化占比 12.4%**，正负样本不平衡程度相对温和。AUC 指标本身对类别不平衡具有鲁棒性，但在极端不平衡下校准仍然重要。
3. **正样本定义**：`label_action_type = 2`（转化）为正，其余为负。框架已通过 `pos_weight`（负/正样本数之比）自动计算类别权重。

---

## 3. AUC 优化策略

### 3.1 损失函数选择

AUC 优化的核心是让正样本得分高于负样本：

1. **Binary Cross-Entropy (BCE)**：最常用的 pointwise 损失，隐式优化 AUC
2. **Pairwise Loss（BPR / Hinge）**：直接优化正负样本对的排序，与 AUC 定义更一致
3. **AUC 近似损失**：直接对 AUC 做可微近似（如 squared hinge surrogate），适合 AUC 是唯一指标的场景

### 3.2 特征工程方向

- **行为类型区分**：显式编码点击 vs 转化，让模型学到转化用户的区分能力
- **序列建模**：利用 domain_a–d 的用户行为序列捕捉时序模式
- **缺失值处理**：高缺失率特征（>50%）需要 learned missing token 策略

### 3.3 模型 pipeline

```
输入特征 → 特征嵌入 → 序列编码（Transformer / GRU）
                                    ↓
                           用户-物品交互层 → 预测分数 → AUC 排名
```

- 重点在于**预测分数的排序质量**，而非绝对概率校准
- 模型复杂度需权衡 AUC 收益和推理延迟约束

---

## 4. 本届指标确认事项

- [x] 评估指标为 AUC（已由官方 README 确认）
- [x] 推理延迟限制（赛道和轮次相关）
- [x] 正样本定义：`label_action_type = 2`（转化）为正，其余为负。当前采样数据不含曝光（`label_type=0`），仅有点击（1）和转化（2），因此离线任务等价于**点击样本中的转化预测（CVR）**
- [ ] 是否有样本加权机制？

---

## 5. 当前代码中的评估指标

当前 PCVR 评估入口由 `PCVRExperiment.evaluate()` 驱动，写入 `evaluation.json` 的指标是：

| 指标 | 来源 | 诊断用途 |
|------|------|----------|
| **AUC** | `binary_auc()` | 整体排序质量（竞赛主指标） |
| **LogLoss** | `binary_logloss()` | 二元交叉熵，用于观察概率输出是否稳定 |
| **sample_count** | 评估样本计数 | 校验评估覆盖样本量 |

`taac2026.domain.metrics` 里还保留了 `compute_classification_metrics()`、`group_auc()` 和 Brier helper，可用于离线分析或后续扩展；但它们不是当前 PCVR evaluate CLI 默认写出的字段。

### 5.1 指标用法

```python
from taac2026.domain.metrics import binary_auc, binary_logloss

auc = binary_auc(labels, probabilities)
logloss = binary_logloss(labels, probabilities)

# PCVR evaluate 当前写出的核心 payload 形态：
# results = {
#     "auc": 0.72,
#     "logloss": 0.54,
#     "sample_count": 1000,
# }
```

### 5.2 为什么需要多个指标

只看 overall AUC 可能会掩盖以下问题，这些适合作为后续离线诊断维度扩展：

- **用户群覆盖不足**：整体 AUC 高但分群覆盖低，说明大量用户分组无法有效排序
- **校准错位**：AUC 高但 LogLoss 或校准误差差，预测概率不可信，影响下游出价
- **正样本检出不足**：AUC 尚可但转化样本 recall 不够

### 5.3 评估最佳实践

- 验证集划分使用**时间窗口切分**（框架默认按 `timestamp` 排序后截取后 20%，见 `DataConfig.val_ratio`），而非随机抽样
- 监控 AUC 时同时观察正样本率、预测分数分布，排查极端情况
- 注意推理延迟——在本地评估中同时测量推理吞吐量

---

## 6. 待补充的诊断维度

以下分析有助于实验排期决策，但尚未实现：

### 6.1 时间漂移诊断（P0 优先级）

- [ ] **train/val 标签率漂移**：时间切分后对比正样本率变化量，超过 2pp 需调整切分策略或加入时间感知损失
- [ ] **user/item overlap**：验证集中训练集已见用户占比、已见物品占比 → 冷启动占比
- [ ] **冷启动用户 AUC**：单独评估从未在训练集出现的用户群的 AUC，定位冷启动损失
- [ ] **特征覆盖率漂移**：按时间窗口统计高缺失特征的缺失率趋势，识别数据质量退化

### 6.2 分群切片评估（P1 优先级）

- [ ] **用户活跃度分桶 AUC**：按 `user_stats.activity_distribution()` 的分桶，输出各桶 AUC 和覆盖率
- [ ] **物品热度分桶 AUC**：高热/中热/低热/冷启动物品各自的 AUC
- [ ] **序列长度分桶 AUC**：短序列 vs 长序列用户的 AUC 差异 → 指导 max_seq_len 设置
- [ ] **缺失率分桶 AUC**：高缺失率用户 vs 低缺失率用户的 AUC → 验证 missing token 策略效果

### 6.3 特征有效性分析（P1 优先级）

- [ ] **单字段 lift**：在 baseline 基础上逐个移除特征，观察 AUC delta
- [ ] **域贡献度**：逐域 mask 后的 AUC 差异，决定域间资源分配
- [ ] **特征交叉增益**：两两特征组合的 AUC 增量 top-K
- [ ] **label 条件下的 null lift**：dataset_eda 已提供正负样本缺失率对比，需在模型层验证其影响

### 6.4 校准与排序诊断

- [ ] **预测分数分布**：正负样本的分数直方图重叠度 → 判断模型区分能力
- [ ] **校准 vs AUC scatter**：跨实验对比排序能力与校准能力的 tradeoff
- [ ] **分群覆盖趋势**：随训练进行 coverage 是否在下降（过拟合高活跃用户）
