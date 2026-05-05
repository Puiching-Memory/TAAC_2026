---
icon: lucide/clipboard-list
---

# 评估指标分析

## 1. 竞赛评估指标：AUC

PCVR 任务使用 AUC（Area Under ROC Curve）作为主评估指标，衡量模型对正负样本的排序能力。

### 1.1 AUC 的直觉含义

AUC = 随机抽一个正样本和一个负样本，模型给正样本的预测分高于负样本的概率。

- AUC = 0.5：随机猜测
- AUC = 1.0：完美排序
- 实际范围：0.65 - 0.85（典型 CTR/CVR 场景）

### 1.2 延迟约束

线上推理有延迟限制，模型复杂度需要在 AUC 和推理速度之间平衡。

## 2. 数据分布对 AUC 的影响

### 关键发现

- **类别不平衡**：转化事件（label_type=2）远少于点击事件，影响 AUC 的有效样本量
- **特征缺失**：部分特征覆盖率低，对 AUC 的贡献有限
- **序列长度**：短序列用户的特征信息不足，预测难度更大

## 3. AUC 优化策略

### 3.1 损失函数选择

| 损失函数          | 说明              | 适用场景                      |
| ----------------- | ----------------- | ----------------------------- |
| BCE               | 标准二分类交叉熵  | 默认选择                      |
| Focal Loss        | 降低易分样本权重  | 类别不平衡严重时              |
| Pairwise AUC Loss | 直接优化 AUC 排序 | Symbiosis 使用（weight=0.05） |

### 3.2 特征工程方向

- 高基数特征的 Embedding 策略
- 序列特征的编码方式（Transformer vs RankMixer）
- NS Groups 的分组策略

### 3.3 模型 pipeline

- 双优化器：Adagrad（稀疏） + AdamW（稠密）
- AMP 混合精度训练
- Early Stopping 基于验证集 AUC

## 4. 本届指标确认事项

- 主指标：AUC
- 评估粒度：全量样本（不分群）
- 提交格式：`predictions.json`（`{"predictions": {user_id: score}}`）

## 5. 当前代码中的评估指标

### 5.1 指标用法

`domain/metrics.py` 提供的指标：

| 函数                             | 说明                             |
| -------------------------------- | -------------------------------- |
| `binary_auc`                     | 标准 AUC                         |
| `binary_logloss`                 | 对数损失                         |
| `binary_score_diagnostics`       | 预测分诊断（均值、方差、分位数） |
| `binary_auc_bootstrap_ci`        | Bootstrap 置信区间               |
| `group_auc`                      | 按用户分组的 GAUC                |
| `compute_classification_metrics` | 汇总所有指标                     |

### 5.2 为什么需要多个指标

- AUC 衡量排序能力，LogLoss 衡量校准程度
- GAUC 按用户分组，更贴近真实推荐场景
- Bootstrap CI 评估指标的统计显著性

### 5.3 评估最佳实践

```bash
# 评估并输出所有指标
uv run taac-evaluate single \
  --experiment experiments/pcvr/symbiosis
```

## 6. 待补充的诊断维度

### 6.1 时间漂移诊断（P0 优先级）

- 训练集和验证集的时间分布差异
- 模型在不同时间段的 AUC 稳定性

### 6.2 分群切片评估（P1 优先级）

- 按用户活跃度分群的 AUC
- 按物品类别分群的 AUC
- 按序列长度分群的 AUC

### 6.3 特征有效性分析（P1 优先级）

- 特征消融实验
- 特征重要性排序

### 6.4 校准与排序诊断

- 预测分分布 vs 实际转化率
- 校准曲线分析
