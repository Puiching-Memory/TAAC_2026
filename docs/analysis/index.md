---
icon: lucide/chart-column
---

# 数据集分析与可视化

数据集分析文档，包含 EDA 报告、性能基准和评估指标分析。

## 文档索引

| 文档                          | 说明                                               |
| ----------------------------- | -------------------------------------------------- |
| [EDA 报告](dataset-eda.md)    | 数据集探索性分析，覆盖特征分布、缺失率、序列长度等 |
| [性能基准](benchmarks.md)     | 数据管道吞吐量基准测试                             |
| [评估指标分析](evaluation.md) | AUC 指标解读、优化策略和诊断维度                   |

## 当前状态

- EDA 报告已生成预渲染图表（`docs/assets/figures/eda/`，19 个 ECharts JSON）
- 性能基准图表已生成（`docs/assets/figures/benchmarks/`，6 个 ECharts JSON）
- 评估指标文档覆盖 AUC 优化策略和诊断维度

## 快速背景

PCVR 任务的数据集为 Parquet 格式，120 列：

| 列类型           | 数量 | 说明                                                       |
| ---------------- | ---- | ---------------------------------------------------------- |
| ID/标签          | 5    | user_id, item_id, label_type, label_action_type, timestamp |
| user_int_feats   | 46   | 用户整数特征                                               |
| user_dense_feats | 10   | 用户稠密特征                                               |
| item_int_feats   | 14   | 物品整数特征                                               |
| 序列特征         | 45   | 4 个域（a, b, c, d）的序列数据                             |

生成 EDA 报告：

```bash
uv run taac-dataset-eda \
  --dataset-path data/sample_1000_raw/demo_1000.parquet \
  --schema-path data/sample_1000_raw/schema.json \
  --output-dir docs/assets/figures/eda
```
