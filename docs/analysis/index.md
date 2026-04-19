---
icon: lucide/chart-column
---

# 数据集分析与可视化

本分区提供 TAAC 2026 数据集的自动化探索性分析，以及从上届竞赛论文中提炼的数据层面经验。

所有统计数据和图表均由 `taac-dataset-eda` CLI 工具自动生成，可随时重跑刷新：

```bash
uv run taac-dataset-eda                            # 默认 sample 数据集
uv run taac-dataset-eda --dataset path/to/data     # 自定义数据路径
uv run taac-dataset-eda --json-path figures/eda/stats.json  # 同时输出 JSON
uv run taac-bench-report                           # 生成性能基准图表 JSON
```

## 文档索引

| 文档                                 | 说明                                           |
| ------------------------------------ | ---------------------------------------------- |
| [数据集 EDA 报告](dataset-eda.md)    | 本届数据集的自动化分析报告（含图表）           |
| [性能基准](benchmarks.md)            | benchmark JSON 到 ECharts 图表的自动化汇总页   |
| [评估指标分析](evaluation.md)        | 评估协议解读与指标优化方向                     |
| [TAAC 2025 论文洞察](taac2025-insights.md) | TAAC 2025 论文关键经验提炼 |

## 编程接口

分析功能也可作为 Python API 直接调用：

```python
from taac2026.infrastructure.io.datasets import iter_dataset_rows
from taac2026.reporting.dataset_eda import (
    classify_columns,
    compute_column_stats,
    compute_label_distribution,
    compute_sequence_lengths,
    echarts_label_distribution,
)

rows = list(iter_dataset_rows("TAAC2026/data_sample_1000"))
groups = classify_columns(list(rows[0].keys()))
label_dist = compute_label_distribution(iter(rows))

# 生成 ECharts JSON 配置（可直接写入 .echarts.json 文件）
import json
chart_opt = echarts_label_distribution(label_dist)
print(json.dumps(chart_opt, ensure_ascii=False, indent=2))
```

## 快速背景

TAAC（腾讯广告算法挑战赛）是针对工业广告场景的**全模态生成推荐**竞赛。核心任务是：给定用户的全模态广告互动历史序列，预测用户下一个最可能交互的广告。

**与经典推荐的关键区别**：

- 生成式范式：自回归序列预测，而非判别式重排序
- 全模态输入：协作 ID + 文本嵌入 + 视觉嵌入
- 异构行为：曝光 / 点击 / 转化，需区分行为类型
- 工业级规模：百万到千万级用户序列
