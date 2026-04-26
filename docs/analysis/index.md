---
icon: lucide/chart-column
---

# 数据集分析与可视化

本分区提供 TAAC 2026 数据集的自动化探索性分析，以及从上届竞赛论文中提炼的数据层面经验。

EDA 统计数据和图表面向当前官方 flat-column sample 格式维护。若你在本地重跑相关流程，建议显式传入 `demo_1000.parquet` 与同目录 `schema.json`，并把 `docs/assets/figures/eda/` 下刷新后的 JSON 一并提交：

```bash
uv run taac-dataset-eda \
    --dataset-path data/sample_1000_raw/demo_1000.parquet \
    --schema-path data/sample_1000_raw/schema.json

uv run taac-dataset-eda \
    --dataset-path path/to/demo_1000.parquet \
    --schema-path path/to/schema.json \
    --output outputs/reports/dataset_eda.json

uv run taac-bench-report --output outputs/reports/benchmark_report.json  # 写出占位 benchmark 报告
```

## 文档索引

| 文档                                 | 说明                                           |
| ------------------------------------ | ---------------------------------------------- |
| [数据集 EDA 报告](dataset-eda.md)    | 本届数据集的自动化分析报告（含图表）           |
| [性能基准](benchmarks.md)            | 当前 benchmark 占位报告入口说明                 |
| [评估指标分析](evaluation.md)        | 评估协议解读与指标优化方向                     |
| [TAAC 2025 论文洞察](taac2025-insights.md) | TAAC 2025 论文关键经验提炼 |

## 当前状态

当前仓库保留的数据分析入口以文档资产和 CLI 调用约定为主，不再维护稳定的 Python 级 EDA API 示例。若后续恢复 Python API，应统一以官方 `demo_1000.parquet` + `schema.json` 这套 flat-column 输入约定为准。

## 快速背景

TAAC（腾讯广告算法挑战赛）是针对工业广告场景的**全模态生成推荐**竞赛。核心任务是：给定用户的全模态广告互动历史序列，预测用户下一个最可能交互的广告。

**与经典推荐的关键区别**：

- 生成式范式：自回归序列预测，而非判别式重排序
- 全模态输入：协作 ID + 文本嵌入 + 视觉嵌入
- 异构行为：曝光 / 点击 / 转化，需区分行为类型
- 工业级规模：百万到千万级用户序列
