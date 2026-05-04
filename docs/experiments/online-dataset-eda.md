---
icon: lucide/bar-chart
---

# Online Dataset EDA

线上友好的数据集探索性分析（EDA）工具，以实验包形式集成，可在训练 bundle 内直接运行，无需额外依赖。

## 概述

`online_dataset_eda` 是一个维护类实验包（`kind: maintenance`），对 Parquet 格式的数据集进行流式统计分析，输出结构化报告。它专为线上环境设计：纯流式处理，内存占用可控，不需要 matplotlib 等可视化库。

该包需要数据集（`requires_dataset: True`）。

## 分析内容

| 分析项               | 方法                              | 说明                                   |
| -------------------- | --------------------------------- | -------------------------------------- |
| 列布局概览           | schema 解析                       | scalar / user_int / user_dense / item_int / sequence 列数 |
| 空值率               | 逐列扫描                          | 按空值率降序排列                       |
| 稀疏特征基数         | KMV Sketch（k=4096）              | 近似去重计数，内存可控                 |
| 基数分桶             | 1-10 / 11-100 / 101-1000 / 1001+ | 稀疏特征基数分布                       |
| 序列长度分布         | Reservoir Sampling（n=16384）     | min/Q1/median/Q3/max/mean/p95/empty_rate |
| 序列重复率           | 逐行采样                          | 每个序列域的 token 重复率              |
| 稠密特征分布         | 在线均值/方差                     | mean、std、zero_frac                   |
| 共现缺失矩阵         | 二次扫描                          | 高空值列两两共现缺失计数               |
| 用户活跃度           | Bottom-K 用户采样（n=50000）      | 采样用户的行为次数分布                 |
| 跨域用户重叠         | Bottom-K 用户采样                 | 序列域两两 Jaccard 重叠率              |

## 两遍扫描架构

1. **第一遍**：流式扫描全量（或采样）数据，收集空值、基数、序列长度、稠密统计、用户采样
2. **第二遍**：基于第一遍的采样用户和高空值列，计算共现缺失矩阵和跨域重叠

每遍扫描使用 PyArrow 逐 batch 读取，batch 大小默认 128 行。

## 配置

配置项在 `experiments/maintenance/online_dataset_eda/__init__.py` 的 `OnlineDatasetEDAConfig` 中定义，也支持环境变量覆盖：

| 参数                       | 默认值   | 环境变量                    | 说明                           |
| -------------------------- | -------- | --------------------------- | ------------------------------ |
| `dataset_path`             | -        | -                           | 数据集路径（必填）             |
| `schema_path`              | 自动推导 | `TAAC_SCHEMA_PATH`          | schema.json 路径               |
| `batch_rows`               | 128      | `ONLINE_EDA_BATCH_ROWS`     | 每批读取行数                   |
| `cardinality_sketch_k`     | 4096     | -                           | KMV Sketch 容量               |
| `user_sample_limit`        | 50000    | -                           | 用户采样上限                   |
| `sequence_sample_size`     | 16384    | -                           | 序列长度采样容量               |
| `max_rows`                 | None     | `ONLINE_EDA_MAX_ROWS`       | 最大扫描行数                   |
| `sample_percent`           | None     | `ONLINE_EDA_SAMPLE_PERCENT` | 采样百分比（与 max_rows 互斥） |
| `progress_step_percent`    | 10.0     | -                           | 进度打印间隔（%）              |

## 运行

```bash
uv run taac-train --experiment experiments/maintenance/online_dataset_eda \
  --dataset-path data/sample_1000_raw/demo_1000.parquet \
  --schema-path data/sample_1000_raw/schema.json
```

支持目录（含多个 `.parquet` 文件）或单文件路径。schema.json 自动从数据集同级目录推导，也可通过 `--schema-path` 或 `TAAC_SCHEMA_PATH` 显式指定。

## 线上打包

```bash
uv run taac-package-train --experiment experiments/maintenance/online_dataset_eda --output-dir outputs/bundle
```

该维护类实验支持训练 bundle 打包，生成 `run.sh` 与 `code_package.zip`，可在线上环境直接执行 EDA 任务。

由于 `online_dataset_eda` 不实现模型推理接口，因此不适用 `taac-package-infer`。

## 输出

运行后将报告打印到 stdout，包含以下节：

- **Dataset**：路径、总行数、是否采样
- **Column Layout**：各类型列数、序列域分布
- **Top Null Rates**：高空值特征排名
- **Top Cardinalities**：高基数稀疏特征排名
- **Sequence Length Summary**：各序列域长度分布
- **Sequence Repeat Rate**：各序列域 token 重复率
- **Dense Feature Summary**：稠密特征均值/标准差/零值比例
- **Cardinality Bins**：基数分桶分布
- **Sampled User Activity**：采样用户行为次数分布
- **Sampled Cross-Domain Overlap**：跨域 Jaccard 重叠矩阵
- **Approximation**：各近似方法的参数说明

返回值为完整的结构化报告 dict，可用于下游自动化分析。

## 设计约束

- 纯 Python + PyArrow，不依赖 pandas、matplotlib 或其他可视化库
- 流式处理，不将全量数据加载到内存
- 近似算法（KMV、Reservoir、Bottom-K）控制内存上限
- 标签相关分析（label_distribution、feature_auc、null_rate_by_label）已跳过，因为线上 EDA 场景通常无标签

## 来源

- 运行器源码：`experiments/maintenance/online_dataset_eda/runner.py`
- 包入口：`experiments/maintenance/online_dataset_eda/__init__.py`
