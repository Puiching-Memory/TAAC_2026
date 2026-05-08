---
icon: lucide/bar-chart
---

# Online Dataset EDA

Online Dataset EDA 是一个维护类实验包，用来在线上训练环境里对 parquet 数据做文本化探索分析。它按 batch 流式扫描，不依赖可视化库，也不要求把全量数据读进内存。

## 本地运行

```bash
bash run.sh train \
  --experiment experiments/online_dataset_eda \
  --dataset-path data/sample_1000_raw/demo_1000.parquet \
  --schema-path data/sample_1000_raw/schema.json
```

支持单个 parquet 文件或包含多个 parquet 的目录。

## 实验契约

入口文件 `experiments/online_dataset_eda/__init__.py` 导出 `ExperimentSpec`：

```python
EXPERIMENT = ExperimentSpec(
    name="online_dataset_eda",
    kind="maintenance",
    requires_dataset=True,
)
```

这个实验必须有 `dataset_path`。`train_fn` 会拒绝 extra args；采样和 batch 参数通过 runner config 或环境变量控制。

## 线上运行

先生成训练 bundle：

```bash
uv run taac-package-train \
  --experiment experiments/online_dataset_eda \
  --output-dir outputs/bundles/online_dataset_eda
```

线上通过平台变量给数据路径：

```bash
export TRAIN_DATA_PATH=/path/to/train_parquet_or_dir
export TAAC_SCHEMA_PATH=/path/to/schema.json

bash run.sh
```

这个实验不支持推理 bundle。

## 报告怎么看

报告打印到 stdout。优先看这些节：

- `Dataset`：实际扫到的数据路径、行数和采样状态。
- `Column Layout`：schema 中各类列的数量和序列域分布。
- `Label Distribution`：`label_type == 2` 的正样本数、负样本数、缺失数和正样本率。
- `Null Rate By Label`：正负样本中的特征空值率差异。
- `Top Null Rates`：高空值特征。
- `Top Cardinalities`：高基数稀疏特征。
- `Top Sequence Token Cardinalities`：序列 side-info token 的基数估计，不包含时间列。
- `Top User Feature Label Lift`：用户离散特征候选 token 的正样本率、lift 和平滑 log-odds。
- `Top Item Feature Label Lift`：物品离散特征候选 token 的正样本率、lift 和平滑 log-odds。
- `Top Categorical Pair Associations`：低基数类别列之间的 Cramer's V 关联度。
- `Sequence Length Summary`：序列长度分布。
- `Dense Feature Summary`：稠密特征均值、方差、标准差和零值比例。
- `Sampled Cross-Domain Coverage`：bottom-k 抽样用户在各序列域的覆盖率。
- `Approximation`：近似统计使用的采样参数。

返回 summary 只保留自动化友好的少量字段：

```python
{
    "experiment_name": "online_dataset_eda",
    "run_dir": "...",
    "dataset_role": "...",
    "row_count": ...,
    "sampled": ...,
}
```

完整报告打印到 stdout。

## Runner 配置

`OnlineDatasetEDAConfig` 字段：

| 字段                    | 默认                      | 作用                      |
| ----------------------- | ------------------------- | ------------------------- |
| `dataset_path`          | 必填                      | parquet 文件或目录        |
| `schema_path`           | 自动 / `TAAC_SCHEMA_PATH` | schema 路径               |
| `batch_rows`            | 128                       | PyArrow 每批读取行数      |
| `cardinality_sketch_k`  | 4096                      | KMV sketch 容量           |
| `user_sample_limit`     | 50000                     | 用户 bottom-k 采样上限    |
| `sequence_sample_size`  | 16384                     | 序列长度 reservoir 样本数 |
| `max_rows`              | None                      | 最大扫描行数              |
| `sample_percent`        | None                      | 按百分比限制扫描行数      |
| `progress_step_percent` | 10.0                      | 进度日志间隔              |
| `label_feature_candidate_k` | 128 | 每个 user/item 离散列保留的 label lift 候选 token 数 |
| `label_feature_top_k` | 20 | label lift 输出行数 |
| `label_feature_min_support` | 20 | label lift token 最小样本数 |
| `categorical_pair_max_columns` | 16 | 类别列相关性最多纳入的列数 |
| `categorical_pair_max_cardinality` | 128 | 类别列相关性允许的最大列基数 |
| `categorical_pair_sample_rows` | 50000 | 类别列相关性 reservoir 样本行数 |
| `categorical_pair_top_k` | 20 | 类别列相关性输出行数 |

环境变量覆盖：

| 变量                        | 作用             |
| --------------------------- | ---------------- |
| `TAAC_SCHEMA_PATH`          | 显式 schema 路径 |
| `ONLINE_EDA_BATCH_ROWS`     | 覆盖 batch rows  |
| `ONLINE_EDA_MAX_ROWS`       | 限制最大扫描行数 |
| `ONLINE_EDA_SAMPLE_PERCENT` | 按百分比采样     |

`ONLINE_EDA_MAX_ROWS` 和 `ONLINE_EDA_SAMPLE_PERCENT` 互斥。

## 近似算法

为控制线上内存，EDA 不做全量重型分析：

- 稀疏基数使用 KMV sketch。
- 序列 side-info token 基数也使用 KMV sketch。
- user/item token 与 label 的关联先用 bounded 候选 sketch，再在第二遍扫描里对候选做精确正负计数。
- 序列长度分布使用 reservoir sampling。
- 用户活跃度、跨域覆盖和跨域重叠使用 bottom-k 用户采样。
- 类别列相关性只纳入低基数列，并用 reservoir 行样本计算 Cramer's V。
- 高空值列共现缺失需要第二遍扫描。

因此报告适合判断量级、分布和异常方向，不适合作为精确离线统计报表。

## 调小扫描量

大数据上先限制扫描量：

```bash
export ONLINE_EDA_MAX_ROWS=100000
```

或者按比例采样：

```bash
export ONLINE_EDA_SAMPLE_PERCENT=5
```

不要同时设置这两个变量。

## 源码入口

- 实验入口：`experiments/online_dataset_eda/__init__.py`
- 分析逻辑：`experiments/online_dataset_eda/runner.py`
- 维护实验测试：`tests/contract/experiments/test_online_dataset_eda_runner.py`

## 最小复核

```bash
uv run pytest tests/contract/experiments/test_maintenance_experiments.py -q
uv run pytest tests/contract/experiments/test_online_dataset_eda_runner.py -q
```
