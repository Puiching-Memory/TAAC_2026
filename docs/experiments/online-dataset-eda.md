---
icon: lucide/bar-chart
---

# Online Dataset EDA：比较数据分布

该维护实验流式扫描 parquet，比较训练与推理的缺失、基数、dense 和序列分布。它支持 train/infer，不支持模型评估；推理入口不产生预测文件，因此不是正式提交模型。

## 先生成训练 profile

完成[样例准备](../getting-started.md)后：

```bash
uv run --with torch==2.13.0 taac-train \
  --experiment experiments/online_dataset_eda \
  --dataset-path data/sample_1000_raw/demo_1000.parquet \
  --schema-path data/sample_1000_raw/schema.json \
  --run-dir outputs/eda_train
```

完整报告只在 stdout 中打印为一行：

```text
ONLINE_DATASET_EDA_RESULT={...}
```

保存这行作为对照。CLI 最后的 summary 只有少量字段，不是完整 profile；默认也不会自动写 profile 或 drift 文件。

## 比较推理数据

把训练结果保存为 `outputs/eda_train/profile.json`，或通过 `ONLINE_EDA_REFERENCE_PROFILE_JSON` 传入原始结果行。文件与内联 JSON 是两种输入方式，选择其一。

```bash
export ONLINE_EDA_REFERENCE_PROFILE="$(pwd)/outputs/eda_train/profile.json"
uv run --with torch==2.13.0 taac-evaluate infer \
  --experiment experiments/online_dataset_eda \
  --dataset-path data/sample_1000_raw/demo_1000.parquet \
  --schema-path data/sample_1000_raw/schema.json \
  --result-dir outputs/eda_infer
```

此例用同一数据检查比较路径；分析漂移时换成真实 infer parquet。该实验的 `--checkpoint` 可作为参考 profile 文件路径使用，不是模型权重。

## 应读取的字段

| 字段 | 用途 |
| --- | --- |
| `row_count`、`sampled`、`approximation` | 确认扫描规模与近似口径 |
| `stats.column_layout`、`stats.null_rates` | 核对布局、缺失变化 |
| `stats.label_distribution` | 检查 `label_type == 2` 的正样本口径 |
| `stats.cardinality`、`stats.token_overlap_sketch` | 基数、token overlap、近似 OOV |
| `stats.sequence_lengths` | 长序列和空域变化 |
| `stats.dense_distributions` | 尺度、方差、零值比例 |
| `comparison` | 与参考 profile 的差异和风险提示 |

KMV sketch、重复率与用户覆盖统计有采样误差。对照时保持 sketch 大小、采样参数、schema 和代码版本一致；分布变化不直接证明模型因果失效。

## 控制扫描成本

参数通过 runner 配置与环境变量提供，额外训练参数会被拒绝。

| 环境变量 | 作用 |
| --- | --- |
| `ONLINE_EDA_ANALYSIS_LEVEL` | `fast` 或 `full` |
| `ONLINE_EDA_BATCH_ROWS` | Arrow 读取 batch，默认 8192 |
| `ONLINE_EDA_MAX_ROWS` | 最多扫描行数 |
| `ONLINE_EDA_SAMPLE_PERCENT` | 抽样百分比，与 MAX_ROWS 互斥 |
| `ONLINE_EDA_ENABLE_LABEL_LIFT` | 是否做更重的标签关联统计 |
| `ONLINE_EDA_CARDINALITY_SKETCH_K` | 基数 sketch 大小 |
| `ONLINE_EDA_TOKEN_OVERLAP_SKETCH_K` | overlap sketch 大小 |
| `ONLINE_EDA_REFERENCE_PROFILE` | 对照文件 |
| `ONLINE_EDA_REFERENCE_PROFILE_JSON` | 对照 JSON 或完整结果行 |

`full` 和 label lift 成本更高，应先运行 fast profile 定位要复核的字段。完整变量表与校验以 `experiments/online_dataset_eda/runner.py` 的配置解析为准。

## 打包与验证

```bash
uv run --with torch==2.13.0 taac-package-train \
  --experiment experiments/online_dataset_eda --output-dir outputs/bundles/eda_train
uv run --with torch==2.13.0 taac-package-infer \
  --experiment experiments/online_dataset_eda --output-dir outputs/bundles/eda_infer
uv run --with torch==2.13.0 pytest \
  tests/contract/experiments/test_online_dataset_eda_runner.py -q
```

平台数据变量分别为 `TRAIN_DATA_PATH` 与 `EVAL_DATA_PATH`，schema 由 `TAAC_SCHEMA_PATH` 指定。使用生成包前先核对[当前平台兼容性](../guide/online-training-bundle.md)。
