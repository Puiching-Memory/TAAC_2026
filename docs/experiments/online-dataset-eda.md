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
  --schema-path docs/archive/files/schema/sample_1000_raw.schema.json
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

## 2026-05-08 线上全量数据结论

这次线上平台运行成功，用户脚本退出码为 `0`。平台初始化阶段出现的 `cp: cannot stat` 和 `touch /data/initCmdCompletedFile` 属于平台启动噪声，主 EDA 任务已完整跑完。

扫描对象是线上训练集 `/data_ams/academic_training_data`，共 `1,010,000` 行，未采样，`batch_rows=128`。字段布局为 `scalar=5`、`user_int=46`、`user_dense=10`、`item_int=14`、`sequence=45`，序列域分布为 `seq_a=9`、`seq_b=14`、`seq_c=12`、`seq_d=10`。

### 标签分布

正样本 `96,876`，负样本 `913,124`，正样本率 `0.095917`，约为 `1:9.43`。后续模型的输出层 bias 可以按该先验初始化，BCE 类损失也可以参考 `pos_weight ~= 9.43`，但如果评估重视概率校准，需要在验证集上单独检查 calibration。

### 缺失值

高缺失特征集中在部分 user/item 离散列：

- `user_int_feats_101` 缺失率 `0.920411`。
- `user_int_feats_102` 缺失率 `0.895672`。
- `user_int_feats_103` 缺失率 `0.882711`。
- `item_int_feats_83/84/85` 缺失率均为 `0.879531`。
- `user_int_feats_99/100/109` 缺失率也超过 `0.81`。

缺失本身带有标签信号，不建议直接丢弃高缺失列。`item_int_feats_83/84/85` 的正样本缺失率约 `0.842`，负样本缺失率约 `0.884`，差异约 `-4.15pp`；`user_int_feats_104` 差异约 `-3.79pp`；`item_int_feats_11` 则是正样本缺失率更高，差异约 `+2.92pp`。离散特征应保留 missing/unknown token，稠密特征建议同时提供 presence mask。

### 基数与 embedding

最高基数来自序列 side-info token：

- `domain_c_seq_47` 约 `71,619,665`。
- `domain_b_seq_69` 约 `55,566,346`。
- `domain_c_seq_29` 约 `5,392,434`。
- `domain_c_seq_34`、`domain_c_seq_36` 接近 `0.9M`。

静态 item 特征中，`item_int_feats_16` 基数约 `21,958`，`item_int_feats_11` 基数约 `20,044`，可以直接建常规 embedding。千万级序列字段不适合全量 vocabulary 化，优先使用 hash embedding、分域分字段 embedding、截断池化或离线聚合。

### 特征标签信号

Item 侧离散特征信号明显强于 user 侧：

- `item_int_feats_16=20543`：support `444`，正样本率 `0.979730`，lift `10.214367`。
- `item_int_feats_16=11833`：support `3360`，正样本率 `0.855357`，lift `8.917696`。
- `item_int_feats_10=121`：support `4297`，正样本率 `0.675820`，lift `7.045899`。
- `item_int_feats_5=267` 和 `item_int_feats_6=237`：support 均为 `4265`，正样本率约 `0.674795`。

User 侧也有信号，但更稀疏、更弱。可优先关注 `user_int_feats_48`、`user_int_feats_57`、`user_int_feats_63`、`user_int_feats_86`。建模容量应优先给 item 侧强特征，再补 user 侧交互。

### 序列分布

线上数据每行序列很重，空序列比例接近 `0`：

| 序列域  | mean          | median   | p95      | max      | empty_rate |
| ------- | ------------- | -------- | -------- | -------- | ---------- |
| `seq_a` | `751.332150`  | `558.0`  | `1969.0` | `2000.0` | `0.000740` |
| `seq_b` | `722.638174`  | `488.0`  | `1969.0` | `2000.0` | `0.001132` |
| `seq_c` | `512.818620`  | `379.0`  | `1438.0` | `4000.0` | `0.000493` |
| `seq_d` | `2455.213142` | `2722.5` | `3910.0` | `4000.0` | `0.000975` |

重复率也很高：`seq_a=0.252239`、`seq_b=0.945815`、`seq_c=0.975259`、`seq_d=0.994539`。直接对完整序列做 Python 级处理或全长 attention 都不现实。建议对 `seq_b/seq_c/seq_d` 做去重、计数截断、last-K、hash-bag pooling 或预聚合；`seq_a` 重复率较低，可以保留更多时序和最近行为信息。

### 稠密特征

`user_dense_feats_62-66` 数值尺度很大，均值在 `1e5` 量级，标准差可到 `1e6`，不应裸喂 MLP，建议 clip、log transform 或标准化。`user_dense_feats_89-91` 方差稳定在 `0.1` 左右，像是已标准化特征。`user_dense_feats_87` 零值比例 `0.620482`，zero 本身可能是有意义状态。

### 建模优先级

建议的起步顺序：

1. 优先建好 `item_int_feats_16/11/10/5/6` 等强 item embedding。
2. 为高缺失 user/item 特征保留 missing token，并为 dense 缺失添加 mask。
3. 对四个序列域先做 hash-bag/去重池化，避免全长 attention。
4. 加入序列长度、重复率或截断后 token 数等轻量统计特征。
5. 对 `user_dense_feats_62-66` 做尺度处理后再进入 dense tower。

### 运行性能

这次全量 EDA 总耗时约 `12h21m`。第一遍扫描约 `11h50m`，第二遍约 `31m`。慢点主要在第一遍对长序列做基数、长度、重复率和标签缺失统计。日常迭代建议先限制扫描量：

```bash
export ONLINE_EDA_MAX_ROWS=100000
```

或使用比例采样：

```bash
export ONLINE_EDA_SAMPLE_PERCENT=5
```

完整全量扫描适合最终确认数据分布，不适合作为频繁迭代步骤。

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
