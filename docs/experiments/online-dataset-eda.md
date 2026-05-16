---
icon: lucide/bar-chart
---

# Online Dataset EDA

Online Dataset EDA 是一个维护类实验包，用来在线上训练和推理环境里对 parquet 数据做文本化探索分析。它按 batch 流式扫描，不依赖可视化库，也不要求把全量数据读进内存。当前默认目标从“全量精细报表”调整为“快速生成可比较 profile”，用于发现 train 和 infer 数据之间的分布差异。

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

再生成推理 bundle：

```bash
uv run taac-package-infer \
  --experiment experiments/online_dataset_eda \
  --output-dir outputs/bundles/online_dataset_eda_infer
```

线上推理阶段通过平台变量给评测数据路径：

```bash
export EVAL_DATA_PATH=/path/to/infer_parquet_or_dir
export EVAL_RESULT_PATH=/path/to/result_dir
export ONLINE_EDA_REFERENCE_PROFILE_JSON='ONLINE_DATASET_EDA_RESULT={...}'
export TAAC_SCHEMA_PATH=/path/to/schema.json

python infer.py
```

训练和推理阶段都只把完整结果打印到 stdout，不再默认写 profile / drift JSON 文件。线上平台看日志时搜索前缀 `ONLINE_DATASET_EDA_RESULT=`，这一行后面就是紧凑 JSON。

如果需要在推理阶段直接做 train/infer diff，把训练阶段日志里的整行 `ONLINE_DATASET_EDA_RESULT=...` 复制到 `ONLINE_EDA_REFERENCE_PROFILE_JSON`。本地调试时仍可用 `ONLINE_EDA_REFERENCE_PROFILE` 指向手工保存的训练 profile 文件。

## 报告怎么看

报告是单行紧凑 JSON，适合直接从线上日志采集。优先看这些字段：

- `dataset_path`、`row_count`、`sampled`：实际扫到的数据路径、行数和采样状态。
- `comparison`：推理 profile 与训练 profile 的差异摘要和 risk flags。
- `comparison.null_rate_drift`：特征缺失率在 train/infer 间的变化。
- `comparison.cardinality_drift`：稀疏 token 基数估计的变化，使用 log2 ratio 排序。
- `comparison.token_overlap_drift`：train/infer token KMV 指纹重叠和近似 OOV 方向。
- `comparison.sequence_length_drift`：四个序列域长度、p95 和空序列率变化。
- `comparison.dense_distribution_drift`：稠密特征均值/标准差变化。
- `stats.column_layout`：schema 中各类列的数量和序列域分布。
- `stats.label_distribution`：`label_type == 2` 的正样本数、负样本数、缺失数和正样本率。
- `stats.null_rates`：高空值特征。
- `stats.cardinality`：高基数稀疏特征。
- `stats.token_overlap_sketch`：用于 train/infer token overlap/OOV 对比的小型 hash 指纹。
- `stats.sequence_token_cardinality`：序列 side-info token 的基数估计，不包含时间列。
- `stats.user_feature_label_lift` / `stats.item_feature_label_lift`：默认关闭，设置 `ONLINE_EDA_ENABLE_LABEL_LIFT=1` 或 `ONLINE_EDA_ANALYSIS_LEVEL=full` 后输出。
- `stats.sequence_lengths`：序列长度分布。
- `stats.dense_distributions`：稠密特征均值、方差、标准差和零值比例。
- `stats.cross_domain_coverage`：bottom-k 抽样用户在各序列域的覆盖率。
- `approximation`：近似统计使用的采样参数。

返回 summary 只保留自动化友好的少量字段：

```python
{
    "experiment_name": "online_dataset_eda",
    "run_dir": "...",
    "dataset_role": "...",
    "row_count": ...,
    "sampled": ...,
    "stdout_result": True,
}
```

推理返回还会包含：

```python
{
    "experiment_name": "online_dataset_eda",
    "result_dir": "...",
    "dataset_role": "infer",
    "row_count": ...,
    "sampled": ...,
    "reference_profile_path": "...",
    "risk_flags": [...],
    "stdout_result": True,
}
```

完整结果只在 stdout 的 `ONLINE_DATASET_EDA_RESULT=` 单行里。自动化分析优先读取这行日志。

## 性能策略

当前 runner 默认使用 Arrow compute 做单遍扫描，默认 `batch_rows=8192`。相比旧版 `batch_rows=128` 加二遍 Python `to_pylist()` 扫描，默认路径显著减少 Python 行级处理。

默认 profile 会保留这些快速、可比较的统计：

- 缺失率、标签分布、列布局。
- 稀疏和序列 side-info token 基数，使用 KMV sketch + batch 内 Arrow unique 采样。
- 序列长度分布、重复率抽样、用户跨域覆盖抽样。
- 稠密特征均值、方差、标准差、零值比例。

默认关闭的重型分析包括 label lift 的逐 token 标签统计。需要最终确认强特征时再打开：

```bash
export ONLINE_EDA_ENABLE_LABEL_LIFT=1
```

或者使用 full 模式：

```bash
export ONLINE_EDA_ANALYSIS_LEVEL=full
```

日常排查 train/infer drift 建议先使用默认 fast 模式，再针对可疑列开启 full。

## 2026-05-12 线上 Train/Infer 对比结论

这次线上 EDA 分别扫描训练集 `/data_ams/academic_training_data` 和推理集 `/data_ams/academic_infer_data`。训练集 `1,010,000` 行，推理集 `310,000` 行；两边 schema 完全一致，`schema_signature=0f427b26ed744814222527d2`，字段布局也一致：`scalar=5`、`user_int=46`、`user_dense=10`、`item_int=14`、`sequence=45`，序列域为 `seq_a=9`、`seq_b=14`、`seq_c=12`、`seq_d=10`。

线上推理阶段出现 `predictions.json` 缺失是维护类 EDA 实验没有生成预测文件导致的评测外壳错误，不影响 EDA 结果。需要关注的是数据分布：schema 没变，但 dense 分布、缺失率、item 基数和长序列长度都发生了可见漂移。这解释了随机 valid AUC 高而线上 AUC 降到 `0.77` 的主要风险来源。

### 标签与样本规模

| 项目 | Train | Infer | 结论 |
| ---- | ----- | ----- | ---- |
| 行数 | `1,010,000` | `310,000` | 推理集约为训练集 `30.7%` |
| `label_type` | 正样本率 `0.095917` | 全缺失 | 推理集无标签是正常线上评测形态 |
| schema signature | `0f427b26ed744814222527d2` | `0f427b26ed744814222527d2` | 不是 schema mismatch |

训练正样本率约 `1:9.43`。模型输出层 bias、calibration 和 early stopping 仍应围绕这个先验做本地验证；推理集没有 label，不能用线上 infer 直接估计点击率变化。

### 缺失率漂移

最明显的缺失率变化集中在 item side 和部分 user sparse：

| 特征 | Train null_rate | Infer null_rate | 变化 |
| ---- | --------------- | --------------- | ---- |
| `item_int_feats_11` | `0.496732` | `0.531232` | `+0.034500` |
| `user_int_feats_15` | `0.141426` | `0.110219` | `-0.031207` |
| `user_int_feats_80` | `0.246279` | `0.272871` | `+0.026592` |
| `user_int_feats_92` | `0.484775` | `0.509374` | `+0.024599` |

训练集本身还存在极高缺失列：`user_int_feats_101=0.920411`、`user_int_feats_102=0.895672`、`user_int_feats_103=0.882711`、`item_int_feats_83/84/85=0.879531`，`user_int_feats_99/100/109` 也超过 `0.81`。这些特征不能简单丢弃或把缺失压成默认 `0`；缺失模式本身是信号，线上还会漂移。当前框架应使用 sparse missing embedding 和 dense missing indicator。

### Item 基数漂移

推理集 item token 覆盖明显收缩，这会让依赖 item ID 记忆的模型在隐藏评测上失效：

| 特征 | Train cardinality | Infer cardinality | 结论 |
| ---- | ----------------- | ----------------- | ---- |
| `item_int_feats_16` | `22,192` | `14,280` | 强 item 特征覆盖变窄 |
| `item_int_feats_11` | `20,301` | `13,473` | 与缺失率同时漂移 |
| `item_int_feats_8` | `2,042` | `1,648` | 中等 item 特征收缩 |
| `item_int_feats_7` | `2,303` | `1,933` | 中等 item 特征收缩 |
| `item_int_feats_12` | `2,305` | `1,933` | 中等 item 特征收缩 |

训练集 item 侧标签信号很强，例如 `item_int_feats_16=20543` 的 support `444`、正样本率 `0.979730`、lift `10.214367`，`item_int_feats_16=11833` 的 support `3360`、正样本率 `0.855357`、lift `8.917696`，`item_int_feats_10=121` 的 support `4297`、正样本率 `0.675820`、lift `7.045899`。这些强信号会抬高随机 valid，但在线上 infer 覆盖收缩时也最容易变成 item memorization。

因此 item 侧要做三件事：高基数 item/sequence 字段使用 hash compression；对 item tower 加 dropout 或共享约束；显式加入 candidate item 与 user sequence 的 matching 特征，让模型学习“候选 item 是否匹配用户历史”，而不是只记住 item ID 的平均点击率。

### 序列长度漂移

训练集和推理集都属于长序列场景，空序列比例接近 `0`，但推理集的 `seq_c/seq_d` 更长：

| 序列域 | Train mean | Infer mean | Train p95 | Infer p95 | 结论 |
| ------ | ---------- | ---------- | --------- | --------- | ---- |
| `seq_c` | `512.818620` | `532.676313` | `1384` | `1479` | 推理更长，尾部更重 |
| `seq_d` | `2455.213142` | `2578.061668` | 接近 `4000` | 接近 `4000` | 推理整体右移，median `2697.5 -> 2940.5`，q1 `1292 -> 1449.75` |

训练集序列重复率很高：`seq_a=0.252239`、`seq_b=0.945815`、`seq_c=0.975259`、`seq_d=0.994539`。全长 attention 或只取固定 tail 都容易对长度和重复模式过敏。模型应同时使用 last-K、masked mean、recent token、sequence stats，并让 `seq_top_k` 成为真实的上限；`seq_b/seq_c/seq_d` 更适合去重、计数、hash-bag pooling 或轻量 memory，而不是全序列逐 token 学习。

### Dense 分布漂移

`user_dense_feats_62-66` 是本次最明显的 covariate shift。它们均值整体下降，标准差有的暴涨、有的下降，说明线上 infer 的 dense 数值尺度和尾部分布不同：

| 特征 | Train mean | Infer mean | Train std | Infer std | 结论 |
| ---- | ---------- | ---------- | --------- | --------- | ---- |
| `user_dense_feats_62` | `157584.211708` | `148791.725521` | `497528.347708` | `1730383.659216` | 方差暴涨，最危险 |
| `user_dense_feats_63` | `151848.923644` | `142169.143390` | `555086.051150` | `463069.831985` | 均值下降，方差下降 |
| `user_dense_feats_64` | `137591.687689` | `126591.212296` | `547448.764265` | `482969.123060` | 均值下降，方差下降 |
| `user_dense_feats_65` | `166409.104522` | `152456.244962` | `896489.323673` | `563584.249370` | 均值下降，方差下降 |
| `user_dense_feats_66` | `280508.077991` | `257412.537657` | `1101399.508611` | `865947.554984` | 均值下降，方差下降 |

这些 dense 不应裸喂 MLP。至少要输入 missing indicator；更稳的做法是加入 clip/log1p/标准化，或者把极端值分桶成 rank/bucket 特征。`user_dense_feats_87` 在训练集零值比例约 `0.620482`，zero 也应视为状态，而不是普通连续值。

### Token overlap / OOV sketch

EDA 会输出 `stats.token_overlap_sketch`，每个 sparse/sequence token 列保留最多 `ONLINE_EDA_TOKEN_OVERLAP_SKETCH_K=256` 个 KMV hash 指纹。infer 对比 train 时，`comparison.token_overlap_drift` 会给出：

- `jaccard`：两边 sketch 的近似重叠。
- `current_novel_sketch_rate`：infer sketch 中不在 train sketch 的比例，可看作近似 OOV 方向。
- `reference_only_sketch_rate`：train sketch 中未出现在 infer sketch 的比例。

这不是精确集合差，而是受控计算量的早期告警。它复用 cardinality 的 batch 内 token 样本，不额外做全量 set 存储，适合在线上日志中比较 item/token 覆盖变化。

### 建模与验证落地

这次对比之后，默认建模方向应从“随机 valid 高分”转向“隐藏 infer 泛化”：

1. 验证切分优先使用 `timestamp_auto`；没有可靠时间列时使用 `user_hash` 或 `sample_hash`，避免纯随机 batch valid 掩盖分布漂移。
2. sparse 缺失用显式 missing embedding，dense 缺失用 missing indicator，不把缺失和合法 `0` 混在一起。
3. 序列编码必须长度鲁棒：限制 `seq_top_k`、使用 sequence stats、recent + mean + matching 多路特征。
4. 降低 item memorization：高基数字段 hash compression，item tower 不单独主导输出，增加 candidate item 与 user sequence 的匹配约束。
5. 保留 token overlap/OOV sketch，每次 train/infer EDA 都比较覆盖变化，防止只看 schema 相同就误判数据一致。

### 运行性能

旧版全量 EDA 总耗时约 `12h21m`，第一遍扫描约 `11h50m`，第二遍约 `31m`。当前 Arrow 单遍版本在线上完整训练集约数分钟完成；推理集 `310,000` 行也在数分钟级完成。日常迭代建议仍先限制扫描量：

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
| `output_dir`            | None                      | 兼容 train/infer 请求；默认不写结果文件 |
| `reference_profile_path` | None                     | 用于 train/infer diff 的训练 profile |
| `reference_profile_json` | None                     | 从训练日志复制的单行 profile JSON |
| `dataset_role`          | `online`                  | `train` / `infer` / `online` |
| `batch_rows`            | 8192                      | PyArrow 每批读取行数      |
| `cardinality_sketch_k`  | 2048                      | KMV sketch 容量           |
| `token_overlap_sketch_k` | 256                      | token overlap/OOV sketch 容量 |
| `token_sample_limit_per_batch` | 20000             | 每列每 batch 进入基数估计的 token 上限 |
| `user_sample_limit`     | 20000                     | 用户跨域覆盖抽样行数上限  |
| `sequence_sample_size`  | 8192                      | 序列长度 reservoir 样本数 |
| `repeat_sample_rows_per_domain` | 20000            | 每个序列域重复率抽样行数上限 |
| `max_rows`              | None                      | 最大扫描行数              |
| `sample_percent`        | None                      | 按百分比限制扫描行数      |
| `progress_step_percent` | 10.0                      | 进度日志间隔              |
| `top_k` | 20 | 文本报告输出行数 |
| `enable_label_lift` | False | 是否启用 user/item token label lift |
| `label_feature_top_k` | 20 | label lift 输出行数 |
| `label_feature_min_support` | 20 | label lift token 最小样本数 |
| `label_feature_sample_rows` | 50000 | label lift 抽样行数 |

环境变量覆盖：

| 变量                        | 作用             |
| --------------------------- | ---------------- |
| `TAAC_SCHEMA_PATH`          | 显式 schema 路径 |
| `ONLINE_EDA_BATCH_ROWS`     | 覆盖 batch rows  |
| `ONLINE_EDA_MAX_ROWS`       | 限制最大扫描行数 |
| `ONLINE_EDA_SAMPLE_PERCENT` | 按百分比采样     |
| `ONLINE_EDA_ANALYSIS_LEVEL` | `fast` / `full` |
| `ONLINE_EDA_REFERENCE_PROFILE` | 显式 train profile 路径 |
| `ONLINE_EDA_REFERENCE_PROFILE_JSON` | 训练日志中的 `ONLINE_DATASET_EDA_RESULT=...` 单行 |
| `ONLINE_EDA_ENABLE_LABEL_LIFT` | 开关 label lift |
| `ONLINE_EDA_CARDINALITY_SKETCH_K` | 覆盖 KMV sketch 容量 |
| `ONLINE_EDA_TOKEN_OVERLAP_SKETCH_K` | 覆盖 token overlap/OOV sketch 容量 |
| `ONLINE_EDA_TOKEN_SAMPLE_LIMIT_PER_BATCH` | 覆盖每 batch token 采样上限 |
| `ONLINE_EDA_USER_SAMPLE_LIMIT` | 覆盖用户覆盖抽样上限 |
| `ONLINE_EDA_SEQUENCE_SAMPLE_SIZE` | 覆盖序列长度 reservoir 大小 |
| `ONLINE_EDA_REPEAT_SAMPLE_ROWS_PER_DOMAIN` | 覆盖重复率抽样上限 |
| `ONLINE_EDA_TOP_K` | 覆盖文本报告输出行数 |

`ONLINE_EDA_MAX_ROWS` 和 `ONLINE_EDA_SAMPLE_PERCENT` 互斥。

## 近似算法

为控制线上内存，EDA 不做全量重型分析：

- 稀疏基数使用 KMV sketch，输入来自每个 batch 的 Arrow unique token 样本。
- token overlap/OOV 使用更小的 KMV hash 指纹，复用基数统计的 token 样本。
- 序列 side-info token 基数也使用同样方法，不读取时间列。
- 序列长度分布使用 reservoir sampling。
- 序列重复率、用户活跃度、跨域覆盖和跨域重叠使用均匀行抽样。
- user/item token 与 label 的关联默认关闭；开启后在抽样行上做近似统计。

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
