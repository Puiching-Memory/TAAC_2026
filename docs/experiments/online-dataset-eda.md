---
icon: lucide/bar-chart
---

# Online Dataset EDA

## 摘要

Online Dataset EDA 是一个维护类实验包，用来在训练和推理环境中流式扫描 parquet 数据，输出可比较的文本 profile。它不训练模型、不依赖可视化库，也不把全量数据读进内存；它的目标是快速回答：线上 train 与 infer 数据是否真的同分布。

当前默认模式从“全量精细报表”转为“快速生成可比较 profile”。完整结果只打印到 stdout 的单行：

```text
ONLINE_DATASET_EDA_RESULT={...}
```

这使它适合在平台日志中复制、归档和做 train/infer diff。

## 一、为什么需要它

推荐模型线上掉分时，schema mismatch 往往不是唯一原因。更常见的是 schema 相同，但分布已经变了：

- item 侧高基数字段覆盖收缩。
- 用户或物品字段缺失率漂移。
- dense 特征均值、方差和零值比例变化。
- 长序列长度分布右移。
- infer token 与 train token overlap 降低。

这些变化会让随机 valid AUC 虚高，尤其是依赖 item memorization、dense shortcut 或固定 tail 序列的模型。Online Dataset EDA 把这些风险变成结构化 profile。

## 二、实验契约

入口文件 `experiments/online_dataset_eda/__init__.py` 导出：

```python
EXPERIMENT = ExperimentSpec(
    name="online_dataset_eda",
    kind="maintenance",
    requires_dataset=True,
)
```

关键契约：

- 支持训练 bundle 和推理 bundle。
- 必须有 parquet 数据路径。
- `train_fn` 会拒绝 extra args；采样和 batch 参数通过 runner config 或环境变量控制。
- 默认只向 stdout 打印完整 JSON，不写 profile / drift 文件。
- 推理阶段可通过 `ONLINE_EDA_REFERENCE_PROFILE_JSON` 传入训练 profile，直接输出 diff。

## 三、本地运行

本地扫描 demo parquet：

```bash
bash run.sh train \
  --experiment experiments/online_dataset_eda \
  --dataset-path data/sample_1000_raw/demo_1000.parquet \
  --schema-path docs/archive/files/schema/sample_1000_raw.schema.json
```

`--dataset-path` 可以是单个 parquet 文件，也可以是包含多个 parquet 的目录。

## 四、线上运行

训练侧 bundle：

```bash
uv run taac-package-train \
  --experiment experiments/online_dataset_eda \
  --output-dir outputs/bundles/online_dataset_eda
```

训练侧平台环境：

```bash
export TRAIN_DATA_PATH=/path/to/train_parquet_or_dir
export TAAC_SCHEMA_PATH=/path/to/schema.json
bash run.sh
```

推理侧 bundle：

```bash
uv run taac-package-infer \
  --experiment experiments/online_dataset_eda \
  --output-dir outputs/bundles/online_dataset_eda_infer
```

推理侧平台环境：

```bash
export EVAL_DATA_PATH=/path/to/infer_parquet_or_dir
export EVAL_RESULT_PATH=/path/to/result_dir
export ONLINE_EDA_REFERENCE_PROFILE_JSON='ONLINE_DATASET_EDA_RESULT={...}'
export TAAC_SCHEMA_PATH=/path/to/schema.json
python infer.py
```

`ONLINE_EDA_REFERENCE_PROFILE_JSON` 可以直接粘贴训练日志中的整行结果。本地调试时，也可以用 `ONLINE_EDA_REFERENCE_PROFILE` 指向手工保存的训练 profile 文件。

## 五、报告结构

输出 JSON 的重点字段：

| 字段                         | 说明                                                                  |
| ---------------------------- | --------------------------------------------------------------------- |
| `dataset_path`               | 实际扫描的数据路径。                                                  |
| `row_count` / `sampled`      | 扫描行数和采样状态。                                                  |
| `stats.column_layout`        | scalar、user_int、user_dense、item_int、sequence 的列数和序列域布局。 |
| `stats.label_distribution`   | `label_type == 2` 的正负样本、缺失和正样本率。                        |
| `stats.null_rates`           | 高缺失率字段。                                                        |
| `stats.cardinality`          | 稀疏字段基数估计。                                                    |
| `stats.sequence_lengths`     | 各序列域长度分布。                                                    |
| `stats.dense_distributions`  | dense 均值、方差、标准差、零值比例。                                  |
| `stats.token_overlap_sketch` | train/infer token overlap 与近似 OOV 对比所需 KMV 指纹。              |
| `comparison.*`               | 推理 profile 与训练 profile 的差异摘要。                              |
| `approximation`              | 近似统计的采样参数。                                                  |

自动化 summary 只保留少量字段，例如：

```python
{
    "experiment_name": "online_dataset_eda",
    "dataset_role": "train" | "infer" | "online",
    "row_count": ...,
    "sampled": ...,
    "stdout_result": True,
}
```

真正分析应读取 stdout 中的 `ONLINE_DATASET_EDA_RESULT=` 单行。

## 六、性能策略

当前 runner 默认使用 Arrow compute 单遍扫描，默认 `batch_rows=8192`。默认 fast profile 会保留：

- 缺失率、标签分布、列布局。
- 稀疏和序列 side-info token 基数。
- token overlap/OOV KMV sketch。
- 序列长度分布、重复率抽样和用户跨域覆盖抽样。
- dense 均值、方差、标准差和零值比例。

默认关闭重型 label lift。需要最终确认强特征时再打开：

```bash
export ONLINE_EDA_ENABLE_LABEL_LIFT=1
```

或使用 full 模式：

```bash
export ONLINE_EDA_ANALYSIS_LEVEL=full
```

大数据上建议先限制扫描量：

```bash
export ONLINE_EDA_MAX_ROWS=100000
```

或按比例采样：

```bash
export ONLINE_EDA_SAMPLE_PERCENT=5
```

`ONLINE_EDA_MAX_ROWS` 与 `ONLINE_EDA_SAMPLE_PERCENT` 互斥。

## 七、近似算法

为控制线上内存，EDA 使用近似统计：

- 稀疏基数使用 KMV sketch，输入来自每个 batch 的 Arrow unique token 样本。
- token overlap/OOV 使用更小的 KMV hash 指纹，复用基数统计的 token 样本。
- 序列 side-info token 基数不读取时间列。
- 序列长度分布使用 reservoir sampling。
- 序列重复率、用户活跃度、跨域覆盖和跨域重叠使用均匀行抽样。
- user/item token 与 label 的关联默认关闭；开启后在抽样行上做近似统计。

这些结果适合判断量级、方向和异常，不适合作为精确离线统计报表。

## 八、2026-05-12 线上对比结论

这次线上 EDA 分别扫描训练集 `/data_ams/academic_training_data` 和推理集 `/data_ams/academic_infer_data`。训练集 `1,010,000` 行，推理集 `310,000` 行。两边 schema 完全一致，`schema_signature=0f427b26ed744814222527d2`，字段布局也一致：

| 类型             | 数量                                          |
| ---------------- | --------------------------------------------- |
| scalar           | `5`                                           |
| user_int         | `46`                                          |
| user_dense       | `10`                                          |
| item_int         | `14`                                          |
| sequence         | `45`                                          |
| sequence domains | `seq_a=9`，`seq_b=14`，`seq_c=12`，`seq_d=10` |

结论不是 schema mismatch，而是分布漂移。推理阶段出现 `predictions.json` 缺失，是维护类 EDA 实验没有生成预测文件导致的评测外壳错误，不影响 EDA 结果。

### 标签与样本规模

| 项目             | Train                      | Infer                      | 结论                           |
| ---------------- | -------------------------- | -------------------------- | ------------------------------ |
| 行数             | `1,010,000`                | `310,000`                  | 推理集约为训练集 `30.7%`       |
| `label_type`     | 正样本率 `0.095917`        | 全缺失                     | 推理集无标签是正常线上评测形态 |
| schema signature | `0f427b26ed744814222527d2` | `0f427b26ed744814222527d2` | schema 一致                    |

训练正样本率约 `1:9.43`。输出层 bias、calibration 和 early stopping 应围绕这个先验做本地验证；推理集没有 label，不能直接估计点击率变化。

### 缺失率漂移

| 特征                | Train null_rate | Infer null_rate | 变化        |
| ------------------- | --------------- | --------------- | ----------- |
| `item_int_feats_11` | `0.496732`      | `0.531232`      | `+0.034500` |
| `user_int_feats_15` | `0.141426`      | `0.110219`      | `-0.031207` |
| `user_int_feats_80` | `0.246279`      | `0.272871`      | `+0.026592` |
| `user_int_feats_92` | `0.484775`      | `0.509374`      | `+0.024599` |

训练集本身还有极高缺失列：`user_int_feats_101=0.920411`、`user_int_feats_102=0.895672`、`user_int_feats_103=0.882711`、`item_int_feats_83/84/85=0.879531`。缺失不能简单压成合法 `0`；missing embedding 和 missing indicator 应成为模型输入的一部分。

### Item 基数漂移

推理集 item token 覆盖明显收缩：

| 特征                | Train cardinality | Infer cardinality | 结论                 |
| ------------------- | ----------------- | ----------------- | -------------------- |
| `item_int_feats_16` | `22,192`          | `14,280`          | 强 item 特征覆盖变窄 |
| `item_int_feats_11` | `20,301`          | `13,473`          | 与缺失率同时漂移     |
| `item_int_feats_8`  | `2,042`           | `1,648`           | 中等 item 特征收缩   |
| `item_int_feats_7`  | `2,303`           | `1,933`           | 中等 item 特征收缩   |
| `item_int_feats_12` | `2,305`           | `1,933`           | 中等 item 特征收缩   |

训练集 item 侧标签信号很强，例如 `item_int_feats_16=20543` 的 support `444`、正样本率 `0.979730`、lift `10.214367`。这类信号会抬高随机 valid，也最容易在线上覆盖收缩时退化为 item memorization。

### 序列长度漂移

| 序列域  | Train mean    | Infer mean    | Train p95   | Infer p95   | 结论               |
| ------- | ------------- | ------------- | ----------- | ----------- | ------------------ |
| `seq_c` | `512.818620`  | `532.676313`  | `1384`      | `1479`      | 推理更长，尾部更重 |
| `seq_d` | `2455.213142` | `2578.061668` | 接近 `4000` | 接近 `4000` | 推理整体右移       |

训练集序列重复率很高：`seq_a=0.252239`、`seq_b=0.945815`、`seq_c=0.975259`、`seq_d=0.994539`。全长 attention 或固定 tail 都可能过敏；模型应同时使用 last-K、masked mean、recent token、sequence stats 或 memory selector。

### Dense 分布漂移

`user_dense_feats_62-66` 是最明显的 covariate shift：

| 特征                  | Train mean      | Infer mean      | Train std        | Infer std        | 结论     |
| --------------------- | --------------- | --------------- | ---------------- | ---------------- | -------- |
| `user_dense_feats_62` | `157584.211708` | `148791.725521` | `497528.347708`  | `1730383.659216` | 方差暴涨 |
| `user_dense_feats_63` | `151848.923644` | `142169.143390` | `555086.051150`  | `463069.831985`  | 均值下降 |
| `user_dense_feats_64` | `137591.687689` | `126591.212296` | `547448.764265`  | `482969.123060`  | 均值下降 |
| `user_dense_feats_65` | `166409.104522` | `152456.244962` | `896489.323673`  | `563584.249370`  | 均值下降 |
| `user_dense_feats_66` | `280508.077991` | `257412.537657` | `1101399.508611` | `865947.554984`  | 均值下降 |

Dense 不应裸喂 MLP。至少要加 missing indicator；更稳的做法是 clip/log1p/标准化，或把极端值分桶成 rank/bucket 特征。

## 九、建模落地建议

EDA 结论直接影响实验包设计：

1. 验证切分优先使用 `timestamp_auto`；没有可靠时间列时用 `user_hash` 或 `sample_hash`。
2. sparse 缺失用 missing embedding，dense 缺失用 missing indicator。
3. 长序列使用 `seq_top_k`、sequence stats、recent + memory 或 matching 多路特征。
4. 降低 item memorization：高基数字段 hash compression，item tower 不单独主导输出。
5. 保留 token overlap/OOV sketch，每次 train/infer EDA 都比较覆盖变化。

这也是 Symbiosis 引入 metadata token、risk token、memory selector 和 missing token 的主要原因。

## 十、Runner 配置

主要配置与环境变量：

| 配置 / 环境变量                                                | 作用                            |
| -------------------------------------------------------------- | ------------------------------- |
| `batch_rows` / `ONLINE_EDA_BATCH_ROWS`                         | Arrow 每批读取行数，默认 `8192` |
| `max_rows` / `ONLINE_EDA_MAX_ROWS`                             | 最大扫描行数                    |
| `sample_percent` / `ONLINE_EDA_SAMPLE_PERCENT`                 | 按百分比采样                    |
| `analysis_level` / `ONLINE_EDA_ANALYSIS_LEVEL`                 | `fast` / `full`                 |
| `enable_label_lift` / `ONLINE_EDA_ENABLE_LABEL_LIFT`           | 是否启用 label lift             |
| `cardinality_sketch_k` / `ONLINE_EDA_CARDINALITY_SKETCH_K`     | KMV sketch 容量                 |
| `token_overlap_sketch_k` / `ONLINE_EDA_TOKEN_OVERLAP_SKETCH_K` | overlap/OOV sketch 容量         |
| `reference_profile_json` / `ONLINE_EDA_REFERENCE_PROFILE_JSON` | 训练日志中的 profile JSON       |
| `reference_profile_path` / `ONLINE_EDA_REFERENCE_PROFILE`      | 本地保存的训练 profile          |
| `TAAC_SCHEMA_PATH`                                             | 显式 schema 路径                |

源码入口：

- `experiments/online_dataset_eda/__init__.py`
- `experiments/online_dataset_eda/runner.py`
- `tests/contract/experiments/test_online_dataset_eda_runner.py`

## 十一、验收

```bash
uv run pytest tests/contract/experiments/test_maintenance_experiments.py -q
uv run pytest tests/contract/experiments/test_online_dataset_eda_runner.py -q
```
