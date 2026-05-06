---
icon: lucide/database-zap
---

# PCVR 数据管道

PCVR 数据管道负责把 parquet 数据变成训练 batch，并在需要时叠加 cache、增强、shuffle 和 observed schema 统计。数据管道不是一个单独命令，而是训练 workflow 通过 `PCVRTrainConfig.data_pipeline` 注入到 `get_pcvr_data()` 的运行时能力。

## 什么时候看这页

- 你要解释 Baseline 和 Baseline+ 为什么数据侧行为不同。
- 你想给新实验加数据增强或 cache。
- 你在排查吞吐、内存或线上数据分布问题。

## 本地和线上数据路径

本地 PCVR 训练、评估和推理默认使用仓库管理的 demo 数据，不接受显式 `--dataset-path`。这可以避免本地 smoke 命令被临时数据路径污染。

线上 bundle 由平台环境变量注入真实数据：

- 训练 / 评估：`TRAIN_DATA_PATH`
- 推理：`EVAL_DATA_PATH`
- schema：`TAAC_SCHEMA_PATH`

维护类实验例外，例如 [Online Dataset EDA](../experiments/online-dataset-eda.md) 本身就是为了读取指定数据集。

## 当前组件

数据管道配置位于 `PCVRTrainConfig.data_pipeline`，类型定义在 `src/taac2026/domain/config.py`。

| 组件                      | 作用                       | 常见用途                       |
| ------------------------- | -------------------------- | ------------------------------ |
| `PCVRDataCacheConfig`     | 缓存已转换的基础 batch     | 加速重复访问，减少数据转换开销 |
| `PCVRSequenceCropConfig`  | 为序列生成尾窗或随机窗口   | 控制有效序列长度，做轻量增强   |
| `PCVRFeatureMaskConfig`   | 随机置零稀疏特征和序列事件 | 提升鲁棒性                     |
| `PCVRDomainDropoutConfig` | 按行丢弃整个序列域         | 模拟缺失域，减少单域依赖       |

cache 有三种模式：

- `none`：关闭。
- `memory`：普通 LRU。
- `opt`：按已知访问轨迹做 OPT 淘汰；不满足条件时会退回安全策略。

配置对象的当前字段：

```python
PCVRDataPipelineConfig(
    cache=PCVRDataCacheConfig(mode="none", max_batches=0),
    transforms=(),
    seed=None,
    strict_time_filter=True,
)
```

transform 配置字段：

```python
PCVRSequenceCropConfig(
    enabled=True,
    views_per_row=1,
    seq_window_mode="tail",      # tail / random_tail / rolling
    seq_window_min_len=1,
)

PCVRFeatureMaskConfig(
    enabled=True,
    probability=0.0,
)

PCVRDomainDropoutConfig(
    enabled=True,
    probability=0.0,
)
```

`PCVRDataPipelineConfig.enabled` 只表示是否存在启用的 transform；cache 是否启用由 `cache.enabled` 单独判断。

## 执行顺序

训练数据从 parquet 到模型输入大致经历这些阶段：

1. `get_pcvr_data()` 解析 parquet 文件或目录，读取 schema。
2. dataset 按 row group / batch 构造基础 PCVR batch。
3. base batch cache 尝试命中；命中后返回 clone，避免增强污染缓存。
4. transform 依次作用在 batch 上。
5. shuffle / worker / prefetch 继续推进 dataloader。
6. training workflow 把 batch 转成 `ModelInput`。

验证和推理不应该启用随机增强。训练增强通过实验包默认配置进入，不应该在评估入口临时拼出来。

## Transform 语义

`PCVRSequenceCropTransform` 会对每个序列域裁剪窗口：

- `tail`：保留尾部窗口。
- `random_tail`：随机选择窗口长度，但窗口仍贴近尾部。
- `rolling`：随机选择窗口起点。
- `views_per_row > 1` 时会复制行，形成多视图训练样本。

`PCVRFeatureMaskTransform` 会随机置零：

- `user_int_feats`
- `item_int_feats`
- 各序列域里的事件 token
- 对序列事件 mask 后会 compact 序列，更新长度和 time bucket。

`PCVRDomainDropoutTransform` 会按行丢弃整个序列域：

- sequence 置零。
- 对应 length 置零。
- time bucket 如果存在也置零。

所有随机 transform 都使用传入的 `torch.Generator`，seed 来自 pipeline 配置和数据读取上下文。

## Cache 语义

cache 存的是增强前的基础 batch。

`memory` 模式是 LRU。适合小数据或重复访问明显的场景。

`opt` 模式需要已知访问 trace。trace 满足预期时按下一次使用距离淘汰；如果发现重复 key 或访问顺序不符合预期，会退回安全行为。多 worker 场景下可能使用 shared-memory cache，具体实现见：

- `PCVRMemoryBatchCache`
- `PCVRSharedBatchCache`

cache 返回 batch clone，因此后续 transform 不会修改缓存里的基础样本。

## Baseline 和 Baseline+

Baseline 关闭数据增强和 cache：

```python
PCVRDataPipelineConfig(
    cache=PCVRDataCacheConfig(mode="none", max_batches=0),
    transforms=(),
)
```

Baseline+ 默认打开更激进的组合：

```python
PCVRDataPipelineConfig(
    cache=PCVRDataCacheConfig(mode="opt", max_batches=512),
    transforms=(
        PCVRSequenceCropConfig(views_per_row=2, seq_window_mode="random_tail", seq_window_min_len=8),
        PCVRFeatureMaskConfig(probability=0.03),
        PCVRDomainDropoutConfig(probability=0.03),
    ),
    seed=42,
)
```

做消融时不要只比较实验名；先确认模型、数据增强、cache 和 backend 哪些同时变了。

## 改配置的位置

普通实验只需要改自己的入口文件：

```text
experiments/<name>/__init__.py
```

共享实现位于：

- 配置对象：`src/taac2026/domain/config.py`
- batch transform：`src/taac2026/infrastructure/data/transforms.py`
- cache：`src/taac2026/infrastructure/data/cache.py`
- 数据集读取：`src/taac2026/infrastructure/data/dataset.py`
- batch 类型：`src/taac2026/infrastructure/data/batches.py`

如果只是调整某个实验的增强策略，改实验包的 `TRAIN_DEFAULTS`。如果要新增一种 transform，再改 `domain/config.py` 和 `infrastructure/data/transforms.py`，并补测试。

## 吞吐压测

如果你改了数据管道，先用 benchmark 做一个小口径对比：

```bash
uv run taac-benchmark-pcvr-data-pipeline \
  --dataset-path data/sample_1000_raw/demo_1000.parquet \
  --schema-path data/sample_1000_raw/schema.json \
  --preset augment
```

`preset` 可选 `none`、`cache`、`augment`。正式性能结论应使用更大的合成数据，并在当前代码和当前环境下重新生成；推荐把 JSON 输出放到 `outputs/benchmarks/`，同时记录 commit、硬件、CUDA / PyTorch 版本和完整命令。

## 最小复核

```bash
uv run pytest tests/unit/infrastructure/data/test_augmentation.py -q
uv run pytest tests/unit/infrastructure/data/test_split.py -q
uv run pytest tests/unit/experiments/test_runtime_contract_matrix.py -q
```

如果改了 cache 或 dataloader 行为，再跑：

```bash
uv run pytest tests/unit/application/training/test_cli.py -q
uv run pytest tests/unit/application/experiments/test_pcvr_runtime.py -q
```
