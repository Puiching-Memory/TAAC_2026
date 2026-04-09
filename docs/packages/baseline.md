# Baseline

`config/gen/baseline` 现在不再承担“仓库里最强本地方案”的角色，而是明确回到它本来该做的事：当一个好读、好改、好复制的参考包。

## 设计目标

这个 baseline 主要服务三类人：

1. 第一次进入仓库、想先跑通完整链路的人。
2. 想基于现有 parquet batch contract 快速起一个自定义实验包的人。
3. 需要一个足够简单、但不是玩具级别的参考实现来做二次开发的人。

所以它优先保证的是：

1. 结构清晰。
2. 改动点明确。
3. 数据层和训练层契约稳定。
4. 关键模块有足够的注释解释“为什么这样分”。

## 当前结构

1. `__init__.py`
   定义默认配置与 `EXPERIMENT`，并把 baseline 明确命名为 `baseline_reference`。
2. `data.py`
   baseline 自己的 parquet 数据管线。它保持可读、可改，但不再作为其他实验包的共享数据入口。
3. `model.py`
   面向扩展的参考模型，采用“显式字段编码 + target-aware history pooling + 融合 MLP”的结构，而不是一开始就塞进复杂 unified backbone。
4. `utils.py`
   共享 loss / optimizer 装配，尽量让 starter 包保持最小认知负担。

## 为什么这样拆

之前的问题是模型身份已经偏向 grok 方案，但目录名仍叫 baseline，导致：

1. 新用户容易误以为 baseline 就代表推荐的默认架构方向。
2. baseline 的教学职责和其他实验包的研究语义容易混在一起。
3. 文档里很难同时表达“这是 starter”与“这是当前更激进的本地方案”。

现在这层职责被拆开了，而且数据边界也更明确：

1. `baseline`：starter / reference package。
2. `grok`：原来从 baseline 演化出来的本地研究方案。
3. 每个实验包自己维护 `data.py`，从原始数据读取到最终模型输入都留在包内。

## 推荐扩展点

如果你想基于这个 baseline 开自己的包，通常最自然的切入点有三类：

1. 替换 `model.py` 里的 `TargetAwareHistoryPool`，把 DIN 风格聚合换成 Transformer / cross-attention / state-space 等模块。
2. 保留 `data.py` 不动，只改 `model.py` 的字段编码与融合方式。
3. 保留 `data.py` 和 `utils.py`，在自己的包里只覆盖 `__init__.py` 和 `model.py`，先做最小可跑版本，再逐步加结构。

## 默认配置

1. 默认数据集：`data/datasets--TAAC2026--data_sample_1000/.../sample_data.parquet`
2. 默认输出目录：`outputs/gen/baseline`
3. 默认训练轮数：5
4. 默认 batch size：64

## 运行方式

```bash
uv run taac-train --experiment config/gen/baseline
uv run taac-evaluate single --experiment config/gen/baseline
uv run taac-search --experiment config/gen/baseline --trials 20
```

如果你只是想复制一个本地可改版本，最常见的做法不是直接改这个目录，而是新建一个包装实验包：

```python
from config.gen.baseline import EXPERIMENT

EXPERIMENT = EXPERIMENT.clone()
EXPERIMENT.name = "my_first_variant"
EXPERIMENT.train.output_dir = "outputs/custom/my_first_variant"
```

## 当前验证状态

这个 starter baseline 已经能直接运行，但在当前工作区里还没有新的、与本次重命名完全同口径的 smoke 记录。

因此：

1. 文档里不会把旧 `baseline` 的历史 grok 产物硬写成这个新 starter baseline 的结果。
2. 如果你要补新的 smoke 记录，建议直接落到 `outputs/smoke/baseline/summary.json` 或单独的手工输出目录，并同步更新 `docs/experiments.md`。
