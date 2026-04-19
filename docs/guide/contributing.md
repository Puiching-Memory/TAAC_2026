---
icon: lucide/git-branch-plus
---

# 开发指南：新增实验包

## 当前约定

新增实验包时，默认优先复用框架层的共享实现。只有当默认 builder 不满足需求时，才额外创建 `data.py` 或 `utils.py`。

### 推荐目录结构

最小接入通常只需要两个文件：

```
config/my_experiment/
├── __init__.py    # 导出 EXPERIMENT
└── model.py       # build_model_component（模型架构）
```

当你需要覆盖默认行为时，再按需增加：

```
config/my_experiment/
├── __init__.py
├── model.py
├── data.py        # 仅在默认数据管道不够用时新增
└── utils.py       # 仅在默认 loss / optimizer 不够用时新增
```

## 第 1 步：创建 `__init__.py`

```python
from __future__ import annotations

from pathlib import Path

from taac2026.domain.config import DataConfig, ModelConfig, TrainConfig
from taac2026.domain.experiment import ExperimentSpec
from taac2026.domain.features import build_default_feature_schema

from .model import build_model_component

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "data" / "datasets--TAAC2026--data_sample_1000"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "config" / "my_experiment"

EXPERIMENT = ExperimentSpec(
    name="my_experiment",
    data=DataConfig(
        dataset_path=str(DEFAULT_DATASET),
        max_seq_len=32,
        max_feature_tokens=16,
        max_event_features=4,
        stream_batch_rows=256,
        val_ratio=0.2,
        label_action_type=2,
        dense_feature_dim=16,
    ),
    model=ModelConfig(
        name="my_experiment",
        vocab_size=131072,
        embedding_dim=128,
        hidden_dim=128,
        dropout=0.1,
        num_layers=4,
        num_heads=4,
        recent_seq_len=32,
        memory_slots=2,
        ffn_multiplier=4.0,
        feature_cross_layers=1,
        sequence_layers=1,
        static_layers=1,
        query_decoder_layers=0,
        fusion_layers=1,
        num_queries=0,
        head_hidden_dim=128,
        segment_count=4,
    ),
    train=TrainConfig(
        seed=7,
        epochs=10,
        batch_size=64,
        eval_batch_size=64,
        num_workers=0,
        output_dir=str(DEFAULT_OUTPUT_DIR),
        learning_rate=1.0e-3,
        weight_decay=1.0e-4,
    ),
    build_data_pipeline=None,
    build_model_component=build_model_component,
    build_loss_stack=None,
    build_optimizer_component=None,
    switches={"logging": True, "visualization": True},
)

EXPERIMENT.feature_schema = build_default_feature_schema(EXPERIMENT.data, EXPERIMENT.model)
```

这里的关键点是：

- `build_model_component` 始终由实验包自己提供
- `build_data_pipeline=None` 会走框架默认 KJT / sparse pipeline
- `build_loss_stack=None` 会走默认 ranking loss
- `build_optimizer_component=None` 会走默认 optimizer builder

## 第 2 步：实现 `model.py`

```python
from __future__ import annotations

import torch
from torch import nn

from taac2026.domain.features import build_default_feature_schema
from taac2026.domain.types import BatchTensors
from taac2026.infrastructure.nn.embedding import TorchRecEmbeddingBagAdapter
from taac2026.infrastructure.nn.heads import ClassificationHead
from taac2026.infrastructure.nn.pooling import masked_mean


class MyExperimentModel(nn.Module):
    def __init__(self, data_config, model_config, dense_dim: int, feature_schema) -> None:
        super().__init__()
        self.sparse_embedding = TorchRecEmbeddingBagAdapter(
            feature_schema,
            table_names=("user_tokens", "candidate_tokens", "context_tokens"),
        )
        self.encoder = nn.Linear(self.sparse_embedding.output_dim + dense_dim, model_config.hidden_dim)
        self.output = ClassificationHead(model_config.hidden_dim)

    def forward(self, batch: BatchTensors) -> torch.Tensor:
        sparse = self.sparse_embedding(batch.sparse_features)
        fused = torch.cat([sparse, batch.dense_features], dim=-1)
        hidden = torch.relu(self.encoder(fused))
        return self.output(hidden)


def build_model_component(data_config, model_config, dense_dim):
    feature_schema = build_default_feature_schema(data_config, model_config)
    return MyExperimentModel(data_config, model_config, dense_dim, feature_schema=feature_schema)
```

实际模型通常会：

- 消费 `batch.sparse_features`
- 按需读取 `batch.sequence_features`
- 使用共享的 `ClassificationHead`、`TargetAwarePool`、`RMSNorm` 等组件

如果你需要从 schema 选择特定表，优先从 `EXPERIMENT.feature_schema` 派生，而不是重新手写一套 token 约定。

## 何时创建 `data.py`

只有在以下情况才建议覆盖默认数据管道：

- 需要与默认 `FeatureSchema` 不兼容的输入表示
- 需要额外的自定义样本级变换
- 需要特殊的 collate / sampler 行为

覆盖时，函数签名仍然必须返回：

```python
def build_data_pipeline(data_config, model_config, train_config):
    return train_loader, val_loader, data_stats
```

## 何时创建 `utils.py`

以下情况适合保留自定义 `utils.py`：

- 需要自定义 auxiliary loss
- 需要特殊优化器分组或非默认优化器
- 需要对特定参数应用不同更新规则

当前仓库里的真实例子：

- DeepContextNet：保留自定义 optimizer builder
- UniRec：保留自定义 optimizer builder
- UniScaleFormer：保留自定义 loss builder

!!! important "不要跨实验包复用 utils"
    如果实验包需要自定义 builder，应当在自己的 `utils.py` 中显式定义或重新导出。测试会校验 builder 的模块归属，避免不同实验包之间出现隐式耦合。

## 验证流程

```bash
# 1. 检查 ExperimentSpec / 默认 builder / 前向契约
uv run pytest tests/test_experiment_packages.py -q

# 2. 跑一次最小训练
uv run taac-train --experiment config/my_experiment

# 3. 跑一次评估
uv run taac-evaluate single --experiment config/my_experiment
```

如果你新增了测试文件，还必须把文件名登记到 `tests/conftest.py` 的 `UNIT_TEST_FILES`、`INTEGRATION_TEST_FILES` 或 `GPU_TEST_FILES` 集合里，否则 pytest 收集会直接失败。

## 数据与 schema 约定

默认数据集路径仍指向 HuggingFace 缓存结构：

```
data/datasets--TAAC2026--data_sample_1000/
```

框架会优先解析本地缓存并读取 `refs/main` 对应快照；如果本地缓存缺失，默认数据管道会按数据集名回退下载。

`feature_schema` 建议通过 `build_default_feature_schema()` 派生，再根据实验需求做局部调整，而不是从零复制整套表定义。
