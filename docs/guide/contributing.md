---
icon: lucide/git-branch-plus
---

# 新增实验包

如何创建和验证一个新的实验包。

## 最小目录

```
config/<experiment_name>/
├── __init__.py
├── model.py
└── ns_groups.json
```

三个文件缺一不可。`discover_experiment_paths()` 会扫描 `config/` 下所有包含这三个文件的目录。

## __init__.py

定义模块级 `EXPERIMENT` 对象：

```python
from pathlib import Path
from taac2026.infrastructure.pcvr.experiment import PCVRExperiment
from taac2026.infrastructure.pcvr.config import PCVRTrainConfig, PCVRModelConfig

EXPERIMENT = PCVRExperiment(
    name="pcvr_my_experiment",
    package_dir=Path(__file__).parent,
    model_class_name="MyModel",
    train_defaults=PCVRTrainConfig(
        model=PCVRModelConfig(
            num_blocks=3,
            num_heads=4,
            dropout_rate=0.02,
        ),
        ns=PCVRNSConfig(
            tokenizer_type="rankmixer",
            user_ns_tokens=5,
            item_ns_tokens=2,
        ),
    ),
)
```

关键字段：

- `name` -- 唯一标识，通常加 `pcvr_` 前缀
- `model_class_name` -- 必须与 `model.py` 中的类名完全一致
- `train_defaults` -- `PCVRTrainConfig` 实例，定义默认超参数

## model.py

实现模型类，继承 `EmbeddingParameterMixin`：

```python
import torch
import torch.nn as nn
from taac2026.infrastructure.pcvr.modeling import (
    EmbeddingParameterMixin,
    FeatureEmbeddingBank,
    NonSequentialTokenizer,
    SequenceTokenizer,
    DenseTokenProjector,
)
from taac2026.infrastructure.pcvr.protocol import ModelInput


class MyModel(EmbeddingParameterMixin, nn.Module):
    def __init__(
        self,
        user_int_feature_specs,
        item_int_feature_specs,
        user_dense_dim,
        item_dense_dim,
        seq_vocab_sizes,
        user_ns_groups,
        item_ns_groups,
        d_model=64,
        emb_dim=64,
        num_blocks=2,
        num_heads=4,
        # ... 其他参数
    ):
        super().__init__()
        # 初始化 Embedding、编码器、预测头等

    def forward(self, inputs: ModelInput) -> torch.Tensor:
        # 返回 (B,) 的 logits
        ...

    def predict(self, inputs: ModelInput):
        # 返回 (logits, embeddings)
        logits = self.forward(inputs)
        embeddings = self._get_embeddings(inputs)
        return logits, embeddings

    def reinit_high_cardinality_params(self):
        # 重初始化高基数 Embedding
        ...
```

## ns_groups.json

JSON 格式的 NS 特征分组：

```json
{
  "user_ns_groups": {
    "U1": [1, 15],
    "U2": [48, 49, 89, 90, 91],
    "U3": [80]
  },
  "item_ns_groups": {
    "I1": [11, 13],
    "I2": [5, 6, 7, 8, 12]
  }
}
```

特征 ID 是列名的数字后缀（`user_int_feats_1` -> fid 1）。

## 本地验证

```bash
# 1. 发现实验包
uv run python -c "from taac2026.infrastructure.experiments.discovery import discover_experiment_paths; print([p.name for p in discover_experiment_paths()])"

# 2. 加载实验包
uv run python -c "from taac2026.infrastructure.experiments.loader import load_experiment_package; exp = load_experiment_package('config/my_experiment'); print(exp.name)"

# 3. 训练 Smoke Test
uv run taac-train \
  --experiment config/my_experiment \
  --dataset-path data/sample_1000_raw/demo_1000.parquet \
  --schema-path data/sample_1000_raw/schema.json \
  --max-epochs 1

# 4. 运行单元测试
uv run pytest tests/unit/test_experiment_packages.py -v
```

## 修改现有包的检查清单

- [ ] `model_class_name` 与 `model.py` 中的类名一致
- [ ] `ns_groups.json` 中的特征 ID 在 schema 范围内
- [ ] `forward()` 返回 `(B,)` 形状的 logits
- [ ] `predict()` 返回 `(logits, embeddings)` 元组
- [ ] `get_sparse_params()` 和 `get_dense_params()` 正确分类参数
- [ ] 训练 Smoke Test 通过（1 epoch）
- [ ] 单元测试通过
