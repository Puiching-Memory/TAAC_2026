---
icon: lucide/git-branch-plus
---

# 新增实验包

如何在当前共享 PCVR 运行时上新增或修改一个实验包。

## 最小目录

```text
experiments/pcvr/<experiment_name>/
├── __init__.py
└── model.py
```

最小契约就是这两个文件。需要包内辅助层时，可以继续增加 `layers.py` 之类的局部模块。

## __init__.py

当前 `PCVRExperiment` 的真实构造参数不止 `name` 和 `train_defaults`。最稳妥的方式是从一个现有实验包复制，再改成你的模型和默认配置。

一个最小可工作的骨架通常至少包含：

```python
from pathlib import Path

from taac2026.infrastructure.pcvr.config import PCVRModelConfig, PCVRNSConfig, PCVRTrainConfig
from taac2026.infrastructure.pcvr.experiment import PCVRExperiment
from taac2026.infrastructure.pcvr.prediction_stack import PREDICTION_HOOKS  # 示例占位，通常复用现有包的默认 hooks
from taac2026.infrastructure.pcvr.runtime_stack import RUNTIME_HOOKS        # 示例占位
from taac2026.infrastructure.pcvr.train_stack import TRAIN_HOOKS            # 示例占位
from taac2026.infrastructure.pcvr.training import parse_pcvr_train_args


TRAIN_DEFAULTS = PCVRTrainConfig(
    model=PCVRModelConfig(
        num_blocks=2,
        num_heads=4,
        dropout_rate=0.02,
    ),
    ns=PCVRNSConfig(
        grouping_strategy="explicit",
        user_groups={"U1": [1, 15]},
        item_groups={"I1": [11, 13]},
        tokenizer_type="rankmixer",
        user_tokens=5,
        item_tokens=2,
    ),
)

EXPERIMENT = PCVRExperiment(
    name="pcvr_my_experiment",
    package_dir=Path(__file__).resolve().parent,
    model_class_name="MyModel",
    train_defaults=TRAIN_DEFAULTS,
    train_arg_parser=parse_pcvr_train_args,
    train_hooks=TRAIN_HOOKS,
    prediction_hooks=PREDICTION_HOOKS,
    runtime_hooks=RUNTIME_HOOKS,
)
```

关键字段：

- `name` -- 唯一标识，通常加 `pcvr_` 前缀
- `package_dir` -- 推荐使用 `Path(__file__).resolve().parent`
- `model_class_name` -- 必须与 `model.py` 中的类名完全一致
- `train_defaults` -- 当前默认训练配置，包括数据、优化器、runtime 和 NS 分组
- `train_arg_parser` / `train_hooks` / `prediction_hooks` / `runtime_hooks` -- 通常先复用现有实验包的默认实现

如果你不是在做新的运行时扩展，而只是接一个新模型，优先复用已有 hook，而不是重新造一套训练 / 推理流程。

## model.py

`model.py` 只负责当前实验自己的模型实现。建议复用共享建模层里的基础组件。

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

    如果模型需要包内私有组件，可以像 `symbiosis` 一样把复杂层拆到同目录下的 `layers.py`。

## NS 分组配置

    当前仓库使用 `PCVRNSConfig` 在 `__init__.py` 中显式声明 NS 分组：

```python
ns=PCVRNSConfig(
        grouping_strategy="explicit",
        user_groups={
        "U1": [1, 15],
        "U2": [48, 49, 89, 90, 91],
        },
        item_groups={
        "I1": [11, 13],
        "I2": [5, 6, 7, 8, 12],
        },
        tokenizer_type="rankmixer",
        user_tokens=5,
        item_tokens=2,
)
```

特征 ID 是列名的数字后缀（`user_int_feats_1` -> fid 1）。

    当前仓库不再要求每个实验包必须自带独立的 `ns_groups.json` 文件。

## 本地验证

```bash
# 1. 发现实验包
    uv run python -c "from pathlib import Path; from taac2026.infrastructure.experiments.discovery import discover_experiment_paths; print(discover_experiment_paths(Path('experiments/pcvr')))"

# 2. 加载实验包
uv run python -c "from taac2026.infrastructure.experiments.loader import load_experiment_package; exp = load_experiment_package('experiments/pcvr/my_experiment'); print(exp.name)"

# 3. 训练 Smoke Test
uv run taac-train \
      --experiment experiments/pcvr/my_experiment \
  --dataset-path data/sample_1000_raw/demo_1000.parquet \
  --schema-path data/sample_1000_raw/schema.json \
      --run-dir outputs/my_experiment_smoke \
      --device cpu \
      --num_workers 0 \
      --batch_size 8 \
      --max_steps 1

    # 4. 运行最小相关单测
    uv run pytest tests/unit/infrastructure/experiments/test_experiment_packages.py -v
    uv run pytest tests/unit/infrastructure/experiments/test_experiment_discovery.py -v
    uv run pytest tests/unit/infrastructure/pcvr/test_runtime_contract_matrix.py -v
```

## 修改现有包的检查清单

- [ ] `model_class_name` 与 `model.py` 中的类名一致
- [ ] `PCVRNSConfig` 中声明的特征 ID 在 schema 范围内
- [ ] `forward()` 返回 `(B,)` 形状的 logits
- [ ] `predict()` 返回 `(logits, embeddings)` 元组
- [ ] `get_sparse_params()` 和 `get_dense_params()` 正确分类参数
- [ ] 训练 Smoke Test 通过（建议 `--device cpu --max_steps 1`）
- [ ] 最小相关单元测试通过
- [ ] 如涉及打包或 runtime sidecar，补跑 `tests/unit/application/test_package_training.py` 或 `tests/unit/application/test_package_inference.py`
