---
icon: lucide/git-branch-plus
---

# 新增实验包

新增 PCVR 实验包时，默认只写包内模型和配置，训练、评估、推理、checkpoint 和线上打包都复用共享 runtime。

## 最小目录

```text
config/my_experiment/
├── __init__.py
├── model.py
└── ns_groups.json
```

不要在实验包里新增 `run.sh`、`train.py`、`trainer.py` 或复制共享 dataloader。除非确实要改变 runtime 行为，否则实验包只负责模型本身。

## __init__.py

`__init__.py` 导出 `EXPERIMENT = PCVRExperiment(...)`：

```python
from __future__ import annotations

from pathlib import Path

from taac2026.infrastructure.pcvr.config import PCVRModelConfig, PCVRNSConfig, PCVRTrainConfig
from taac2026.infrastructure.pcvr.experiment import PCVRExperiment


EXPERIMENT = PCVRExperiment(
    name="pcvr_my_experiment",
    package_dir=Path(__file__).resolve().parent,
    model_class_name="PCVRMyExperiment",
    train_defaults=PCVRTrainConfig(
        model=PCVRModelConfig(num_blocks=2, num_heads=4),
        ns=PCVRNSConfig(groups_json="ns_groups.json"),
    ),
)

__all__ = ["EXPERIMENT"]
```

关键点：

- `model_class_name` 必须与 `model.py` 中导出的类名一致。
- 非 baseline 实验使用自己的类名，例如 `PCVRInterFormer`、`PCVROneTrans` 或 `PCVRMyExperiment`。
- `train_defaults` 里用 `PCVRNSConfig(groups_json="ns_groups.json")` 显式启用包内 `ns_groups.json`。
- 新参数优先放进 `PCVRTrainConfig` 下已有的 typed 子配置，例如 `PCVRModelConfig.num_blocks`、`PCVRModelConfig.d_model`、`PCVRDataConfig.batch_size`。

## model.py

`model.py` 必须暴露 `ModelInput` 和 `model_class_name` 指定的模型类。模型构造函数由 `build_pcvr_model` 调用，签名应接受共享 runtime 传入的 schema 和配置参数：

```python
from __future__ import annotations

import torch
import torch.nn as nn

from taac2026.infrastructure.pcvr.modeling import (
    DenseTokenProjector,
    EmbeddingParameterMixin,
    ModelInput,
    NonSequentialTokenizer,
    SequenceTokenizer,
    masked_mean,
)


class PCVRMyExperiment(EmbeddingParameterMixin, nn.Module):
    def __init__(
        self,
        user_int_feature_specs: list[tuple[int, int, int]],
        item_int_feature_specs: list[tuple[int, int, int]],
        user_dense_dim: int,
        item_dense_dim: int,
        seq_vocab_sizes: dict[str, list[int]],
        user_ns_groups: list[list[int]],
        item_ns_groups: list[list[int]],
        d_model: int = 64,
        emb_dim: int = 64,
        num_blocks: int = 2,
        num_heads: int = 4,
        hidden_mult: int = 4,
        dropout_rate: float = 0.01,
        **kwargs: object,
    ) -> None:
        super().__init__()
        del num_heads, hidden_mult, dropout_rate, kwargs
        self.user_tokenizer = NonSequentialTokenizer(
            user_int_feature_specs,
            user_ns_groups,
            emb_dim,
            d_model,
            user_ns_tokens=0,
            emb_skip_threshold=0,
        )
        self.item_tokenizer = NonSequentialTokenizer(
            item_int_feature_specs,
            item_ns_groups,
            emb_dim,
            d_model,
            item_ns_tokens=0,
            emb_skip_threshold=0,
        )
        self.user_dense = DenseTokenProjector(user_dense_dim, d_model)
        self.item_dense = DenseTokenProjector(item_dense_dim, d_model)
        self.sequence_tokenizers = nn.ModuleDict(
            {domain: SequenceTokenizer(vocab_sizes, emb_dim, d_model) for domain, vocab_sizes in seq_vocab_sizes.items()}
        )
        self.num_ns = self.user_tokenizer.num_tokens + self.item_tokenizer.num_tokens
        self.head = nn.Linear(d_model, 1)

    def _embed(self, inputs: ModelInput) -> torch.Tensor:
        tokens = [self.user_tokenizer(inputs.user_int_feats), self.item_tokenizer(inputs.item_int_feats)]
        user_dense = self.user_dense(inputs.user_dense_feats)
        item_dense = self.item_dense(inputs.item_dense_feats)
        if user_dense is not None:
            tokens.append(user_dense)
        if item_dense is not None:
            tokens.append(item_dense)
        return masked_mean(torch.cat(tokens, dim=1))

    def forward(self, inputs: ModelInput) -> torch.Tensor:
        return self.head(self._embed(inputs)).squeeze(-1)

    def predict(self, inputs: ModelInput) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = self._embed(inputs)
        return self.head(embeddings).squeeze(-1), embeddings


__all__ = ["ModelInput", "PCVRMyExperiment"]
```

实际模型可以更复杂，但仍应满足：

- `forward(inputs)` 返回 logits。
- `predict(inputs)` 返回 `(logits, embeddings)`。
- `num_ns` 是可读属性。
- 优先复用 `taac2026.infrastructure.pcvr.modeling` 的 tokenizer、mask、pooling 和 normalization helper。

## ns_groups.json

每个包都要带自己的 NS 分组文件：

```json
{
  "_purpose": "PCVR non-sequential feature grouping for this experiment.",
  "user_ns_groups": {
    "U1": [1, 15]
  },
  "item_ns_groups": {
    "I1": [11, 13]
  }
}
```

JSON 中的数字是官方列名里的 fid，例如 `user_int_feats_15` 对应 `15`。runtime 会根据当前 schema 映射到实际特征索引。显式传入的文件不存在时会失败，不会静默回退。

## 本地验证

先校验 JSON：

```bash
python -m json.tool config/my_experiment/ns_groups.json >/dev/null
```

再跑实验包契约测试：

```bash
uv run pytest tests/unit/test_experiment_packages.py -q
```

训练 smoke：

```bash
bash run.sh train --experiment config/my_experiment \
    --dataset-path /path/to/parquet_or_dataset_dir \
    --schema-path /path/to/schema.json \
    --num_epochs 1 \
    --batch_size 8 \
    --device cpu
```

打包 smoke：

```bash
uv run taac-package-train --experiment config/my_experiment \
    --output-dir outputs/training_bundles/my_experiment_training_bundle \
    --force

python -m zipfile -l outputs/training_bundles/my_experiment_training_bundle/code_package.zip | head -80
```

最后跑当前单元回归：

```bash
uv run pytest tests/unit -q
```

## 修改现有包的检查清单

- `model_class_name` 与 `model.py` 导出的类一致。
- 非 baseline 包没有暴露 `PCVRHyFormer`。
- `ns_groups.json` 存在，并在 `train_defaults` 中启用。
- 模型能完成 forward、backward 和 predict。
- Bundle 包含目标实验包的 `model.py` 和 `ns_groups.json`。
- 文档中的命令都显式给出数据路径，且只使用当前 CLI 支持的参数。