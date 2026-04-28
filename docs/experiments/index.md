---
icon: lucide/folder-open
---

# 实验包总览

实验包是 TAAC 2026 的核心扩展单元。每个包是 `config/` 下的独立目录，包含模型定义、NS 分组和默认训练配置。新增实验无需修改框架代码。

## 当前实验包

| 实验包         | 模型类         | Blocks | Dropout | NS Tokenizer      | 亮点                             |
| -------------- | -------------- | ------ | ------- | ----------------- | -------------------------------- |
| Baseline       | HyFormer       | 默认   | 默认    | group             | 基准参考                         |
| CTR Baseline   | CTRBaseline    | 2      | 0.01    | group             | 低 Dropout                       |
| DeepContextNet | DeepContextNet | 3      | 0.02    | group             | 最深的 group 栈                  |
| HyFormer       | HyFormer       | 默认   | 默认    | rankmixer (5u/2i) | num_queries=2                    |
| InterFormer    | InterFormer    | 2      | 0.02    | group             | 交叉注意力                       |
| OneTrans       | OneTrans       | 2      | 0.02    | rankmixer (5u/2i) | 单 Transformer                   |
| Symbiosis      | Symbiosis      | 3      | 0.02    | rankmixer (5u/2i) | RoPE + AMP + compile + 11 项开关 |
| UniRec         | UniRec         | 3      | 0.02    | rankmixer (5u/2i) | 统一推荐                         |
| UniScaleFormer | UniScaleFormer | 4      | 0.02    | rankmixer (5u/2i) | 最深栈                           |

## 包内文件

每个实验包目录必须包含三个文件：

```
config/<experiment_name>/
├── __init__.py         # EXPERIMENT = PCVRExperiment(name, package_dir, model_class_name, train_defaults)
├── model.py            # 模型类实现
└── ns_groups.json      # NS 特征分组映射
```

**`__init__.py`** 定义模块级 `EXPERIMENT` 对象：

```python
from taac2026.infrastructure.pcvr.experiment import PCVRExperiment
from taac2026.infrastructure.pcvr.config import PCVRTrainConfig

EXPERIMENT = PCVRExperiment(
    name="pcvr_baseline",
    package_dir=Path(__file__).parent,
    model_class_name="PCVRHyFormer",
    train_defaults=PCVRTrainConfig(...),
)
```

**`model.py`** 实现模型类，必须继承 `EmbeddingParameterMixin` 并实现：

| 方法                             | 签名                                   | 说明                   |
| -------------------------------- | -------------------------------------- | ---------------------- |
| `forward`                        | `(ModelInput) -> logits`               | 前向传播               |
| `predict`                        | `(ModelInput) -> (logits, embeddings)` | 推理用                 |
| `get_sparse_params`              | `() -> list[Parameter]`                | 稀疏参数               |
| `get_dense_params`               | `() -> list[Parameter]`                | 稠密参数               |
| `reinit_high_cardinality_params` | `() -> None`                           | 重初始化低频 Embedding |

**`ns_groups.json`** 定义 NS 特征分组，格式见 [架构与概念](../architecture.md#ns-groups)。

## 运行任意实验包

```bash
uv run taac-train --experiment config/<name> \
  --dataset-path data/sample_1000_raw/demo_1000.parquet \
  --schema-path data/sample_1000_raw/schema.json
```

## 打包任意实验包

```bash
uv run taac-package-train --experiment config/<name> --output-dir outputs/bundle
uv run taac-package-infer --experiment config/<name> --output-dir outputs/bundle
```

## 新增或修改实验包

详见 [新增实验包](../guide/contributing.md)。
