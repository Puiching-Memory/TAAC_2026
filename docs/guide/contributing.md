---
icon: lucide/git-branch-plus
---

# 新增实验包

新增实验包时，先让它像现有实验一样被发现、能训练、能评估，再考虑模型创新。共享训练、评估、推理和打包能力都在 `src/taac2026/`，实验包应该保持薄。

## 最小目录

```text
experiments/my_experiment/
├── __init__.py
└── model.py
```

需要私有层时可以加 `layers.py`，但不要把共享 runtime 复制进实验包。

## 从哪里复制

| 你要做什么                 | 建议起点                                                   |
| -------------------------- | ---------------------------------------------------------- |
| HyFormer 小改              | `experiments/baseline`                                     |
| HyFormer + 增强 / TileLang | `experiments/baseline_plus`                                |
| 用户-物品交互结构          | `experiments/interformer`                                  |
| 统一 Transformer 结构      | `experiments/onetrans`                                     |
| 多组件消融                 | `experiments/symbiosis`，但只在确实需要自定义 hooks 时使用 |

## `__init__.py` 负责什么

普通 PCVR 实验通过 `create_pcvr_experiment()` 声明自己：

```python
from pathlib import Path

from taac2026.api import PCVRModelConfig, PCVRNSConfig, PCVRTrainConfig, create_pcvr_experiment


TRAIN_DEFAULTS = PCVRTrainConfig(
    model=PCVRModelConfig(num_blocks=2, num_heads=4, dropout_rate=0.02),
    ns=PCVRNSConfig(
        grouping_strategy="explicit",
        user_groups={"U1": [1, 15]},
        item_groups={"I1": [11, 13]},
        tokenizer_type="rankmixer",
        user_tokens=5,
        item_tokens=2,
    ),
)

EXPERIMENT = create_pcvr_experiment(
    name="pcvr_my_experiment",
    package_dir=Path(__file__).resolve().parent,
    model_class_name="MyModel",
    train_defaults=TRAIN_DEFAULTS,
)
```

关键点：

- `name` 要唯一，通常用 `pcvr_` 前缀。
- `model_class_name` 必须和 `model.py` 里的类名一致。
- NS 分组现在写在 `PCVRNSConfig` 里，不需要独立 `ns_groups.json`。
- 普通模型实验不需要手写 hooks。

只有确实改变训练、预测或 runtime 行为时，才传 hook override。可以参考 `experiments/symbiosis/__init__.py`。

## 实验发现机制

加载逻辑在 `src/taac2026/application/experiments/registry.py`：

- `--experiment experiments/my_experiment` 会按文件系统路径加载。
- `--experiment experiments.my_experiment` 会按 Python module 加载。
- 被加载模块必须导出 `EXPERIMENT`。
- `EXPERIMENT` 可以是 `ExperimentSpec`，也可以是带 `name`、`train`、`evaluate`、`infer` 方法的对象。

`create_pcvr_experiment()` 位于 `src/taac2026/application/experiments/factory.py`。它会把默认 PCVR hooks 组装成 `PCVRExperiment`，普通实验通常不需要自己实现 `train()`、`evaluate()` 或 `infer()`。

`PCVRExperiment` 的运行逻辑在 `src/taac2026/application/experiments/experiment.py`：

- `train()` 解析本地 demo 数据或线上数据路径，调用训练 workflow。
- `evaluate()` 根据 checkpoint sidecar 重建模型和 schema，写 `evaluation.json`。
- `infer()` 读取 checkpoint 和 schema，写 `predictions.json`。
- 进入实验包模型代码前，会临时把实验目录放到 `sys.path`，并隔离 `model` / `utils` 这类插件模块名。

## `model.py` 负责什么

模型类需要能被共享 PCVR runtime 构造，并满足训练 / 评估 / 推理契约：

- `forward(inputs)` 返回 `(B,)` logits。
- `predict(inputs)` 返回 `(logits, embeddings)`。
- 稀疏和稠密参数能被正确分组。
- checkpoint 侧车里的 `train_config.json` 和 `schema.json` 能在评估 / 推理阶段复用。

优先复用共享建模组件：

```python
from taac2026.infrastructure.modeling import (
    EmbeddingParameterMixin,
    FeatureEmbeddingBank,
    ModelInput,
    NonSequentialTokenizer,
    SequenceTokenizer,
)
```

论文特有组件可以留在实验包内；通用能力应该沉到 `src/taac2026/infrastructure/modeling/`。

## Checkpoint Sidecar

训练成功后，runtime 期望 checkpoint 目录至少包含：

```text
global_step*/
├── model.safetensors
├── schema.json
└── train_config.json
```

`model.safetensors` 只保存权重；`schema.json` 和 `train_config.json` 才能让评估 / 推理重新构造输入契约、NS 分组、runtime execution 和模型默认参数。只复制权重文件会导致评估或推理失败。

改这些内容时要特别小心：

- `PCVRModelConfig` 字段名
- `PCVRNSConfig` 的 grouping strategy 和 fid 分组
- `ModelInput` 字段
- 自定义 hook 写入 `train_config.json` 的额外 key
- checkpoint 目录命名和最新 checkpoint 解析规则

## 本地验证顺序

先看能否发现并加载：

```bash
uv run python -c "from taac2026.application.experiments.registry import load_experiment_package; print(load_experiment_package('experiments/my_experiment').name)"
```

再跑最小 smoke：

```bash
bash run.sh train \
  --experiment experiments/my_experiment \
  --run-dir outputs/my_experiment_smoke \
  --device cpu \
  --num_workers 0 \
  --batch_size 8 \
  --max_steps 1
```

最后跑契约测试：

```bash
uv run pytest tests/contract/experiments/test_packages.py -q
uv run pytest tests/contract/experiments/test_runtime_contract_matrix.py -q
uv run pytest tests/unit/application/experiments -q
```

涉及 bundle 时再加：

```bash
uv run pytest tests/integration/application/packaging -q
uv run pytest tests/unit/application/bootstrap tests/integration/application/bootstrap -q
```

如果要确认打包内容，直接看 zip：

```bash
uv run taac-package-train \
  --experiment experiments/my_experiment \
  --output-dir outputs/bundles/my_experiment_training \
  --json

python -m zipfile -l outputs/bundles/my_experiment_training/code_package.zip | sed -n '1,120p'
```

zip 里应该包含 `project/src/taac2026/**`、当前实验包、必要的父级 `__init__.py`、`pyproject.toml` 和 manifest；不应该把整个 `outputs/`、`tests/` 或其他无关实验包打进去。

## 提交前检查

- 实验名、目录名和模型类名能对应上。
- `PCVRNSConfig` 里的 fid 来自当前 schema。
- 本地 PCVR smoke 默认不需要 `--dataset-path`；调试自定义 parquet 时可以显式传本地路径和 schema。
- `forward()` 和 `predict()` 输出形状符合契约。
- 训练后能生成 `global_step*/model.safetensors`、`schema.json` 和 `train_config.json`。
- 如果改了共享 runtime，至少跑过相关 unit test，而不是只跑自己的实验。
