---
icon: lucide/folder-open
---

# 实验包总览

实验包是当前仓库的核心扩展单元。框架层负责训练、评估、推理、打包和通用运行时，实验包只保留实验特有的模型、默认配置和少量包内辅助模块。

当前有两类实验包：

- `experiments/`：PCVR 模型实验
- `experiments/`：维护 / 分析类实验

## 当前实验包

### 模型训练包

| 实验包         | 路径                              | 说明                                 |
| -------------- | --------------------------------- | ------------------------------------ |
| Baseline       | `experiments/baseline`       | 共享 PCVR 运行时上的基准 HyFormer 包 |
| InterFormer    | `experiments/interformer`    | InterFormer 结构实验                 |
| OneTrans       | `experiments/onetrans`       | OneTrans 结构实验                    |
| Symbiosis      | `experiments/symbiosis`      | 带包内辅助层和额外训练参数的融合实验 |

### 维护工具包

| 实验包             | 用途             | 需要数据集 | 说明                             |
| ------------------ | ---------------- | ---------- | -------------------------------- |
| Host Device Info   | 主机设备诊断采集 | 否         | 采集 GPU、网络、依赖源等环境快照 |
| Online Dataset EDA | 数据集探索分析   | 是         | 流式 Parquet 统计，线上友好      |

## 包内文件

### 模型训练包

PCVR 实验包的最小契约是两个文件：

```text
experiments/<experiment_name>/
├── __init__.py
└── model.py
```

在此基础上，实验包可以按需要增加自己的辅助模块，例如：

```text
experiments/symbiosis/
├── __init__.py
├── layers.py
└── model.py
```

#### `__init__.py`

`__init__.py` 负责导出模块级 `EXPERIMENT`。当前真实契约是：

- 使用 `PCVRExperiment`
- 声明 `name`
- 指定 `package_dir`
- 指定 `model_class_name`
- 提供 `train_defaults`
- 提供 `train_arg_parser`
- 提供 `train_hooks`
- 提供 `prediction_hooks`
- 提供 `runtime_hooks`

大多数新实验都会从一个现有包复制这些默认 hook，再替换实验名、模型类名和默认配置。

#### `model.py`

`model.py` 负责导出由 `model_class_name` 指定的模型类。模型类通常：

| 方法                             | 签名                                   | 说明                   |
| -------------------------------- | -------------------------------------- | ---------------------- |
| `forward`                        | `(ModelInput) -> logits`               | 前向传播               |
| `predict`                        | `(ModelInput) -> (logits, embeddings)` | 推理用                 |
| `get_sparse_params`              | `() -> list[Parameter]`                | 稀疏参数分组           |
| `get_dense_params`               | `() -> list[Parameter]`                | 稠密参数分组           |
| `reinit_high_cardinality_params` | `() -> None`                           | 重初始化低频 Embedding |

当前仓库中的 NS 分组直接写在 `PCVRNSConfig` 里；不要再把包内独立 `ns_groups.json` 当成必备文件。

### 维护工具包

维护工具包使用 `ExperimentSpec` 而非 `PCVRExperiment`，不需要模型定义和 NS 分组：

```text
experiments/<tool_name>/
├── __init__.py
└── runner.py
```

- [Host Device Info](host-device-info.md) -- 主机与设备诊断采集
- [Online Dataset EDA](online-dataset-eda.md) -- 数据集探索性分析

## 运行任意实验包

```bash
bash run.sh train --experiment experiments/<name> \
  --run-dir outputs/<name>
```

维护类实验也走同一入口，例如：

```bash
bash run.sh train --experiment experiments/host_device_info
```

## 打包任意实验包

```bash
uv run taac-package-train --experiment experiments/<name> --output-dir outputs/bundles/<name>_training
uv run taac-package-infer --experiment experiments/<name> --output-dir outputs/bundles/<name>_inference
```

维护类实验只支持训练 Bundle，不支持推理 Bundle。

## 新增或修改实验包

详见 [新增实验包](../guide/contributing.md)。
