---
icon: lucide/folder-open
---

# 实验包总览

实验包是 `config/<name>/` 下的 Python 目录包。当前主契约是 `PCVRExperiment`：包内声明实验元数据和默认参数，`model.py` 提供模型类，`ns_groups.json` 提供非序列特征分组。

## 当前实验包

| 实验包 | 目录 | 模型类 | 简述 |
| --- | --- | --- | --- |
| [Baseline](baseline.md) | `config/baseline` | `PCVRHyFormer` | 官方 HyFormer 风格 baseline |
| [Symbiosis](symbiosis.md) | `config/symbiosis` | `PCVRSymbiosis` | 统一 token 与上下文交换融合模型 |
| [CTR Baseline](ctr-baseline.md) | `config/ctr_baseline` | `PCVRCTRBaseline` | CTR/DIN 风格轻量对照 |
| [DeepContextNet](deepcontextnet.md) | `config/deepcontextnet` | `PCVRDeepContextNet` | 上下文增强深度模型 |
| [HyFormer](hyformer.md) | `config/hyformer` | `PCVRHyFormer` | HyFormer 方向实验包 |
| [InterFormer](interformer.md) | `config/interformer` | `PCVRInterFormer` | 序列与非序列交互建模 |
| [OneTrans](onetrans.md) | `config/onetrans` | `PCVROneTrans` | 统一 token 化与单 Transformer |
| [UniRec](unirec.md) | `config/unirec` | `PCVRUniRec` | 多阶段融合实验 |
| [UniScaleFormer](uniscaleformer.md) | `config/uniscaleformer` | `PCVRUniScaleFormer` | 缩放序列与融合实验 |

## 包内文件

```text
config/<name>/
├── __init__.py
├── model.py
└── ns_groups.json
```

- `__init__.py`：导出 `EXPERIMENT = PCVRExperiment(...)`。
- `model.py`：导出 `ModelInput` 和 `model_class_name` 指定的模型类。
- `ns_groups.json`：训练、评估、推理共用的非序列特征分组。

## 运行任意实验包

```bash
bash run.sh train --experiment config/<name> \
    --dataset-path /path/to/parquet_or_dataset_dir \
    --schema-path /path/to/schema.json \
    --num_epochs 1 \
    --batch_size 8 \
    --device cpu

bash run.sh val --experiment config/<name> \
    --dataset-path /path/to/parquet_or_dataset_dir \
    --schema-path /path/to/schema.json \
    --run-dir outputs/config/<name> \
    --device cpu
```

## 打包任意实验包

```bash
bash run.sh package --experiment config/<name> \
    --output-dir outputs/training_bundles/<name>_training_bundle \
    --force
```

生成目录中只有 `run.sh` 和 `code_package.zip` 两个顶层文件。代码包会包含选中实验包的 `model.py` 和 `ns_groups.json`，不会包含其他实验包。

## 新增或修改实验包

新增包时按 [新增实验包](../guide/contributing.md) 的 `PCVRExperiment` 契约实现，然后至少运行：

```bash
python -m json.tool config/<name>/ns_groups.json >/dev/null
bash run.sh test tests/unit/test_experiment_packages.py -q
bash run.sh test tests/unit/test_package_training.py -q
```