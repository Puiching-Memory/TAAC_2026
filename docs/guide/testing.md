---
icon: lucide/clipboard-list
---

# 测试

当前仓库的可执行回归集中在 `tests/unit/`。历史上的 integration、GPU 和 benchmark 目录已经不再作为当前测试树的一部分；文档和 CI 应以当前存在的 unit suite 为准。

## 常用命令

```bash
uv sync --locked --extra cuda126

bash run.sh test tests/unit -q
bash run.sh test tests/unit/test_experiment_packages.py -q
bash run.sh test tests/unit/test_package_training.py -q
```

也可以直接用 pytest：

```bash
uv run --extra cuda126 pytest tests/unit -q
```

Lint 当前核心代码时：

```bash
uv run --with ruff ruff check src/taac2026 tests/unit
```

## 当前测试文件

| 文件 | 覆盖重点 |
| --- | --- |
| `tests/unit/test_metrics.py` | AUC、logloss 和分类指标边界 |
| `tests/unit/test_runtime_contract_matrix.py` | runtime 契约矩阵与命令行为 |
| `tests/unit/test_checkpoint_and_loader.py` | checkpoint、sidecar 与包加载 |
| `tests/unit/test_experiment_packages.py` | 实验包加载、模型类、forward/backward/predict、NS groups |
| `tests/unit/test_package_training.py` | `run.sh` + `code_package.zip` 双文件 bundle 内容 |
| `tests/unit/test_training_cli.py` | 训练 CLI 参数解析和 extra args 透传 |
| `tests/unit/test_pcvr_protocol.py` | schema、特征规格、NS groups 映射与缺失文件失败 |

## 模块改动后的最小复核

| 改动范围 | 建议命令 |
| --- | --- |
| 实验包 `config/<name>/` | `bash run.sh test tests/unit/test_experiment_packages.py -q` |
| `ns_groups.json` 或 PCVR schema 解析 | `bash run.sh test tests/unit/test_pcvr_protocol.py tests/unit/test_experiment_packages.py -q` |
| 线上打包 | `bash run.sh test tests/unit/test_package_training.py -q` |
| 训练入口参数 | `bash run.sh test tests/unit/test_training_cli.py -q` |
| checkpoint 或 loader | `bash run.sh test tests/unit/test_checkpoint_and_loader.py -q` |
| 指标实现 | `bash run.sh test tests/unit/test_metrics.py -q` |

## 新增测试约定

- 当前新增测试默认放在 `tests/unit/`。
- 测试应尽量使用轻量 synthetic 输入，不依赖完整线上数据。
- 修改实验包时，优先补 `test_experiment_packages.py` 覆盖加载、forward、backward、predict 和命名契约。
- 修改打包逻辑时，检查 `code_package.zip` 中是否包含目标包的 `model.py` 与 `ns_groups.json`，并排除 docs/tests/其他实验包。
- 修改 CLI 时，同时覆盖 argparse 结果和未知参数透传。

## 训练 Smoke 不等于单测

训练 smoke 用来确认真实数据路径、schema、dataloader 和设备环境能跑通：

```bash
bash run.sh train --experiment config/baseline \
    --dataset-path /path/to/parquet_or_dataset_dir \
    --schema-path /path/to/schema.json \
    --num_epochs 1 \
    --batch_size 8 \
    --device cpu
```

它适合在提交前做人工确认，但不能替代 `tests/unit/` 的契约测试。