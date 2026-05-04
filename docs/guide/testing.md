---
icon: lucide/clipboard-list
---

# 测试

## 前置条件

```bash
uv sync --locked --extra dev --extra cuda126
```

## 常用命令

```bash
# 运行所有单元测试
uv run pytest tests/unit -v

# 只运行 unit marker
uv run pytest -m unit -v

# 运行实验包契约测试
uv run pytest tests/unit/infrastructure/experiments/test_experiment_packages.py -v

# 运行运行时契约矩阵
uv run pytest tests/unit/infrastructure/pcvr/test_runtime_contract_matrix.py -v

# 运行训练 / 推理打包测试
uv run pytest tests/unit/application/test_package_training.py -v
uv run pytest tests/unit/application/test_package_inference.py -v

# 运行风格检查
uv run --with ruff ruff check .

# 复现当前 CI 的 CPU 单测命令
uv run --with torch==2.7.1 --with coverage coverage run --data-file=.coverage.cpu -m pytest -m unit -v
```

如果要在本地复现覆盖率门控，CI 还会在测试完成后执行：

```bash
cp .coverage.cpu .coverage
uv run --with coverage coverage report --fail-under=70 \
	--include='src/taac2026/domain/*,src/taac2026/application/training/__init__.py,src/taac2026/application/training/cli.py,src/taac2026/application/training/runtime_optimization.py'
```

## 当前测试树

`tests/unit/` 现在已经按层次拆分，不再使用旧的平铺布局。

| 目录 / 文件                                                               | 覆盖范围                                                      |
| ------------------------------------------------------------------------- | ------------------------------------------------------------- |
| `tests/unit/application/test_training_cli.py`                             | 训练 CLI 参数和请求装配                                       |
| `tests/unit/application/test_evaluation_cli.py`                           | 评估 CLI 参数和请求装配                                       |
| `tests/unit/application/test_evaluation_infer_entrypoint.py`              | 推理入口环境变量到 CLI 的桥接                                 |
| `tests/unit/application/test_package_training.py`                         | 训练 Bundle 打包结构、manifest 和运行时约定                   |
| `tests/unit/application/test_package_inference.py`                        | 推理 Bundle 打包结构、manifest 和运行时约定                   |
| `tests/unit/application/test_run_sh_commands.py`                          | `run.sh` 只支持 `train` / `val` / `eval` / `infer` 的命令契约 |
| `tests/unit/application/test_model_performance_reporting.py`              | 报表与可视化汇总逻辑                                          |
| `tests/unit/application/test_pcvr_optimizer_benchmark.py`                 | 优化器 benchmark 报表逻辑                                     |
| `tests/unit/application/test_pcvr_tilelang_ops_benchmark.py`              | TileLang benchmark 报表逻辑                                   |
| `tests/unit/domain/test_metrics.py`                                       | AUC、LogLoss、GAUC、诊断指标                                  |
| `tests/unit/infrastructure/experiments/test_experiment_discovery.py`      | 实验包扫描与发现                                              |
| `tests/unit/infrastructure/experiments/test_experiment_packages.py`       | PCVR 实验包契约、模型类名、前向 / 反向 / predict              |
| `tests/unit/infrastructure/experiments/test_checkpoint_and_loader.py`     | Checkpoint 与实验包装载                                       |
| `tests/unit/infrastructure/experiments/test_maintenance_experiments.py`   | 维护类实验包元数据契约                                        |
| `tests/unit/infrastructure/maintenance/test_online_dataset_eda_runner.py` | Online Dataset EDA 运行器                                     |
| `tests/unit/infrastructure/pcvr/test_pcvr_protocol.py`                    | Schema 到模型参数的转换                                       |
| `tests/unit/infrastructure/pcvr/test_runtime_contract_matrix.py`          | 运行时契约矩阵、schema/配置/sidecar 一致性                    |
| `tests/unit/infrastructure/pcvr/test_pcvr_trainer.py`                     | Trainer 循环与优化器协作                                      |
| `tests/unit/infrastructure/pcvr/test_pcvr_data_split.py`                  | Row Group 切分与 observed schema                              |
| `tests/unit/infrastructure/pcvr/test_pcvr_data_augmentation.py`           | 数据增强流水线                                                |
| `tests/unit/infrastructure/pcvr/test_pcvr_infer_runtime.py`               | 评估 / 推理运行时与 sidecar 输出                              |
| `tests/unit/infrastructure/pcvr/test_tilelang_model_smoke.py`             | TileLang 模型级 smoke                                         |
| `tests/unit/infrastructure/test_tilelang_ops.py`                          | TileLang / Torch RMSNorm 算子层契约                           |

## 模块改动后的最小复核

先跑最小相关测试，再决定是否扩大到整个 `tests/unit/`。

| 改动位置                                                    | 最小复核                                                                                                                                                                                                    |
| ----------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `experiments/pcvr/*/model.py`                               | `tests/unit/infrastructure/experiments/test_experiment_packages.py`                                                                                                                                         |
| `experiments/pcvr/*/__init__.py`                            | `tests/unit/infrastructure/experiments/test_experiment_packages.py`、`tests/unit/infrastructure/experiments/test_experiment_discovery.py`、`tests/unit/infrastructure/pcvr/test_runtime_contract_matrix.py` |
| `experiments/maintenance/*`                                 | `tests/unit/infrastructure/experiments/test_maintenance_experiments.py`；若改的是 EDA 运行器，再加 `tests/unit/infrastructure/maintenance/test_online_dataset_eda_runner.py`                                |
| `src/taac2026/application/training/cli.py`                  | `tests/unit/application/test_training_cli.py`                                                                                                                                                               |
| `src/taac2026/application/evaluation/cli.py`                | `tests/unit/application/test_evaluation_cli.py`                                                                                                                                                             |
| `src/taac2026/application/evaluation/infer.py`              | `tests/unit/application/test_evaluation_infer_entrypoint.py`                                                                                                                                                |
| `src/taac2026/application/maintenance/package_training.py`  | `tests/unit/application/test_package_training.py`                                                                                                                                                           |
| `src/taac2026/application/maintenance/package_inference.py` | `tests/unit/application/test_package_inference.py`                                                                                                                                                          |
| `src/taac2026/infrastructure/pcvr/protocol.py`              | `tests/unit/infrastructure/pcvr/test_pcvr_protocol.py`、`tests/unit/infrastructure/pcvr/test_runtime_contract_matrix.py`                                                                                    |
| `src/taac2026/infrastructure/pcvr/trainer.py`               | `tests/unit/infrastructure/pcvr/test_pcvr_trainer.py`                                                                                                                                                       |
| `src/taac2026/infrastructure/pcvr/data.py`                  | `tests/unit/infrastructure/pcvr/test_pcvr_data_split.py`                                                                                                                                                    |
| `src/taac2026/infrastructure/pcvr/data_pipeline.py`         | `tests/unit/infrastructure/pcvr/test_pcvr_data_augmentation.py`                                                                                                                                             |
| `src/taac2026/infrastructure/pcvr/tilelang_ops.py`          | `tests/unit/infrastructure/test_tilelang_ops.py`；必要时加 `tests/unit/infrastructure/pcvr/test_tilelang_model_smoke.py`                                                                                    |
| `src/taac2026/domain/metrics.py`                            | `tests/unit/domain/test_metrics.py`                                                                                                                                                                         |
| `run.sh`                                                    | `tests/unit/application/test_run_sh_commands.py`                                                                                                                                                            |

## 新增测试约定

- 测试文件命名：`test_<module>.py`
- 测试函数命名：`test_<behavior>`
- 使用 `conftest.py` 中的 marker 自动标记为 `unit` 或 `integration`
- 测试数据使用 `data/sample_1000_raw/` 中的示例数据
- 不依赖外部服务或 GPU

## CI 当前检查什么

当前 CI 以 `.github/workflows/ci.yml` 为准，核心检查有三类：

- 风格检查：`uv run --with ruff ruff check .`
- CPU 单元测试：`uv sync --locked --extra dev` 后运行 `uv run --with torch==2.7.1 --with coverage coverage run --data-file=.coverage.cpu -m pytest -m unit -v`
- 覆盖率门控：恢复 `.coverage.cpu` 后执行 `coverage report --fail-under=70`

如果你只是想在本地先做快速自检，优先跑与改动面对应的最小测试集；合并前再跑整套 `tests/unit/`。

## Smoke 不等于单测

训练 Smoke Test（`taac-train` 跑 1 epoch）是集成测试，不替代单元测试：

- 单测验证组件行为（输入输出契约）
- Smoke Test 验证组件协作（端到端流程）
- 两者互补，不可替代
