---
icon: lucide/clipboard-list
---

# 测试

## 常用命令

```bash
# 运行所有单元测试
uv run pytest tests/unit -v

# 运行特定测试文件
uv run pytest tests/unit/test_experiment_packages.py -v

# 运行特定测试
uv run pytest tests/unit/test_experiment_packages.py::test_model_class_name -v

# 带覆盖率
uv run pytest tests/unit --cov=taac2026.domain --cov=taac2026.application --cov-report=term-missing

# 仅运行特定 marker
uv run pytest tests/unit -m unit -v
```

## 当前测试文件

| 文件                                  | 覆盖范围                                                        |
| ------------------------------------- | --------------------------------------------------------------- |
| `test_experiment_packages.py`         | 实验包加载、模型类名、NS Groups、前向/反向传播                  |
| `test_runtime_contract_matrix.py`     | 协议测试：schema 转换、NS 分组映射、batch 转换、Checkpoint 操作 |
| `test_checkpoint_and_loader.py`       | Baseline 包加载和 Checkpoint 侧车文件                           |
| `test_pcvr_protocol.py`               | Schema 到特征规格的转换、NS 分组映射                            |
| `test_pcvr_trainer.py`                | Trainer 测试                                                    |
| `test_pcvr_data_split.py`             | Row Group 分割测试                                              |
| `test_pcvr_data_augmentation.py`      | 数据增强测试                                                    |
| `test_pcvr_infer_runtime.py`          | 推理运行时测试                                                  |
| `test_metrics.py`                     | 指标计算测试                                                    |
| `test_training_cli.py`                | 训练 CLI 测试                                                   |
| `test_evaluation_cli.py`              | 评估 CLI 测试                                                   |
| `test_evaluation_infer_entrypoint.py` | 推理入口测试                                                    |
| `test_experiment_discovery.py`        | 实验包发现测试                                                  |
| `test_package_training.py`            | 训练 Bundle 打包测试                                            |
| `test_package_inference.py`           | 推理 Bundle 打包测试                                            |
| `test_run_sh_commands.py`             | run.sh 命令测试                                                 |

## 模块改动后的最小复核

修改模块后的最小测试集：

| 改动模块                               | 必须运行的测试                                                |
| -------------------------------------- | ------------------------------------------------------------- |
| `config/*/model.py`                    | `test_experiment_packages.py`                                 |
| `config/*/ns_groups.json`              | `test_experiment_packages.py`, `test_pcvr_protocol.py`        |
| `config/*/__init__.py`                 | `test_experiment_packages.py`, `test_experiment_discovery.py` |
| `infrastructure/pcvr/protocol.py`      | `test_pcvr_protocol.py`, `test_runtime_contract_matrix.py`    |
| `infrastructure/pcvr/trainer.py`       | `test_pcvr_trainer.py`                                        |
| `infrastructure/pcvr/data.py`          | `test_pcvr_data_split.py`                                     |
| `infrastructure/pcvr/data_pipeline.py` | `test_pcvr_data_augmentation.py`                              |
| `domain/metrics.py`                    | `test_metrics.py`                                             |
| `application/training/cli.py`          | `test_training_cli.py`                                        |
| `application/evaluation/cli.py`        | `test_evaluation_cli.py`                                      |
| `run.sh`                               | `test_run_sh_commands.py`                                     |

## 新增测试约定

- 测试文件命名：`test_<module>.py`
- 测试函数命名：`test_<behavior>`
- 使用 `conftest.py` 中的 marker 自动标记为 `unit` 或 `integration`
- 测试数据使用 `data/sample_1000_raw/` 中的示例数据
- 不依赖外部服务或 GPU

## 训练 Smoke 不等于单测

训练 Smoke Test（`taac-train` 跑 1 epoch）是集成测试，不替代单元测试：

- 单测验证组件行为（输入输出契约）
- Smoke Test 验证组件协作（端到端流程）
- 两者互补，不可替代
