# Testing

## 概览

测试套件位于 `tests/`，使用 [pytest](https://docs.pytest.org/) 运行。所有测试按文件名自动标记为 **unit**、**integration** 或 **gpu**——无需手写 `@pytest.mark`，由 `tests/conftest.py` 中的文件名集合统一管理。

快速 CI（`.github/workflows/ci.yml`）负责 CPU unit 与 CPU-safe coverage 门槛；GPU 测试与性能测试则分别拆到手动触发的 `.github/workflows/gpu-tests.yml` 和 `.github/workflows/performance-benchmarks.yml`。这样做是因为仓库当前固定的是 CUDA 版 TorchRec 和 fbgemm-gpu，部分看似 CPU 断言的测试在导入阶段也会要求 `libcuda.so.1`，而文档部署不应被等待自托管 runner 的队列阻塞。

## 快速开始

```bash
# 同步环境（锁定版本）
uv sync --locked

# 全量回归
uv run pytest -q

# 仅单元测试
uv run pytest -m unit -q

# 仅集成测试（需要 CUDA/TorchRec 运行栈，CI 在 GPU runner 上执行）
uv run pytest -m integration -q

# GPU 测试（需要 CUDA 硬件）
uv run python scripts/run_gpu_tests.py

# 共享 Transformer backend benchmark（torch / triton / te）
uv run python scripts/benchmark_transformer_backends.py --profiles hyformer-default deepcontextnet-default
```

## 测试分层

| 标记          | 说明                           | 文件                                                                                                                                                                                                                                                                                                                       |
| ------------- | ------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `unit`        | 纯逻辑、CPU 可直接运行、快速   | `test_benchmark_charts` · `test_clean_pycache` · `test_dataset_eda` · `test_github_cleanup` · `test_metrics` · `test_model_performance_plot` · `test_norms` · `test_property_based` · `test_payload` · `test_pooling_heads` · `test_runtime_optimization` · `test_schema_contract` · `test_tech_timeline` · `test_test_collection` · `test_transformer_blocks` |
| `integration` | 跨模块闭环，依赖 TorchRec/CUDA 运行栈 | `test_data_pipeline` · `test_embedding_collection` · `test_evaluate_cli` · `test_experiment_packages` · `test_model_robustness` · `test_optimizers` · `test_profiling` · `test_profiling_unit` · `test_quantization` · `test_runtime_integration` · `test_search` · `test_search_trial` · `test_search_worker` · `test_search_worker_integration` · `test_torchrec_embedding` · `test_training_recovery` |
| `gpu`         | 需要真实 CUDA 设备或 Triton 内核 | `test_gpu` · `test_triton_kernels` · `tests/benchmarks/bench_*.py` |

新增测试文件 **必须** 同步添加到 `tests/conftest.py` 对应的文件名集合中，否则 pytest 收集阶段会直接报错。

## Coverage

```bash
uv run --with coverage coverage erase
uv run --with coverage coverage run -m pytest -m unit -q
uv run --with coverage coverage run --append -m pytest -m integration -q
uv run --with coverage coverage report
```

| 项目     | 值                                                                                            |
| -------- | --------------------------------------------------------------------------------------------- |
| 统计范围 | `src/taac2026/domain`、`src/taac2026/application/search`、`src/taac2026/application/training` |
| 分支覆盖 | 开启                                                                                          |
| 最低门槛 | **70 %**                                                                                      |

配置位于 `pyproject.toml` 的 `[tool.coverage.*]` 段。

快速 CI 不会直接拿 unit-only 数据去校验这组全量范围，而是单独对 CPU-safe 子集执行门槛：`src/taac2026/domain/*`、`src/taac2026/application/training/__init__.py`、`src/taac2026/application/training/cli.py`、`src/taac2026/application/training/runtime_optimization.py`。如果你要看完整仓库覆盖率，仍然要合并上面的 unit 与 integration/gpu 数据再执行 `uv run --with coverage coverage report`。

## 模块改动速查

改完代码后，按下表选择最小验证集，快速确认不回退：

| 改动范围                             | 建议运行                                                                                                |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------- |
| `domain/metrics.py`                  | `test_metrics.py` `test_property_based.py`                                                              |
| `infrastructure/experiments/payload` | `test_payload.py`                                                                                       |
| `application/training/`              | `test_runtime_optimization.py` `test_profiling_unit.py` `test_profiling.py` `test_training_recovery.py` |
| `application/search/`                | `test_search_trial.py` `test_search_worker.py` `test_search_worker_integration.py` `test_search.py`     |
| 数据读取 / batch 组装                | `test_data_pipeline.py` `test_runtime_integration.py`                                                   |
| 实验包 (`config/`)                   | `test_experiment_packages.py` `test_model_robustness.py`                                                |
| 共享 Transformer / TE backend        | `test_transformer_blocks.py` `uv run python scripts/benchmark_transformer_backends.py --profiles hyformer-default deepcontextnet-default` |

示例：

```bash
uv run pytest tests/test_metrics.py tests/test_property_based.py -q
```

## 编写新测试

1. 在 `tests/` 新建 `test_*.py`，同步把文件名加入 `conftest.py` 的 `UNIT_TEST_FILES`、`INTEGRATION_TEST_FILES` 或 `GPU_TEST_FILES`。
2. 统一使用 `uv run pytest ...` 运行，不要直接调 `python -m pytest`。
3. Property-based 测试用 Hypothesis，控制 `max_examples` 保持速度。
4. 涉及训练产出物时，验证 `best.pt`、`summary.json`、`training_curves.json`、`profiling/` 的兼容性。
5. 涉及搜索流程时，覆盖 success、fail、pruned 三种 trial 状态。

## CI 流程

快速 CI 在 `ubuntu-latest` + Python 3.13 上运行 CPU 纯逻辑测试，并只对 CPU-safe 子集执行 coverage 门槛。GPU 测试与 benchmark 改为独立的手动 workflow，在需要时再投递到自托管 GPU runner。文档部署只等待快速 CI 完成。

快速 CI：

1. `uv sync --locked` — 严格锁定环境
2. `uv run python scripts/lint_torch.py` — torch 导入规范检查
3. `coverage run --data-file=.coverage.cpu -m pytest -m unit` — CPU 单元测试 + 覆盖率采集
4. `coverage report --fail-under=70 --include="src/taac2026/domain/*,src/taac2026/application/training/__init__.py,src/taac2026/application/training/cli.py,src/taac2026/application/training/runtime_optimization.py"` — CPU-safe 子集门槛校验（< 70 % 失败）
5. `coverage xml --fail-under=0 --include="src/taac2026/domain/*,src/taac2026/application/training/__init__.py,src/taac2026/application/training/cli.py,src/taac2026/application/training/runtime_optimization.py" -o coverage.xml` — 导出 CPU-only coverage artifact

手动 GPU 测试：

1. `uv sync --locked` — 在自托管 GPU runner 上同步环境
2. `uv run python scripts/verify_gpu_env.py --json` — 记录 GPU compute capability、精度路由、TorchRec / fbgemm / Triton 以及可选 Transformer Engine 工具链证据
3. `coverage run --data-file=.coverage.gpu -m pytest -m "integration or gpu"` — 执行 integration + gpu 标记测试
4. 上传 `.coverage.gpu` artifact，供需要时与 `.coverage.cpu` 合并查看完整 coverage

手动性能测试：

1. `uv sync --locked` — 在自托管 GPU runner 上同步环境
2. `uv run pytest tests/benchmarks -o python_files='bench_*.py' --benchmark-json=benchmark-result.json -v` — 产出 benchmark 原始结果
3. `uv run taac-bench-report --input benchmark-result.json` — 生成 benchmark 图表缓存供文档站点复用

如果只在本地 CUDA 机器上复现实验，可继续使用 `scripts/run_gpu_tests.py`。
