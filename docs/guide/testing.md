---
icon: lucide/clipboard-list
---

# 测试

测试的目标不是“每次都跑最重的一套”，而是先覆盖你刚改动的契约，再在合并前扩大范围。

## 准备环境

```bash
uv sync --locked --extra dev --extra cuda126
```

只做 CPU 单测时，通常 `--extra dev` 已经足够；涉及本地 CUDA 训练或 accelerator 时再同步 CUDA profile。

## 最常用命令

```bash
# 快速单元测试
uv run pytest tests/unit -q

# PR CPU 门禁口径
uv run pytest -m "(unit or contract or integration or benchmark_cpu) and not gpu and not benchmark_gpu" -q

# 实验包契约
uv run pytest tests/contract/experiments/test_packages.py -q
uv run pytest tests/contract/experiments/test_runtime_contract_matrix.py -q

# 打包与 run.sh 集成
uv run pytest tests/integration/application/packaging -q
uv run pytest tests/unit/application/bootstrap tests/integration/application/bootstrap -q

# 风格检查
uv run ruff check .

# 文档构建
uv run zensical build --strict
```

## Pytest 标记

`pyproject.toml` 注册了这些 marker：

| marker          | 含义                            |
| --------------- | ------------------------------- |
| `unit`          | 快速单元测试                    |
| `contract`      | 稳定接口、实验包和 bundle 契约  |
| `integration`   | 跨模块流程测试                  |
| `gpu`           | 需要 CUDA GPU 的测试            |
| `benchmark_cpu` | 可在 CPU 上跑的 benchmark 测试  |
| `benchmark_gpu` | 只适合本地 GPU CLI 的 benchmark |

`tests/conftest.py` 会按路径自动给 `tests/unit/**`、`tests/contract/**`、`tests/integration/**`、`tests/benchmarks/**` 和 `tests/gpu/**` 标对应 marker。因此 CI 用 marker 表达 CPU 安全门禁，而不是依赖单一目录。

## 按改动选择最小复核

| 改动位置                         | 先跑                                                                                                      |
| -------------------------------- | --------------------------------------------------------------------------------------------------------- |
| `experiments/<name>/__init__.py` | `tests/contract/experiments/test_packages.py`，`tests/contract/experiments/test_runtime_contract_matrix.py` |
| `experiments/<name>/model.py`    | `tests/contract/experiments/test_packages.py`                                                           |
| 维护类实验                       | `tests/contract/experiments/test_maintenance_experiments.py`                                            |
| Online Dataset EDA               | `tests/contract/experiments/test_online_dataset_eda_runner.py`                                          |
| 训练 CLI / args                  | `tests/unit/application/training/test_cli.py`                                                             |
| 评估 / 推理 CLI                  | `tests/unit/application/evaluation`                                                                       |
| 打包逻辑                         | `tests/integration/application/packaging`，`tests/unit/application/bootstrap`，`tests/integration/application/bootstrap` |
| checkpoint / sidecar             | `tests/unit/infrastructure/test_checkpoints.py`，`tests/contract/experiments/test_runtime_contract_matrix.py` |
| 数据管道                         | `tests/unit/infrastructure/data`                                                                          |
| accelerator                      | `tests/unit/infrastructure/accelerators`                                                                  |
| 文档                             | `uv run zensical build --strict`                                                                          |

## CI 会做什么

以 `.github/workflows/ci.yml` 为准，当前 CI 主要做三件事：

- Python 3.13 的 Ruff 检查。
- Python 3.10 到 3.13 的 CPU 测试：`unit`、`contract`、`integration` 和 `benchmark_cpu`，排除 `gpu` / `benchmark_gpu`。
- 在规范 Python 版本上采集全 `src/taac2026` 覆盖率，并对指定核心模块做覆盖率门控。

CI 只会被这些路径触发：

```text
.github/workflows/ci.yml
experiments/**
src/**
tests/**
pyproject.toml
uv.lock
```

纯 `docs/**` 改动不跑 CI；文档部署由 Pages workflow 处理。CI 的 `uv sync` 和 `uv run` 都会显式传 `--python`，避免本地 `.python-version` 覆盖 workflow matrix。

本地复现 CPU 单测口径：

```bash
uv run --python 3.13 --with torch==2.7.1 --with coverage \
  coverage run --data-file=.coverage.cpu --source=src/taac2026 \
  -m pytest -m "(unit or contract or integration or benchmark_cpu) and not gpu and not benchmark_gpu" -v
```

覆盖率报告：

```bash
cp .coverage.cpu .coverage
uv run --python 3.13 --with coverage coverage report --fail-under=70 \
  --include='src/taac2026/domain/*,src/taac2026/application/training/__init__.py,src/taac2026/application/training/args.py,src/taac2026/application/training/cli.py,src/taac2026/application/training/workflow.py'
```

覆盖率门控只覆盖 `domain/` 和 training CLI / workflow 的核心文件。它不是全仓库覆盖率，新增实验模型时仍以契约测试和 smoke 为准。
CI 也会输出全 `src/taac2026` 覆盖率摘要作为盲区提示，但这份摘要暂不作为 fail-under 门禁。

## GPU 和 Benchmark

GPU 相关测试在没有 CUDA 时会 skip，不能用 CPU CI 结果证明 TileLang kernel 可用。改 accelerator 后至少在有 CUDA 的机器上跑：

```bash
uv run pytest tests/unit/infrastructure/accelerators -q
uv run pytest -m gpu tests/gpu -q
```

benchmark 入口的 CPU 通路测试在 `tests/benchmarks/cpu`，会进入 CPU 门禁；真实性能结论仍然不要塞进常规 CI。需要性能结论时，用 CLI 生成 JSON，记录硬件、CUDA / PyTorch 版本、commit 和完整命令。

## Smoke Test 的位置

训练 smoke 是集成验证，不替代单元测试。它适合在下面几种情况补跑：

- 新增模型包后确认端到端能走完。
- 改了训练 workflow、checkpoint 或模型输入契约。
- 单测通过，但你怀疑真实 batch 协作会出问题。

示例：

```bash
uv run pytest tests/integration/application/bootstrap/test_cpu_smoke.py -q

bash run.sh train \
  --experiment experiments/baseline \
  --run-dir outputs/baseline_smoke \
  --device cpu \
  --num_workers 0 \
  --batch_size 8 \
  --max_steps 1
```

## 常见失败

| 现象                                              | 优先检查                                              |
| ------------------------------------------------- | ----------------------------------------------------- |
| `unrecognized arguments`                          | 训练参数多为下划线，评估 / 推理参数多为连字符         |
| `local PCVR runs no longer accept --dataset-path` | 普通 PCVR 本地测试不要传数据路径                      |
| accelerator 测试 skip                             | 当前机器没有 CUDA 或缺 TileLang / Triton runtime      |
| bundle 测试失败但本地 import 正常                 | 检查 zip 内容和 manifest，而不是只看仓库根目录 import |
| 文档构建失败                                      | 先看相对链接、`zensical.toml` nav 和图片路径          |
