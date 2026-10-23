---
icon: lucide/clipboard-list
---

# 测试：选择范围与复现 CI

从受影响契约开始验证。共享行为变更需要完整 CPU 门禁；模型或训练契约变更再补端到端 smoke；加速器行为必须在可用目标 GPU 上核对。

## 环境

```bash
uv sync --locked --extra dev
```

基础/dev 依赖不安装 Torch。下列 CPU 命令显式注入 `torch==2.13.0`，与 CI 一致。GPU 开发环境则同步 `--extra dev --extra cuda132`，运行时不再注入另一份 Torch。

## 常用命令

```bash
# 快速单测
uv run --with torch==2.13.0 pytest tests/unit -q

# 完整 CPU 门禁
uv run --python 3.13 --with torch==2.13.0 pytest \
  -m "(unit or contract or integration or benchmark_cpu) and not gpu and not benchmark_gpu" -q

# 训练 → checkpoint → 推理
uv run --with torch==2.13.0 pytest tests/integration/application/test_pcvr_roundtrip.py -q

# 实验接口和配置重建
uv run --with torch==2.13.0 pytest tests/contract/experiments -q

# 静态检查与文档
uv run ruff check .
uv run zensical build --strict --clean
```

## 按改动选择检查

| 改动 | 先验证 |
| --- | --- |
| 实验默认值、模型与配置扩展 | `tests/contract/experiments` 与对应 `tests/unit/experiments` |
| 训练 CLI | `tests/unit/application/training/test_cli.py` |
| 评估/推理 | `tests/unit/application/evaluation` |
| 数据转换、cache、切分 | `tests/unit/infrastructure/data` |
| 保存、sidecar | `tests/unit/infrastructure/test_checkpoints.py` 与 roundtrip |
| 打包/bootstrap | `tests/integration/application/packaging`、`tests/unit/application/bootstrap`、`tests/integration/application/bootstrap` |
| 优化器/trainer | `tests/unit/infrastructure/optimization`、`tests/unit/infrastructure/runtime` |
| accelerator | 对应 unit、GPU 数值/梯度与 CLI benchmark |
| 文档 | 严格构建、链接、参数与路径核对 |

测试 marker 在 `pyproject.toml` 注册，由 `tests/conftest.py` 按目录补充。`unit`/`contract`/`integration` 描述范围，`gpu`/`benchmark_gpu` 描述环境需求；CPU 门禁显式排除 GPU。

## CI 的版本和覆盖率

Python 3.12 到 3.14 都在兼容性矩阵中：3.12 和 3.14 运行 unit+contract，3.13 运行完整 CPU 门禁并采集覆盖率。Ruff 在 Python 3.13 执行，Vulture 为非阻塞检查。

| 覆盖率分组 | 门限 |
| --- | ---: |
| 关键训练与推理 runtime | 80% |
| CPU 安全核心 | 80% |
| 七个 PCVR 模型实验 | 75% |

源码 include 列表以 `.github/workflows/ci.yml` 为准。全源码摘要不设门禁，不能用整体百分比替代关键模块覆盖。

```bash
uv run --python 3.13 --with torch==2.13.0 --with coverage \
  coverage run --data-file=.coverage.cpu --source=src/taac2026,experiments \
  -m pytest -m "(unit or contract or integration or benchmark_cpu) and not gpu and not benchmark_gpu" -v
```

CI 路径过滤包含 workflow、experiments、src、tests、pyproject 与 lockfile。纯 docs 改动由 Pages workflow 严格构建，不触发 Python CI；混合代码/文档改动等待 CI 成功后部署，见[站点指南](local-site.md)。

## GPU 与性能验证

在已经同步 CUDA extra 的目标环境：

```bash
uv run pytest tests/unit/infrastructure/accelerators -q
uv run pytest -m gpu tests/gpu -q
```

记录 skip 原因、设备和命令。没有 CUDA 时的 skip 不支持“GPU 已通过”。算子需验证 forward、backward、mask、dtype 和误差；性能还需[相同口径测量](performance-benchmarks.md)，短 smoke 不代表完整训练或线上环境。

## 提交与结果报告

需要本地提交钩子时运行 `bash tools/install-git-hooks.sh`；钩子通过 uv 执行 Ruff，不能代替测试。提交前检查 diff 与 `git diff --cached --check`。

报告实际执行命令、通过/失败/skip 和未覆盖环境。文档变更只需文档相关检查，不必为纯文字修改启动模型训练。
