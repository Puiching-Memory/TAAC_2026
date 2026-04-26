---
icon: lucide/house
---

# TAAC 2026 Experiment Workspace

**迈向统一序列建模与特征交互的大规模推荐系统**

这是一个面向 [TAAC 2026](https://algo.qq.com/#intro) 的实验工作区。仓库把共享 PCVR 训练运行时、目录式实验包、线上训练打包、评估和回归测试放在同一套工程里，让每个新模型都能沿着同一条路径接入、训练和复核。

!!! note "声明"
    本仓库是 TAAC 2026 其中一个参赛队伍的代码仓库，不代表官方。

## 核心能力

| 能力 | 说明 |
| --- | --- |
| 统一入口 | 顶层 `run.sh` 覆盖训练、验证、推理、测试和打包 |
| PCVR 实验包 | 每个 `config/<name>/` 包声明 `PCVRExperiment`、模型类和 `ns_groups.json` |
| 共享训练运行时 | 数据读取、模型构建、trainer、checkpoint 和评估协议集中在 `src/taac2026/infrastructure/pcvr` |
| 线上训练打包 | 生成平台上传用的 `run.sh` 与 `code_package.zip` 双文件目录 |
| 搜索与报告 | 保留搜索请求记录、EDA、技术时间线和占位 benchmark 报告工具 |
| 单元回归 | 当前可执行回归集中在 `tests/unit/`，覆盖实验包、协议、打包和 CLI 契约 |

## 内置实验包

当前共有 **9** 个 PCVR 实验包：

| 实验包 | 目录 | 说明 |
| --- | --- | --- |
| [Baseline](experiments/baseline.md) | `config/baseline` | 官方 HyFormer 风格 baseline，保留 `PCVRHyFormer` 名称 |
| [Symbiosis](experiments/symbiosis.md) | `config/symbiosis` | 本仓库维护的融合式 PCVR 实验模型 |
| [CTR Baseline](experiments/ctr-baseline.md) | `config/ctr_baseline` | DIN/CTR 风格轻量对照模型 |
| [DeepContextNet](experiments/deepcontextnet.md) | `config/deepcontextnet` | 上下文增强的深度交互模型 |
| [HyFormer](experiments/hyformer.md) | `config/hyformer` | HyFormer 论文方向的实验包 |
| [InterFormer](experiments/interformer.md) | `config/interformer` | 序列与非序列特征交互建模 |
| [OneTrans](experiments/onetrans.md) | `config/onetrans` | 统一 token 化与单 Transformer 建模 |
| [UniRec](experiments/unirec.md) | `config/unirec` | 多阶段融合方向实验包 |
| [UniScaleFormer](experiments/uniscaleformer.md) | `config/uniscaleformer` | 缩放序列建模与融合实验包 |

## 技术栈

- Python `>=3.10,<3.14`，本地推荐使用仓库固定的 Python 3.10.20。
- 本地依赖管理使用 `uv` 与 `uv.lock`。
- 仅支持 Linux 运行时。
- 仓库默认依赖已包含 pytest、hypothesis 和 benchmark 工具；CUDA 运行时统一使用 `cuda126` extra，对齐线上 CUDA 12.6 事实。
- 线上 bundle 默认使用平台已激活的 Python/Conda 环境，不要求平台安装 `uv`。

## 快速预览

```bash
uv python install 3.10.20
uv sync --locked --extra cuda126

bash run.sh train --experiment config/baseline \
    --dataset-path /path/to/parquet_or_dataset_dir \
    --schema-path /path/to/schema.json \
    --num_epochs 1 \
    --batch_size 8 \
    --device cpu

bash run.sh val --experiment config/baseline \
    --dataset-path /path/to/parquet_or_dataset_dir \
    --schema-path /path/to/schema.json \
    --run-dir outputs/config/baseline \
    --device cpu

bash run.sh package --experiment config/baseline --force
uv run pytest tests/unit -q
```

→ 详细步骤见 [快速开始](getting-started.md)

→ 线上双文件格式见 [线上训练打包](guide/online-training-bundle.md)

→ 新实验包契约见 [架构与概念](architecture.md) 与 [新增实验包](guide/contributing.md)