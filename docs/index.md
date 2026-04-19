---
icon: lucide/house
---

# TAAC 2026 Experiment Workspace

**迈向统一序列建模与特征交互的大规模推荐系统**

---

## 项目定位

这是一个面向 [TAAC 2026](https://algo.qq.com/#intro) 的实验工作区。我们把共享训练底座、目录式实验包、统一输出产物和回归测试放进同一套工程里，让新实验可以更快接入、训练、评估和复核。

!!! note "声明"
    本仓库是 TAAC 2026 其中一个参赛队伍的代码仓库，不代表官方。

## 核心能力

| 能力             | 说明                                                          |
| ---------------- | ------------------------------------------------------------- |
| **统一训练框架** | 一条命令完成训练、评估、checkpoint 保存                       |
| **目录式实验包** | 每个实验包独立管理数据、模型、损失函数，互不干扰              |
| **超参数搜索**   | 基于 Optuna，自动检测 GPU 空闲显存并发派发 trial              |
| **回归测试**     | Unit / Integration / Property 三层测试，CI 自动覆盖率门控     |
| **论文复现**     | 内置 InterFormer、OneTrans、HyFormer 等已发表工作的可运行实现 |

## 内置实验包

当前共 **10** 个独立实验包，覆盖从基础 baseline 到前沿论文的多种架构：

| 实验包                                          | 架构特点                        | 来源                                                                                                            |
| ----------------------------------------------- | ------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| [Baseline](experiments/baseline.md)             | 最小参考实现，强调可扩展性      | 本仓库                                                                                                          |
| [CTR Baseline](experiments/ctr-baseline.md)     | DIN 风格注意力                  | [creatorwyx/TAAC2026-CTR-Baseline](https://github.com/creatorwyx/TAAC2026-CTR-Baseline)                         |
| [DeepContextNet](experiments/deepcontextnet.md) | 上下文感知建模                  | [suyanli220/TAAC-2026-Baseline](https://github.com/suyanli220/TAAC-2026-Baseline-Tencent-Advertisement-Contest) |
| [Grok](experiments/grok.md)                     | 分段建模 + pairwise 损失        | 本仓库                                                                                                          |
| [HyFormer](experiments/hyformer.md)             | 多序列分支 + Query Decode/Boost | [论文](https://arxiv.org/abs/2601.12681)                                                                        |
| [InterFormer](experiments/interformer.md)       | 双向序列-特征交互               | [论文](https://arxiv.org/abs/2411.09852)                                                                        |
| [OneTrans](experiments/onetrans.md)             | 统一 Tokenizer + 单 Transformer | [论文](https://arxiv.org/abs/2510.26104)                                                                        |
| [O_o](experiments/oo.md)                        | 简化统一设计                    | [salmon1802/O_o](https://github.com/salmon1802/O_o)                                                             |
| [UniRec](experiments/unirec.md)                 | 多阶段融合                      | [hojiahao/TAAC2026](https://github.com/hojiahao/TAAC2026)                                                       |
| [UniScaleFormer](experiments/uniscaleformer.md) | 缩放序列 + 融合                 | [twx145/Unirec](https://github.com/twx145/Unirec)                                                               |

## 技术栈

- **Python** ≥ 3.12
- **PyTorch** ≥ 2.6
- **uv** 作为包管理器
- **Optuna** ≥ 4.4 用于超参数搜索
- **pytest** + Hypothesis 用于测试

## 快速预览

```bash
# 安装环境
uv python install 3.13
uv sync --locked

# 训练 baseline
uv run taac-train --experiment config/baseline

# 评估
uv run taac-evaluate single --experiment config/baseline

# 超参数搜索
uv run taac-search --experiment config/baseline --trials 20
```

→ 详细步骤见 [快速开始](getting-started.md)
