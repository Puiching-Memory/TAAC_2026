---
icon: lucide/house
---

# TAAC 2026 Experiment Workspace

腾讯广告算法大赛 2026 的实验工作区。基于 PyTorch 构建，专注于 PCVR（点击后转化率）预测任务。

## 核心能力

- **插件式实验包** -- PCVR 实验位于 `experiments/pcvr/`，运维/分析实验位于 `experiments/maintenance/`。每个包独立携带模型定义或任务入口、NS 分组与默认配置，新增实验无需修改框架代码。
- **统一训练/评估/推理入口** -- 通过 `taac-train`、`taac-evaluate` 等 CLI 命令驱动所有实验。
- **可组合数据管道** -- 序列裁剪、特征掩码、域 Dropout、Shuffle Buffer 等增强组件可自由组合。
- **线上打包** -- `taac-package-train` / `taac-package-infer` 生成符合比赛平台要求的 Bundle。

## 内置实验包

| 实验包                                          | 模型           | NS Tokenizer | 亮点                       |
| ----------------------------------------------- | -------------- | ------------ | -------------------------- |
| [Baseline](experiments/baseline.md)             | HyFormer       | group        | 基准参考                   |
| [InterFormer](experiments/interformer.md)       | InterFormer    | group        | 交叉注意力                 |
| [OneTrans](experiments/onetrans.md)             | OneTrans       | rankmixer    | 单 Transformer             |
| [Symbiosis](experiments/symbiosis.md)           | Symbiosis      | rankmixer    | AMP + RoPE + 11 项特性开关 |

## 技术栈

- Python 3.10 - 3.13 / PyTorch 2.7+ / CUDA 12.6
- `uv` 包管理器
- Parquet 列式数据格式
- Ruff 代码风格 / Pytest 测试 / 70% 覆盖率门限
- Zensical (MkDocs Material) 文档站

## 快速预览

```bash
# 安装
uv sync --extra dev --extra cuda126

# 训练
uv run taac-train --experiment experiments/pcvr/baseline

# 评估
uv run taac-evaluate single --experiment experiments/pcvr/baseline
```

PCVR quick preview 默认会通过 `datasets` 拉取 Hugging Face 上的 `demo_1000.parquet`，本地不再支持显式 `--dataset-path`；线上 Bundle 仍由平台提供真实数据路径。
