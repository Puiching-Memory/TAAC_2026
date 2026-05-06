---
icon: lucide/house
---

# TAAC 2026 Docs

这是 TAAC 2026 实验工作区的文档站。它服务两类读者：想尽快把实验跑起来的人，以及准备修改实验包、运行时或线上 bundle 的人。

如果你第一次打开这个仓库，先读 [快速开始](getting-started.md)。分区入口负责导览，具体页面负责实现细节。

## 我想做什么

| 目标 | 去哪里 |
| ---- | ------ |
| 本地跑一次 baseline | [快速开始](getting-started.md) |
| 选择或比较实验包 | [实验包总览](experiments/index.md) |
| 找任务型文档 | [指南总览](guide/index.md) |
| 查 schema / 官方快照 | [归档](archive/index.md) |
| 读研究笔记 | [想法](ideas/index.md) 与 [论文](papers/index.md) |

## 当前仓库在做什么

主任务是 PCVR 二分类。仓库提供：

- 统一实验包入口：`experiments/<name>/`
- 本地训练、评估、推理入口：`bash run.sh train|val|eval|infer`
- 线上上传物生成：`taac-package-train` 和 `taac-package-infer`
- 可复用 PCVR 数据管道、模型组件、optimizer 和 accelerator backend
- 面向 GitHub Pages 的 Zensical 文档站

本地 PCVR smoke 会使用默认 demo parquet；不要给本地 PCVR 训练显式传 `--dataset-path`。线上 bundle 由平台变量提供真实数据路径。

## 文档分工

- 顶层页面说明仓库是什么、怎么启动、代码怎么分层。
- 分区 `index.md` 负责导览和选择。
- 实验页记录具体实验的模型结构、默认配置、修改点和测试契约。
- 指南页记录任务的实现细节、命令、输入输出、环境变量和排障。
- 归档页保存历史资产，不代表当前运行时契约。

## 入口速查

```bash
# 本地训练
bash run.sh train --experiment experiments/baseline --run-dir outputs/baseline_smoke

# 本地评估
bash run.sh val --experiment experiments/baseline --run-dir outputs/baseline_smoke

# 推理
bash run.sh infer --experiment experiments/baseline --checkpoint outputs/baseline_smoke --result-dir outputs/baseline_infer

# 打包
uv run taac-package-train --experiment experiments/baseline --output-dir outputs/bundles/baseline_training
uv run taac-package-infer --experiment experiments/baseline --output-dir outputs/bundles/baseline_inference
```
