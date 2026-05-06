---
icon: lucide/archive
---

# 重要资产归档

这里存放需要随文档站长期保留的参考资产。归档页的目标是提供稳定的浏览与下载入口，不改变仓库里的默认运行时路径，也不替代当前受维护的实验包实现。

## 当前归档项

| 资产                   | 用途                               | 入口                                                  |
| ---------------------- | ---------------------------------- | ----------------------------------------------------- |
| sample_1000_raw schema | 本地 demo 数据集的默认 schema 参考 | [样例 Schema](sample-schema.md)                       |
| online_schema schema   | 线上训练与 infer 共用的声明式 schema 快照 | [Online Schema 快照](training-schema-snapshot.md) |
| 官方训练 Baseline 快照 | 比赛参考训练基线的历史源码快照     | [官方训练 Baseline 快照](official-train-baseline.md)  |
| 官方推理 Baseline 快照 | 比赛参考推理基线的历史源码快照     | [官方推理 Baseline 快照](official-infer-baseline.md)  |

## 归档原则

- 归档副本放在 docs 目录下，确保 GitHub Pages 构建时能直接发布。
- 所有 schema 静态副本统一放在 files/schema 目录下，通过文件名区分不同来源和用途。
- 本地和线上运行仍以仓库真实入口为准，例如 data/sample_1000_raw、experiments/baseline 与打包 CLI。
- 历史参考快照用于对照、追溯和资料保存，不应直接当作当前框架的长期运行时依赖。