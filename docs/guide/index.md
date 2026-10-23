---
icon: lucide/book-open-text
---

# 指南：运行、诊断与维护

按当前任务选择一条路径。首次运行先看[快速开始](../getting-started.md)，模型差异在[实验包](../experiments/index.md)。

## 运行和上传

| 任务 | 文档 |
| --- | --- |
| 生成、检查、模拟平台上传物 | [训练与推理 Bundle](online-training-bundle.md) |
| 查看官方操作流程快照 | [平台用户指南](official-competition-docs.md) |
| 核对历史设备、Python 与网络 | [平台环境快照](competition-online-server.md) |

## 调整训练

| 任务 | 文档 |
| --- | --- |
| 理解 parquet、切分、缓存与增强 | [PCVR 数据管道](pcvr-data-pipeline.md) |
| 选择 dense optimizer 与 sparse 更新 | [优化器](optimizers.md) |
| 设置 warmup、衰减与早停 | [学习率调度](lr-schedules.md) |
| 用重计算节省 activation 内存 | [梯度检查点](gradient-checkpointing.md) |
| 接入新模型 | [新增实验包](contributing.md) |

## 验证和维护

| 任务 | 文档 |
| --- | --- |
| 选择测试和复现 CI 口径 | [测试](testing.md) |
| 生成性能 JSON、检查误差 | [性能测量](performance-benchmarks.md) |
| 清理本地可再生缓存 | [缓存清理](cache-cleanup.md) |
| 查找和整理 Actions 记录 | [日志管理](repo-log-management.md) |
| 预览、构建和发布文档 | [本地文档站](local-site.md) |

历史结果在 [Benchmark](../benchmark/index.md)，原始 schema 与官方 baseline 快照在[归档](../archive/index.md)。这些材料不覆盖当前运行契约。
