---
icon: lucide/house
---

# TAAC 2026 实验文档

TAAC 2026 是一个面向 PCVR 二分类的实验工作区。模型放在 `experiments/`，共享框架负责数据读取、训练、评估、推理和上传包生成。先用小样本验证完整流程，再在目标数据与设备上评估效果。

## 从任务进入

| 你要完成的事 | 阅读入口 | 完成后应得到什么 |
| --- | --- | --- |
| 第一次运行 | [快速开始](getting-started.md) | 一份可评估、可推理的 checkpoint |
| 选择模型 | [实验包](experiments/index.md) | 模型差异与默认训练配置 |
| 修改框架 | [架构与契约](architecture.md) | 模块职责和跨流程文件格式 |
| 上传平台 | [训练与推理 Bundle](guide/online-training-bundle.md) | 两个同目录上传文件 |
| 定位数据问题 | [数据管道](guide/pcvr-data-pipeline.md) / [EDA](experiments/online-dataset-eda.md) | 切分、缺失与分布差异证据 |
| 测试或测性能 | [测试](guide/testing.md) / [性能测量](guide/performance-benchmarks.md) | 可复核的检查结果或 JSON |
| 查阅研究与旧材料 | [论文](papers/index.md) / [研究笔记](ideas/index.md) / [归档](archive/index.md) | 来源、假设及适用范围 |

## 使用文档前的三个约定

1. 命令默认在仓库根目录的 Linux shell 中执行。项目支持 Python 3.12–3.14，本地依赖由 `uv.lock` 管理；文档构建与模型运行是不同的验证范围。
2. 本地默认数据来自 Hugging Face 的 demo parquet。仓库只保存参考 schema；首次运行应按照快速开始显式准备数据和 schema。
3. 实验实现、论文报告和历史测量各自有独立证据。实验包存在不表示已复现论文分数，CPU smoke 通过也不表示线上 CUDA 性能已验证。

## 一次实验的产物流向

```text
parquet + schema + 实验配置
           ↓ train
global_step*/{model.safetensors, schema.json, train_config.json}
           ├─ val   → evaluation.json + validation_predictions.jsonl
           └─ infer → predictions.json
```

上传包封装代码与入口，模型权重由训练产物提供。详细操作见[指南索引](guide/index.md)。
