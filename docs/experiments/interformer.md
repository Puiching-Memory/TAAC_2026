---
icon: lucide/flask-conical
---

# InterFormer：异构信息交互

InterFormer 让非序列表示与行为序列交替交换信息，避免先把整段历史压成单个向量后再做所有交互。当前包是适配共享 PCVR 运行时的模型实现，原论文结果见[阅读笔记](../papers/interformer.md)。

## 模型如何处理输入

非序列字段采用 group tokenizer，保留配置中的分组边界；序列域分别编码。block 中维护两类表示，通过交叉交互更新上下文，再生成分类输出。与 Baseline 对照时，tokenizer 和 query 数也不同，不能把默认配置差异全部归因于交互结构。

## 验证重点

检查全空序列和 padding mask，确认不同序列域都能接收梯度。做质量比较时固定验证集，另记录序列长度、词表规模、训练耗时和预测分布；论文中的部署收益不能直接用于此包。

## 默认配置

以下值来自当前实验 `__init__.py`，不是硬件容量建议。

| 配置 | 值 |
| --- | --- |
| 模型类 | `PCVRInterFormer` |
| batch / workers | `256 / 8` |
| split / sampling | `row_group_tail / step_random` |
| 序列上限 | `seq_a:256,seq_b:256,seq_c:512,seq_d:512` |
| d_model / emb_dim | `64 / 64` |
| blocks / heads | `2 / 4` |
| dense optimizer / lr | `adamw / 0.0001` |
| max_steps / patience_steps | `100000 / 25000` |
| AMP / compile | `false / false` |
| seed | `42` |

## 运行与验证

先完成[快速开始](../getting-started.md)的数据准备。在 CPU 上执行一步，并显式关闭本实验可能默认启用的 AMP/compile：

```bash
uv run --with torch==2.13.0 taac-train \
  --experiment experiments/interformer \
  --dataset-path data/sample_1000_raw/demo_1000.parquet \
  --schema-path data/sample_1000_raw/schema.json \
  --run-dir outputs/interformer_smoke \
  --optimizer.device cpu --data.num_workers 0 \
  --data.batch_size 8 --optimizer.max_steps 1 \
  --runtime.no_amp --runtime.no_compile
```

训练后的评估、推理、打包遵循[统一流程](../getting-started.md)，替换实验路径即可。验收至少包含实验契约和完整 sidecar 重建；CPU smoke 仅说明这次配置的流程可运行。

```bash
uv run --with torch==2.13.0 pytest tests/contract/experiments/test_packages.py tests/contract/experiments/test_runtime_contract_matrix.py -q
```

源码：[实验声明](https://github.com/Puiching-Memory/TAAC_2026/blob/main/experiments/interformer/__init__.py)、[模型实现](https://github.com/Puiching-Memory/TAAC_2026/blob/main/experiments/interformer/model.py)。涉及 CUDA backend 时补目标设备验证，范围见[测试指南](../guide/testing.md)。
