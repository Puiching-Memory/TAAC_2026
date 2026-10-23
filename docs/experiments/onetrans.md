---
icon: lucide/flask-conical
---

# OneTrans：统一 Transformer

OneTrans 把静态字段与行为序列放入同一 Transformer 计算路径，研究统一交互和序列建模。该包是 OneTrans-inspired PCVR 实现，背景见[论文笔记](../papers/onetrans.md)。

## 模型如何处理输入

静态稀疏字段经 RankMixer tokenizer 形成固定预算 token，dense 与序列 token 一同参与主干交互。金字塔式序列收缩控制深层计算量，减少每层都处理完整长序列的成本。

默认配置面沿用共享 `PCVRModelConfig`。字段存在不代表所有模型都以相同方式使用它；解释某个开关的效果时应核对 `model.py` 的消费位置。

## 验证重点

比较统一序列长度与每层保留预算，检查 target/静态上下文是否保持可见。与 TokenFormer 比较时，要控制输入 token 数、层数、增强与位置编码，区分 token 收缩和 sliding mask 的影响。

## 默认配置

以下值来自当前实验 `__init__.py`，不是硬件容量建议。

| 配置 | 值 |
| --- | --- |
| 模型类 | `PCVROneTrans` |
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
  --experiment experiments/onetrans \
  --dataset-path data/sample_1000_raw/demo_1000.parquet \
  --schema-path data/sample_1000_raw/schema.json \
  --run-dir outputs/onetrans_smoke \
  --optimizer.device cpu --data.num_workers 0 \
  --data.batch_size 8 --optimizer.max_steps 1 \
  --runtime.no_amp --runtime.no_compile
```

训练后的评估、推理、打包遵循[统一流程](../getting-started.md)，替换实验路径即可。验收至少包含实验契约和完整 sidecar 重建；CPU smoke 仅说明这次配置的流程可运行。

```bash
uv run --with torch==2.13.0 pytest tests/contract/experiments/test_packages.py tests/contract/experiments/test_runtime_contract_matrix.py -q
```

源码：[实验声明](https://github.com/Puiching-Memory/TAAC_2026/blob/main/experiments/onetrans/__init__.py)、[模型实现](https://github.com/Puiching-Memory/TAAC_2026/blob/main/experiments/onetrans/model.py)。涉及 CUDA backend 时补目标设备验证，范围见[测试指南](../guide/testing.md)。
