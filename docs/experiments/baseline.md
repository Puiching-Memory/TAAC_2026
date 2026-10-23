---
icon: lucide/flask-conical
---

# Baseline：HyFormer 参照

Baseline 保留 HyFormer 的多域 query 交互结构，默认关闭数据增强、cache、AMP 和 compile，适合作为运行流程和训练配方的参照。这里的“参照”不表示它在所有数据集上表现最优。

## 模型如何处理输入

用户与物品稀疏字段由 RankMixer tokenizer 压成固定数量的非序列 token；dense 字段投影后加入上下文。各序列域独立 embedding 并融合时间桶，query 在 HyFormer block 中读取序列，再与静态特征交互，最后分类。

默认用户/物品 RankMixer token 数为 5/2，query 数为 2。字段分组保存在 `model.ns`，不能把 fid 当作 schema 的连续下标。高基数字段受 `emb_skip_threshold` 等设置影响；新增数据时要核对实际保留特征。

本包借鉴[HyFormer](../papers/hyformer.md)，使用当前 PCVR schema、统一 trainer 和 sidecar，并非论文数据集与训练预算的完整复现。

## 适合做的对照

先固定数据切分、batch、seed 和训练步数，只比较一个变量：模型结构、增强、优化器或 backend。若与 Symbiosis 等默认配方比较，必须记录这些训练变量的差异。

## 默认配置

以下值来自当前实验 `__init__.py`，不是硬件容量建议。

| 配置 | 值 |
| --- | --- |
| 模型类 | `PCVRHyFormer` |
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
  --experiment experiments/baseline \
  --dataset-path data/sample_1000_raw/demo_1000.parquet \
  --schema-path data/sample_1000_raw/schema.json \
  --run-dir outputs/baseline_smoke \
  --optimizer.device cpu --data.num_workers 0 \
  --data.batch_size 8 --optimizer.max_steps 1 \
  --runtime.no_amp --runtime.no_compile
```

训练后的评估、推理、打包遵循[统一流程](../getting-started.md)，替换实验路径即可。验收至少包含实验契约和完整 sidecar 重建；CPU smoke 仅说明这次配置的流程可运行。

```bash
uv run --with torch==2.13.0 pytest tests/contract/experiments/test_packages.py tests/contract/experiments/test_runtime_contract_matrix.py -q
```

源码：[实验声明](https://github.com/Puiching-Memory/TAAC_2026/blob/main/experiments/baseline/__init__.py)、[模型实现](https://github.com/Puiching-Memory/TAAC_2026/blob/main/experiments/baseline/model.py)。涉及 CUDA backend 时补目标设备验证，范围见[测试指南](../guide/testing.md)。
