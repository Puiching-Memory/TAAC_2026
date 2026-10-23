---
icon: lucide/flask-conical
---

# Symbiosis：来源感知的统一 Token 流

Symbiosis 将稀疏字段、dense、缺失状态、候选物品、行为事件和统计摘要编码为统一 token，并用来源与风险信息约束交互。当前主体为 V2，默认启用 V3 确定性 memory selector；分布鲁棒性是待验证假设。

## Token 与读出

`UnifiedTokenBatch` 包含 `(B,T,D)` tokens、padding mask、role/domain/risk ids 和读出位置。value 投影叠加来源信息；dense 使用 robust transform，missing pattern 单独表达。metadata attention 规定不同角色能读取哪些 token。主干结束后使用 CLS、candidate 与全局 masked mean 分类。

## V3 如何分配历史预算

| 域 | recent | memory |
| --- | ---: | ---: |
| seq_a | 8 | 4 |
| seq_b | 8 | 4 |
| seq_c | 20 | 10 |
| seq_d | 24 | 12 |

默认 `quality_stratified` 将 recent 之前的历史分桶，按非零特征密度、有效时间、桶内新近程度和连续重复惩罚选择事件。它不增加可学习路由或额外 loss。raw 序列上限与进入 backbone 的 token 预算是两件事。

## 正确的消融参数

训练配置定义在 `experiments/symbiosis/config.py`，使用 Tyro 的嵌套下划线参数：

```text
--model.no_v3_enabled
--model.v3_memory_selection_mode stratified
--model.v3_memory_selection_mode uniform
--model.no_v2_use_metadata_attention_bias
--model.no_v2_use_candidate_readout
--model.no_v2_use_missing_tokens
--model.no_v2_use_sequence_stats_tokens
--model.v2_tokenization_mode group_compressed
```

不要使用旧的 `--symbiosis-v3-*` 或 argparse 风格命令。已有 sidecar 缺 V3 字段时会校验失败，不会静默采用 V2。

默认增强为 feature mask 0.03、非序列 sparse dropout 0.10、domain dropout 0.02。Muon warmup 为 100 steps，验证间隔 500、patience 2500，dense/sparse 学习率分别为 0.0001/0.01。

## 应观察的证据

同时看时间切分验证、缺失/长序列分桶、token active ratio 与表示有效秩。先固定训练配方，再逐项关闭 selector、metadata 和 missing token；小样本或单 seed 改善不足以证明泛化收益。

## 默认配置

以下值来自当前实验 `__init__.py`，不是硬件容量建议。

| 配置 | 值 |
| --- | --- |
| 模型类 | `PCVRSymbiosis` |
| batch / workers | `128 / 8` |
| split / sampling | `timestamp_auto / row_group_sweep` |
| 序列上限 | `seq_a:256,seq_b:256,seq_c:1024,seq_d:2048` |
| d_model / emb_dim | `256 / 256` |
| blocks / heads | `4 / 8` |
| dense optimizer / lr | `muon / 0.0001` |
| max_steps / patience_steps | `100000 / 2500` |
| AMP / compile | `true / true` |
| seed | `42` |

## 运行与验证

先完成[快速开始](../getting-started.md)的数据准备。在 CPU 上执行一步，并显式关闭本实验可能默认启用的 AMP/compile：

```bash
uv run --with torch==2.13.0 taac-train \
  --experiment experiments/symbiosis \
  --dataset-path data/sample_1000_raw/demo_1000.parquet \
  --schema-path data/sample_1000_raw/schema.json \
  --run-dir outputs/symbiosis_smoke \
  --optimizer.device cpu --data.num_workers 0 \
  --data.batch_size 8 --optimizer.max_steps 1 \
  --runtime.no_amp --runtime.no_compile
```

训练后的评估、推理、打包遵循[统一流程](../getting-started.md)，替换实验路径即可。验收至少包含实验契约和完整 sidecar 重建；CPU smoke 仅说明这次配置的流程可运行。

```bash
uv run --with torch==2.13.0 pytest tests/contract/experiments/test_packages.py tests/contract/experiments/test_runtime_contract_matrix.py -q
```

源码：[实验声明](https://github.com/Puiching-Memory/TAAC_2026/blob/main/experiments/symbiosis/__init__.py)、[模型实现](https://github.com/Puiching-Memory/TAAC_2026/blob/main/experiments/symbiosis/model.py)。涉及 CUDA backend 时补目标设备验证，范围见[测试指南](../guide/testing.md)。
