---
icon: lucide/flask-conical
---

# DualQ：双侧 Query 与时间特征

DualQ 将 [zzhlkw-ai/TAAC2026](https://github.com/zzhlkw-ai/TAAC2026) 的 `round2best` 模型思路接入共享运行时。它迁移了模型侧交互，未迁移所有来源数据集的 profile 特征，因此不应宣称复现来源成绩。

## 已实现的路径

用户和物品分别提供 4/2 个 query，构造时要求 `num_queries == user_q_tokens + item_q_tokens`。多域行为经时间对齐 interleave；pair fids `62,63,64,65,66,89,90,91` 从用户普通稀疏组中分流到 pair tokenizer。时间间隔、统计和全局时间 token 从规范输入在模型内派生。

高基数 ID 使用 hash compression。FM Highway 把 user×item、sequence×item 的显式二阶交互直接送往分类头，借鉴 [UniDot](https://github.com/linrongc/kdd26_unidot) 思路，是此仓库增加的消融，并非来源原样代码。

## 迁移边界

| 能力 | 状态 |
| --- | --- |
| 双侧 query、多域 attention、pair tokenizer | 模型内已接入 |
| 时间 gap/stat/global token、高基数压缩 | 模型内已接入 |
| FM Highway、BCE + 0.05 pairwise AUC | 当前配方已接入 |
| id 频次桶、item dense 统计、item time dense | 未迁移来源数据 profile |
| OOF CTR/CVR 目标编码 | 未实现 |
| `use_item_time_query_token` | 缺少配套数据特征，显式拒绝 |
| 来源未启用的 DIN 路径 | 未迁移 |

EMA 默认开启，decay 0.999，start step 1500。pairwise loss 在没有有效正负 pair 的 batch 上贡献零；它不替代独立验证 AUC。

## 消融参数

```text
--model.no_use_fm_highway
--model.no_use_global_time_token
--model.no_use_seq_gap_buckets
--model.no_use_time_gap_domain_gates
--model.no_use_fid87_token_residual
--model.no_use_time_decay_summary
```

时间与 pair 特征必须按实际 schema 核对。未经训练集外统计隔离的 profile/目标编码可能泄漏标签，不能直接在推理时重建。

## 默认配置

以下值来自当前实验 `__init__.py`，不是硬件容量建议。

| 配置 | 值 |
| --- | --- |
| 模型类 | `PCVRDualQ` |
| batch / workers | `576 / 6` |
| split / sampling | `timestamp_auto / step_random` |
| 序列上限 | `seq_a:256,seq_b:256,seq_c:512,seq_d:512` |
| d_model / emb_dim | `192 / 64` |
| blocks / heads | `2 / 4` |
| dense optimizer / lr | `adamw / 0.000483` |
| max_steps / patience_steps | `100000 / 25000` |
| AMP / compile | `true / true` |
| seed | `7789` |

## 运行与验证

先完成[快速开始](../getting-started.md)的数据准备。在 CPU 上执行一步，并显式关闭本实验可能默认启用的 AMP/compile：

```bash
uv run --with torch==2.13.0 taac-train \
  --experiment experiments/dualq \
  --dataset-path data/sample_1000_raw/demo_1000.parquet \
  --schema-path data/sample_1000_raw/schema.json \
  --run-dir outputs/dualq_smoke \
  --optimizer.device cpu --data.num_workers 0 \
  --data.batch_size 8 --optimizer.max_steps 1 \
  --runtime.no_amp --runtime.no_compile
```

训练后的评估、推理、打包遵循[统一流程](../getting-started.md)，替换实验路径即可。验收至少包含实验契约和完整 sidecar 重建；CPU smoke 仅说明这次配置的流程可运行。

```bash
uv run --with torch==2.13.0 pytest tests/contract/experiments/test_packages.py tests/contract/experiments/test_runtime_contract_matrix.py -q
```

源码：[实验声明](https://github.com/Puiching-Memory/TAAC_2026/blob/main/experiments/dualq/__init__.py)、[模型实现](https://github.com/Puiching-Memory/TAAC_2026/blob/main/experiments/dualq/model.py)。涉及 CUDA backend 时补目标设备验证，范围见[测试指南](../guide/testing.md)。
