---
icon: lucide/flask-conical
---

# QueryFormer：多列 Embedding 与双向检索

QueryFormer 根据公开方案说明独立实现可明确描述的结构。来源为[团队技术文章](https://mp.weixin.qq.com/s/yG6PnTw4U6OYynNLSpvBYg)；本仓库不拥有其私有实现，当前模型不是官方源码移植，文章成绩也不是此包的复现结果。

## 四类交互

| 模块 | 作用 |
| --- | --- |
| QuerySelfAttn | 用户/物品各自 token 内部交互 |
| QueryCrossAttn | 用户与物品 token 双向交换上下文 |
| SeqQueryCrossAttn | 序列域 query 读取更新后的用户/物品 token |
| QuerySeqCrossAttn | 动态 query 再读取行为序列 |

稀疏字段按组 tokenize，dense 按 schema 原始边界经过 DCNv2。默认四列 embedding，每列有独立 sparse、sequence、DCNv2 和 attention 参数，最后融合列表示。高基数特征使用 hash compression。

## 多列如何执行

实现保留显式 H 维：embedding 使用不相交词表区间，Linear/DCNv2/LayerNorm 参数逐列独立，attention 将 `B×H` 合为 batch。共享 padding row 0 固定为零。dense backbone 可 compile，sparse lookup 保持 eager。

列批处理减少 Python 调度，并不减少 H 倍参数与理论计算量，峰值内存也可能增加。它与整模型 `vmap` 是不同路径，性能须在目标形状重新测量。

## 消融与算法差异

```text
--model.num_embedding_columns 1
--model.no_use_query_self_attention
--model.no_use_query_cross_attention
--model.no_use_seq_query_cross_attention
--model.no_use_query_seq_cross_attention
```

当前 dense optimizer 为仓库 Muon，不能等同于来源提及的 MuonPlus。默认 EMA decay 0.999、cosine 调度、BF16 AMP、compile 和 BCE。列数、DCNv2 层数与开关进入 typed sidecar，评估须保留完整配置。

## 默认配置

以下值来自当前实验 `__init__.py`，不是硬件容量建议。

| 配置 | 值 |
| --- | --- |
| 模型类 | `PCVRQueryFormer` |
| batch / workers | `256 / 8` |
| split / sampling | `timestamp_auto / step_random` |
| 序列上限 | `seq_a:256,seq_b:256,seq_c:512,seq_d:512` |
| d_model / emb_dim | `128 / 64` |
| blocks / heads | `2 / 4` |
| dense optimizer / lr | `muon / 0.0001` |
| max_steps / patience_steps | `100000 / 25000` |
| AMP / compile | `true / true` |
| seed | `2026` |

## 运行与验证

先完成[快速开始](../getting-started.md)的数据准备。在 CPU 上执行一步，并显式关闭本实验可能默认启用的 AMP/compile：

```bash
uv run --with torch==2.13.0 taac-train \
  --experiment experiments/queryformer \
  --dataset-path data/sample_1000_raw/demo_1000.parquet \
  --schema-path data/sample_1000_raw/schema.json \
  --run-dir outputs/queryformer_smoke \
  --optimizer.device cpu --data.num_workers 0 \
  --data.batch_size 8 --optimizer.max_steps 1 \
  --runtime.no_amp --runtime.no_compile
```

训练后的评估、推理、打包遵循[统一流程](../getting-started.md)，替换实验路径即可。验收至少包含实验契约和完整 sidecar 重建；CPU smoke 仅说明这次配置的流程可运行。

```bash
uv run --with torch==2.13.0 pytest tests/contract/experiments/test_packages.py tests/contract/experiments/test_runtime_contract_matrix.py -q
```

源码：[实验声明](https://github.com/Puiching-Memory/TAAC_2026/blob/main/experiments/queryformer/__init__.py)、[模型实现](https://github.com/Puiching-Memory/TAAC_2026/blob/main/experiments/queryformer/model.py)。涉及 CUDA backend 时补目标设备验证，范围见[测试指南](../guide/testing.md)。
