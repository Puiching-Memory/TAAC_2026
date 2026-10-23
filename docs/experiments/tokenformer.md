---
icon: lucide/flask-conical
---

# TokenFormer：统一 Token 流与分层注意力

TokenFormer 是本仓库的统一 token 流实验，关注异构字段融合、注意力范围和表示退化。它不是对一个同名通用模型的完整移植，也没有由结构本身保证的防坍塌效果。

## 输入与主干

```text
user fields / dense → separator → sequence events → separator → item fields / dense → target
```

稀疏字段使用 field-level token；dense 按 packet 投影；各域保留尾部行为事件。separator 与位置编码表达片段边界，target token 负责最终读出。

BFTS（Bottom-Full-Top-Sliding）让底层使用 full causal attention，深层逐步缩小窗口；target 保留全局读取权限。NLIR 门控以 `sigmoid(G(x)) * attention(x)` 调制交互更新，再接 SwiGLU。检查实际收益需要表征诊断与消融，不能把设计动机当作实验结论。

默认增强包含 tail crop、0.02 feature mask 和 0.02 domain dropout。NS 配置仍保存在 sidecar，但模型使用自己的字段级 token 解释。

## 消融与限制

先关闭 RoPE或调整 `seq_top_k`，再比较 BFTS 与全局注意力。替换 NLIR 或修改 mask 属于模型代码变更，需要 forward/backward 和空序列验证。浅层全局注意力的内存成本仍随 token 数增加。

## 默认配置

以下值来自当前实验 `__init__.py`，不是硬件容量建议。

| 配置 | 值 |
| --- | --- |
| 模型类 | `PCVRTokenFormer` |
| batch / workers | `256 / 8` |
| split / sampling | `row_group_tail / step_random` |
| 序列上限 | `seq_a:256,seq_b:256,seq_c:512,seq_d:512` |
| d_model / emb_dim | `64 / 64` |
| blocks / heads | `4 / 4` |
| dense optimizer / lr | `adamw / 0.0001` |
| max_steps / patience_steps | `100000 / 25000` |
| AMP / compile | `true / false` |
| seed | `42` |

## 运行与验证

先完成[快速开始](../getting-started.md)的数据准备。在 CPU 上执行一步，并显式关闭本实验可能默认启用的 AMP/compile：

```bash
uv run --with torch==2.13.0 taac-train \
  --experiment experiments/tokenformer \
  --dataset-path data/sample_1000_raw/demo_1000.parquet \
  --schema-path data/sample_1000_raw/schema.json \
  --run-dir outputs/tokenformer_smoke \
  --optimizer.device cpu --data.num_workers 0 \
  --data.batch_size 8 --optimizer.max_steps 1 \
  --runtime.no_amp --runtime.no_compile
```

训练后的评估、推理、打包遵循[统一流程](../getting-started.md)，替换实验路径即可。验收至少包含实验契约和完整 sidecar 重建；CPU smoke 仅说明这次配置的流程可运行。

```bash
uv run --with torch==2.13.0 pytest tests/contract/experiments/test_packages.py tests/contract/experiments/test_runtime_contract_matrix.py -q
```

源码：[实验声明](https://github.com/Puiching-Memory/TAAC_2026/blob/main/experiments/tokenformer/__init__.py)、[模型实现](https://github.com/Puiching-Memory/TAAC_2026/blob/main/experiments/tokenformer/model.py)。涉及 CUDA backend 时补目标设备验证，范围见[测试指南](../guide/testing.md)。
