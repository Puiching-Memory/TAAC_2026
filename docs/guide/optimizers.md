---
icon: lucide/settings-2
---

# 优化器：Dense 与 Sparse 分开配置

共享 trainer 将稀疏 embedding 与稠密参数交给不同 optimizer。切换 dense optimizer 不会改变 sparse 学习率；比较时需要记录两组配置。

## 可选 Dense optimizer

| 值 | 行为 |
| --- | --- |
| `adamw` | PyTorch AdamW，betas `(0.9,0.98)` |
| `fused_adamw` | fused AdamW，不支持的设备或参数组合会失败 |
| `orthogonal_adamw` | AdamW 更新前，对矩阵 dense gradient 做正交化处理 |
| `muon` | 仓库内 Muon，矩阵参数用 Muon，非矩阵用 AdamW 风格更新 |

```bash
uv run --with torch==2.13.0 taac-train \
  --experiment experiments/baseline \
  --dataset-path data/sample_1000_raw/demo_1000.parquet \
  --schema-path data/sample_1000_raw/schema.json \
  --run-dir outputs/baseline_muon \
  --optimizer.dense_optimizer_type muon \
  --optimizer.lr 0.0001 --optimizer.max_steps 10000
```

先按[快速开始](../getting-started.md)准备数据。实验默认值中 Symbiosis、QueryFormer 使用 Muon，其余模型使用 AdamW；CLI 可覆盖。

## Sparse 更新与梯度处理

模型的 `get_sparse_params()` 和 `get_dense_params()` 决定分组。没有 sparse 契约的参数按 dense 处理。Sparse embedding 使用 `PCVRSparseAdagrad`，学习率来自 `--sparse_optimizer.sparse_lr`，没有 weight decay 配置。

trainer 合并 COO 重复行后计算 dense/sparse 联合范数，执行裁剪；非有限范数会跳过两组更新，不推进 optimizer step 和 EMA。fp16 AMP 的 unscale 也通过同一 optimizer 接口执行。

Sparse Adagrad 按行更新参数与累积量，公式为：

```text
sum += grad²
param -= lr * grad / (sqrt(sum) + eps)
```

若按配置重置高基数 embedding，trainer 清除对应参数的 Adagrad 状态，而不是重建整个 optimizer。

## Muon 的特殊参数布局

Muon registry 固定 momentum 0.95、Nesterov 开启、Newton–Schulz 5 步、weight decay 0.01；非矩阵分支 betas `(0.9,0.98)`、eps `1e-8`。具体实现见 `infrastructure/optimization/muon.py` 与 `registry.py`。

`mark_muon_batched_matrix()` 让 `[H,rows,cols]` 参数逐列独立正交化，避免把 H 列错误合并；`mark_muon_adamw()` 指定高维 affine 参数走 AdamW 分支。新增批处理布局需核对这些标记。Muon 不接受 sparse gradient，也不能等同于外部方案的 MuonPlus。

这里描述 optimizer 的状态能力，不代表三件套 checkpoint 包含可完整续训的所有状态。模型重建与训练恢复应分别验证。

## 如何比较

先用[optimizer benchmark](performance-benchmarks.md)检查每步开销，再固定数据、seed、步数与学习率策略比较收敛。合成 MLP 的耗时不包含真实 embedding/数据管道成本，单步快不能替代 time-to-quality。

相关源码位于 `src/taac2026/infrastructure/optimization/` 与 `runtime/trainer.py`；验证使用 `tests/unit/infrastructure/optimization` 和 `tests/unit/infrastructure/runtime/test_trainer.py`。调度和早停见[下一页](lr-schedules.md)。
