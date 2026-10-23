---
icon: lucide/activity
---

# Flash Attention：能力与测量

CLI operator：`flash_attention`。本页说明支持范围与已有测量；重新测量使用相同 shape、dtype、计时边界和误差阈值。

## 能力与历史观察

reference 为 PyTorch SDPA，当前公开 backend 为 torch/TileLang。benchmark 可加 `--is-causal`；真实模型还需处理 mask、QKV layout、dropout 与 backward。

历史 H800、B=8、H=8、Q=K=128、D=64、float16、非 causal forward 记录显示 TileLang 慢于 SDPA。相对误差 7.32 受近零 reference 放大，不能脱离约 4.9e-4 的绝对误差及容差单独判定。该记录不是当前所有 mask 与训练路径的验收。

| backend  | step ms | compile sec | max abs error | max rel error |
| -------- | ------: | ----------: | ------------: | ------------: |
| torch    |  0.0161 |           - |           0.0 |           0.0 |
| tilelang |  0.0577 |        7.23 |       0.00049 |          7.32 |


## 重新测量

在已同步 CUDA 环境中运行：

```bash
mkdir -p outputs/benchmarks
uv run taac-benchmark-pcvr-tilelang-ops \
  --operator flash_attention --device cuda \
  --batch 8 --heads 8 --query-len 128 --kv-len 128 --head-dim 64 \
  --dtype float16 --backends torch,tilelang \
  --steps 100 --warmup-steps 20 --repeats 5 --json \
  > outputs/benchmarks/flash-attention.json
```

先检查 backend 状态和误差，再比较稳态时间、编译开销与完整模型收益。训练路径另测 backward、参数梯度与代表性 mask/shape；CPU 的 unsupported 不是 GPU 验证结果。完整记录规范见[性能指南](../guide/performance-benchmarks.md)。

源码位于 `src/taac2026/infrastructure/accelerators/`，测试位于 `tests/unit/infrastructure/accelerators` 与 `tests/gpu`。
