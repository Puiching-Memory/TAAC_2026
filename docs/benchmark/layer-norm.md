---
icon: lucide/activity
---

# LayerNorm：能力与测量

CLI operator：`layer_norm`。本页说明支持范围与已有测量；重新测量使用相同 shape、dtype、计时边界和误差阈值。

## 能力与历史观察

reference 为 PyTorch last-dim affine LayerNorm。Triton 支持 CUDA float16、bfloat16、float32 和一维 affine 参数；inference 有 forward-only 路径，训练保存 backward 所需统计。TileLang 未接入。

历史记录来自 H800、CUDA 12.6、Torch 2.7.1+cu126、8192×128、float16、no-grad。两种计时口径得到不同排序：纯 kernel 稳态较快，不代表 Python 循环调用更快。表中比值应理解为 **torch 耗时 / Triton 耗时**。当前选择需重测，不能据此全局替换 `nn.LayerNorm`。

| 口径 | torch ms | triton ms | torch / triton | max abs error |
| ---- | -------: | --------: | -------------: | ------------: |
| `taac-benchmark-pcvr-tilelang-ops` / `do_bench` | 0.0171 | 0.0111 | 1.54x | 0.001953 |
| 预编译 callable 的 Python 循环 + CUDA event | 0.0182 | 0.0458 | 0.40x | 0.003906 |


## 重新测量

在已同步 CUDA 环境中运行：

```bash
mkdir -p outputs/benchmarks
uv run taac-benchmark-pcvr-tilelang-ops \
  --operator layer_norm --device cuda \
  --rows 8192 --cols 128 \
  --dtype float16 --backends torch,triton \
  --steps 100 --warmup-steps 20 --repeats 5 --json \
  > outputs/benchmarks/layer-norm.json
```

先检查 backend 状态和误差，再比较稳态时间、编译开销与完整模型收益。训练路径另测 backward、参数梯度与代表性 mask/shape；CPU 的 unsupported 不是 GPU 验证结果。完整记录规范见[性能指南](../guide/performance-benchmarks.md)。

源码位于 `src/taac2026/infrastructure/accelerators/`，测试位于 `tests/unit/infrastructure/accelerators` 与 `tests/gpu`。
