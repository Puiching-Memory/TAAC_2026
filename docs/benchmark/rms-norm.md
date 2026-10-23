---
icon: lucide/activity
---

# RMSNorm：能力与测量

CLI operator：`rms_norm`。本页说明支持范围与已有测量；重新测量使用相同 shape、dtype、计时边界和误差阈值。

## 能力与历史观察

reference 为 `x * rsqrt(mean(x²) + eps) * weight`。TileLang/Triton 路径面向 CUDA 的 float16、bfloat16、float32；具体最后一维和布局约束由 runtime 校验。模型开关为 `model.rms_norm_backend`，不要把 benchmark 的 backend 选择当作模型已切换。

历史 H800、8192×128、float16 的 forward 记录如下；100 steps、20 warmup、5 repeats。原记录未提供可随页核查的完整 commit 和原始 JSON，不用于当前性能承诺。约 4.9 倍的算子差值还需计入首次 JIT 与完整模型调用比例。

| backend  | step ms | compile sec | max abs error | max rel error |
| -------- | ------: | ----------: | ------------: | ------------: |
| torch    |  0.0536 |           - |           0.0 |           0.0 |
| tilelang |  0.0109 |        4.64 |        0.0078 |        0.0049 |


## 重新测量

在已同步 CUDA 环境中运行：

```bash
mkdir -p outputs/benchmarks
uv run taac-benchmark-pcvr-tilelang-ops \
  --operator rms_norm --device cuda \
  --rows 8192 --cols 128 \
  --dtype float16 --backends torch,tilelang,triton \
  --steps 100 --warmup-steps 20 --repeats 5 --json \
  > outputs/benchmarks/rms-norm.json
```

先检查 backend 状态和误差，再比较稳态时间、编译开销与完整模型收益。训练路径另测 backward、参数梯度与代表性 mask/shape；CPU 的 unsupported 不是 GPU 验证结果。完整记录规范见[性能指南](../guide/performance-benchmarks.md)。

源码位于 `src/taac2026/infrastructure/accelerators/`，测试位于 `tests/unit/infrastructure/accelerators` 与 `tests/gpu`。
