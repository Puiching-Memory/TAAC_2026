---
icon: lucide/activity
---

# Embedding Bag Mean：能力与测量

CLI operator：`embedding_bag_mean`。本页说明支持范围与已有测量；重新测量使用相同 shape、dtype、计时边界和误差阈值。

## 能力与历史观察

reference 为 embedding lookup 后忽略 id 0 做 mean pooling。TileLang/Triton 路径覆盖 CUDA float16、bfloat16、float32 和 int32/int64 ids；backward 的重复行累加与 atomic 成本需单独验证。

cuEmbed 是本地 CUDA extension 重写，只覆盖 float16/float32 fixed-hotness forward，拒绝 requires-grad 权重。它没有 CSR、weighted sum、concat 或 backward；首次需要 CUDA Toolkit 完成 JIT。来源见[第三方说明](../third-party-notices.md)。

历史 H800、batch=8192、vocab=1000000、padding=0.25、float16 forward 的形状 sweep 保留如下。它不覆盖训练成本；原记录中部分 backward 形状抵消了 forward 收益，因此不能自动接入默认训练。

|  dim |  bag | torch ms | tilelang ms | speedup |
| ---: | ---: | -------: | ----------: | ------: |
|   32 |    1 |    0.068 |       0.035 |   1.95x |
|   32 |    2 |    0.100 |       0.035 |   2.89x |
|   32 |    4 |    0.118 |       0.031 |   3.80x |
|   32 |    8 |    0.128 |       0.034 |   3.74x |
|   64 |    1 |    0.102 |       0.034 |   3.00x |
|   64 |    2 |    0.124 |       0.031 |   4.04x |
|   64 |    4 |    0.137 |       0.040 |   3.46x |
|   64 |    8 |    0.135 |       0.038 |   3.54x |
|  128 |    1 |    0.089 |       0.022 |   4.03x |
|  128 |    2 |    0.139 |       0.035 |   4.01x |
|  128 |    4 |    0.185 |       0.031 |   5.89x |
|  128 |    8 |    0.265 |       0.032 |   8.30x |


## 重新测量

在已同步 CUDA 环境中运行：

```bash
mkdir -p outputs/benchmarks
uv run taac-benchmark-pcvr-tilelang-ops \
  --operator embedding_bag_mean --device cuda \
  --batch 8192 --embedding-vocab-size 1000000 --embedding-dim 64 --embedding-bag-size 4 --embedding-padding-prob 0.25 --block-cols 64 \
  --dtype float16 --backends torch,tilelang,triton,cuembed \
  --steps 100 --warmup-steps 20 --repeats 5 --json \
  > outputs/benchmarks/embedding-bag-mean.json
```

先检查 backend 状态和误差，再比较稳态时间、编译开销与完整模型收益。训练路径另测 backward、参数梯度与代表性 mask/shape；CPU 的 unsupported 不是 GPU 验证结果。完整记录规范见[性能指南](../guide/performance-benchmarks.md)。

源码位于 `src/taac2026/infrastructure/accelerators/`，测试位于 `tests/unit/infrastructure/accelerators` 与 `tests/gpu`。
