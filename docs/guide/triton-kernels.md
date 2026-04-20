---
icon: lucide/cpu
---

# Triton Kernel 开发

## 当前范围

仓库当前已经把 Triton 路径接进共享算子层，而不是停留在单个示例 kernel：

- `src/taac2026/infrastructure/nn/triton_norm.py` 提供 `triton_rms_norm()`
- `src/taac2026/infrastructure/nn/triton_attention.py` 提供共享 `TritonAttention`
- `src/taac2026/infrastructure/nn/triton_ffn.py` 提供共享 FFN activation / SwiGLU 路径
- `src/taac2026/infrastructure/nn/hstu.py` 的 SiLU attention 会复用同一套共享后端
- `src/taac2026/infrastructure/nn/norms.py` 会在 CUDA 上自动走 Triton RMSNorm，在 CPU 上保留参考实现
- `tests/test_triton_kernels.py` 和 `tests/test_transformer_blocks.py` 已覆盖 kernel 正确性与共享 block 集成

这意味着后续 kernel 扩展不需要再从零搭脚手架，可以直接沿用相同的模块组织、测试模式和 GPU 标记约定。

另外，共享 block 现在也支持一个和 Triton 并列的可选 `Transformer Engine` 后端：

- `TaacTransformerBlock` / `TaacCrossAttentionBlock` 可通过 `attention_backend="te"`、`ffn_backend="te"` 走 `transformer_engine.pytorch`
- 自动精度路由会按当前 GPU 选择 `NVFP4 > MXFP8 > FP8 > BF16 > FP16`
- `TaacMixedCausalBlock` 与 HSTU 自定义 attention 仍保持现有实现，不会自动切到 TE

## 开发约定

新增 Triton kernel 时，至少同时交付三部分：

1. 一个共享模块，放在 `src/taac2026/infrastructure/nn/`
2. 一个纯 PyTorch 参考实现，用于数值对齐
3. 一个 GPU 测试，放在 `tests/test_triton_kernels.py` 或新的 `gpu` 分类测试文件里

如果 kernel 会被共享 block 直接消费，再补一个 block 级测试，确保 mask、causal 路径和 fallback 语义没有偏移。

## 最小工作流

```bash
# 仅同步运行 kernel 测试需要的环境
uv sync --locked

# 运行 Triton kernel 正确性测试
uv run pytest tests/test_triton_kernels.py -q

# 运行共享 transformer / HSTU 集成测试
uv run pytest tests/test_transformer_blocks.py -q
```

如果你要验证 TE 后端而不是 Triton 路径，先补齐可选依赖：

```bash
uv sync --locked --extra te --no-build-isolation-package transformer-engine
uv run python scripts/verify_gpu_env.py --json
```

`verify_gpu_env.py` 现在会额外报告 Compute Capability、推荐精度、以及 TE 的 BF16 / FP8 / MXFP8 / NVFP4 可用性，方便在切换 backend 前先确认机器能力。

如果你只改文档站，不需要同步 Triton 本体依赖，可以继续使用：

```bash
uv sync --locked --no-install-package torch --no-install-package torchrec --no-install-package fbgemm-gpu --no-install-package triton
```

## 正确性基线

Triton kernel 的首要约束不是速度，而是和参考实现对齐。仓库当前约定是：

- 优先比较共享函数，比如 `rms_norm()` 这类框架基线实现
- GPU 测试里直接用 `torch.allclose()` 断言
- 默认使用 `float32` 作为首个正确性检查 dtype，再视需要扩展到 `float16` 或 `bfloat16`

## 向前扩展

当前更值得继续推进的方向不是“是否要有 attention / FFN kernel”，而是以下两类增量：

- 扩大共享模块的覆盖面，让更多实验包直接复用 `TritonAttention`、共享 FFN 和 Triton RMSNorm 自动分发
- 在现有 kernel 上补性能回归基准，持续验证 Triton 路径相对参考实现的收益是否稳定
- 对标准 Transformer block 评估 Triton 与 TE 两条后端在不同 GPU 架构上的吞吐差异

无论做哪类扩展，都先保留共享 Python 参考路径，再把 Triton backend 插到相同接口后面，不要把实验包直接绑死到某个 kernel 细节上。