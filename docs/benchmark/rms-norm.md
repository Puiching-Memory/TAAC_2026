---
icon: lucide/activity
---

# RMSNorm Benchmark

## 支持记录

- CLI operator：`rms_norm`。
- Torch reference：`x * rsqrt(mean(x^2) + eps) * weight`。
- TileLang backend 支持 `float16`、`bfloat16`、`float32`，CUDA tensor，last dimension 需要符合当前 kernel 约束。
- benchmark 只测算子本体，不代表完整模型 step time。
- 主要源码：`src/taac2026/infrastructure/accelerators/normalization/rms_norm.py`。

## 推荐命令

```bash
uv run taac-benchmark-pcvr-tilelang-ops \
  --operator rms_norm \
  --device cuda \
  --rows 8192 \
  --cols 128 \
  --dtype float16 \
  --backends torch,tilelang \
  --steps 100 \
  --warmup-steps 20 \
  --repeats 5 \
  > outputs/benchmarks/tilelang_rms_norm.json
```

## 读数要点

- 变更 `block_rows` 时要一起记录参数，例如 `--block-rows 8`。
- 如果 `tilelang` backend 报 unsupported，先确认 CUDA、TileLang 安装和 dtype / shape 约束。
- RMSNorm 的模型收益取决于调用频率和 tensor shape；单算子速度提升不等于端到端同幅提升。

## 最近验收观察

最近一次本地验收口径：H800、`rows=8192`、`cols=128`、`float16`、`steps=100`、`warmup_steps=20`、`repeats=5`。结果只用于判断当前 kernel 口径是否健康，不作为长期基准。

| backend  | step ms | compile sec | max abs error | max rel error |
| -------- | ------: | ----------: | ------------: | ------------: |
| torch    |  0.0536 |           - |           0.0 |           0.0 |
| tilelang |  0.0109 |        4.64 |        0.0078 |        0.0049 |

当前口径下 TileLang 约 `4.9x` faster。首次 JIT 编译约 4.6s，评估稳态吞吐时不要把它混入 step time。
