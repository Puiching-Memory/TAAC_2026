---
icon: lucide/scan-line
---

# Flash Attention Benchmark

## 支持记录

- CLI operator：`flash_attention`。
- Torch reference：`torch.nn.functional.scaled_dot_product_attention`。
- TileLang benchmark 支持 `--is-causal`，默认无 dropout。
- 真实模型路径还涉及 mask 规划、query/key/value layout 转换和训练 backward；修改 mask 逻辑时必须跑 accelerator 单测。
- 主要源码：`src/taac2026/infrastructure/accelerators/attention/flash_attention.py`。

## 推荐命令

```bash
uv run taac-benchmark-pcvr-tilelang-ops \
  --operator flash_attention \
  --device cuda \
  --batch 8 \
  --heads 8 \
  --query-len 128 \
  --kv-len 128 \
  --head-dim 64 \
  --dtype float16 \
  --backends torch,tilelang \
  --steps 100 \
  --warmup-steps 20 \
  --repeats 5 \
  > outputs/benchmarks/tilelang_flash_attention.json
```

## 读数要点

- `query_len`、`kv_len`、`head_dim` 和 causal mask 会显著改变性能。
- 需要同时看 `max_abs_error` 和 `max_rel_error`；attention 的 half precision 误差通常比 RMSNorm 更敏感。
- 训练态 dropout / backward 使用额外 kernels，不能只用 forward-only 结果推断训练收益。

## 最近验收观察

最近一次本地验收口径：H800、`batch=8`、`heads=8`、`query_len=128`、`kv_len=128`、`head_dim=64`、`float16`、`steps=100`、`warmup_steps=20`、`repeats=5`、非 causal forward-only benchmark。结果只用于判断当前 kernel 口径是否健康，不作为长期基准。

| backend  | step ms | compile sec | max abs error | max rel error |
| -------- | ------: | ----------: | ------------: | ------------: |
| torch    |  0.0161 |           - |           0.0 |           0.0 |
| tilelang |  0.0577 |        7.23 |       0.00049 |          7.32 |

当前小形状 forward-only 口径下 TileLang 约 `0.28x`，也就是慢于 torch SDPA。`max_rel_error` 被接近 0 的 reference 值放大；这一组更应看 `max_abs_error`，绝对误差约 `4.9e-4`。Flash Attention 的收益需要继续按真实模型形状、causal/mask、训练 backward 和 dropout 口径重测。
