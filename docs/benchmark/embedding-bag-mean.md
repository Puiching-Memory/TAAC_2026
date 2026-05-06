---
icon: lucide/table-properties
---

# Embedding Bag Mean Benchmark

## 支持记录

- CLI operator：`embedding_bag_mean`。
- Torch reference：`F.embedding(values, weight, padding_idx=0)` 后忽略 id `0` 做 mean pooling。
- TileLang forward 支持 CUDA 上 `float16`、`bfloat16`、`float32` weight，以及 `int32` / `int64` values。
- backward 使用 `T.atomic_add` 累加到 `grad_weight`，训练态性能对 `embedding_dim` 和 `bag_size` 敏感。
- 默认 `backend="auto"` 只在无梯度推理/评估路径自动启用 TileLang；训练有梯度时默认回 torch。要测训练 TileLang backward，显式传 `backend="tilelang"` 或用专项脚本。
- 主要源码：`src/taac2026/infrastructure/accelerators/embedding/embedding_bag.py`。

## 推荐命令

```bash
uv run taac-benchmark-pcvr-tilelang-ops \
  --operator embedding_bag_mean \
  --device cuda \
  --batch 8192 \
  --embedding-vocab-size 1000000 \
  --embedding-dim 64 \
  --embedding-bag-size 4 \
  --embedding-padding-prob 0.25 \
  --dtype float16 \
  --backends torch,tilelang \
  --steps 200 \
  --warmup-steps 50 \
  --repeats 3 \
  --block-cols 64 \
  > outputs/benchmarks/tilelang_embedding_bag_mean.json
```

## 形状 Sweep

```bash
mkdir -p outputs/benchmarks

for dim in 32 64 128; do
  for bag in 1 2 4 8; do
    uv run taac-benchmark-pcvr-tilelang-ops \
      --operator embedding_bag_mean \
      --device cuda \
      --batch 8192 \
      --embedding-vocab-size 1000000 \
      --embedding-dim "$dim" \
      --embedding-bag-size "$bag" \
      --embedding-padding-prob 0.25 \
      --dtype float16 \
      --backends torch,tilelang \
      --steps 200 \
      --warmup-steps 50 \
      --repeats 3 \
      --block-cols "$dim" \
      > "outputs/benchmarks/tilelang_embedding_bag_mean_dim${dim}_bag${bag}.json"
  done
done
```

## 最近验收观察

最近一次本地验收口径：H800、`batch=8192`、`vocab=1_000_000`、`padding_prob=0.25`、`float16`、forward-only benchmark。结果只用于判断这版 kernel 是否值得继续优化，不作为长期基准。

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

训练态 forward + backward 的专项 microbenchmark 显示，atomic backward 会吃掉不少 forward 收益：`dim=64, bag=8` 约 1.9x，`dim=128, bag<=4` 在当次口径下反而慢。因此训练默认不自动走 TileLang embedding backward。
