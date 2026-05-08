---
icon: lucide/gauge
---

# Benchmark 总览

这个目录记录 TAAC 里的 benchmark 口径，包括数据管道 cache 策略和 accelerator 算子的逐项记录。每个独立 benchmark 页面都应该写清楚支持状态、可复现命令、关键参数、误差或吞吐口径和最近一次本地验收观察。

这里的数字不是长期事实。正式结论必须重新记录 commit、硬件、CUDA / PyTorch / TileLang 版本、完整命令和 JSON 输出。推荐输出到 `outputs/benchmarks/`，不要提交生成结果，除非要做一次明确的报告快照。

## 数据管道索引

| 主题                   | CLI 入口                            | 当前用途                                   | 页面                                      |
| ---------------------- | ----------------------------------- | ------------------------------------------ | ----------------------------------------- |
| PCVR smoke diagnostics | `taac-plot-pcvr-diagnostics`        | demo1000 下的运行成本、预测行为和稳定性图 | [PCVR Smoke Diagnostics](pcvr-diagnostics.md) |
| Cache policies         | `taac-benchmark-pcvr-data-pipeline` | 比较 `lru/fifo/lfu/rr/opt` 策略            | [Cache Policies](cache-policies.md)       |

## 算子索引

| 算子               | CLI `--operator`     | 当前用途                                      | Backend 策略                                            | 页面                                        |
| ------------------ | -------------------- | --------------------------------------------- | ------------------------------------------------------- | ------------------------------------------- |
| RMSNorm            | `rms_norm`           | normalization forward/backward microbenchmark | 由模型 runtime 的 RMSNorm backend 配置决定              | [RMSNorm](rms-norm.md)                      |
| Flash Attention    | `flash_attention`    | attention forward/backward 和 mask 约束验证   | 由 sequence runtime 的 flash attention backend 配置决定 | [Flash Attention](flash-attention.md)       |
| Embedding bag mean | `embedding_bag_mean` | non-sequential sparse feature mean pooling    | 默认 `torch`；显式传 `tilelang` 或 `cuembed` 才启用 accelerator | [Embedding Bag Mean](embedding-bag-mean.md) |

通用命令入口：

```bash
uv run taac-benchmark-pcvr-tilelang-ops --help
```

输出 JSON 中优先看：

| 字段                              | 含义                                                                        |
| --------------------------------- | --------------------------------------------------------------------------- |
| `status`                          | `ok`、`unsupported` 或 `error`；`unsupported` 常见于 CPU、缺 TileLang、缺 CUDA toolkit 或 cuEmbed JIT 不可用 |
| `resolved_backend`                | 实际使用的 backend                                                          |
| `step_time_ms_mean`               | 多次 repeat 后的平均单步耗时                                                |
| `compile_sec`                     | accelerator 首次 JIT 编译时间，不应混入稳态吞吐判断                         |
| `max_abs_error` / `max_rel_error` | 与 torch reference 的误差                                                   |

## 新增算子页面模板

新增 `--operator` 后，在 `docs/benchmark/` 下补一个单独页面，并把它加入本页索引和 `zensical.toml` 导航。建议直接复制下面模板：

````markdown
---
icon: lucide/cpu
---

# Operator Name

## 支持记录

- Torch reference：...
- TileLang backend：dtype、device、shape 和 mask 约束。
- 自动启用策略：...

## 推荐命令

```bash
uv run taac-benchmark-pcvr-tilelang-ops \
  --operator ... \
  --device cuda \
  --dtype float16 \
  --backends torch,tilelang \
  > outputs/benchmarks/tilelang_....json
```

## 最近验收观察

- 硬件 / commit / 环境：...
- 关键结果：...
- 不适用或风险：...
````
