---
icon: lucide/gauge
---

# Benchmark：口径、结果与限制

本分区保存测量方法、已有历史记录和优化候选。历史表格未经本次 GPU 复测，不能称作当前代码性能；新的结论应从[性能指南](../guide/performance-benchmarks.md)重新生成 JSON。

## 按问题选择

| 问题 | 页面 |
| --- | --- |
| 模型输出是否稳定、是否只靠特定样本 | [PCVR 诊断](pcvr-diagnostics.md) |
| 缓存是否值得占用内存 | [Cache Policies](cache-policies.md) |
| normalization kernel 是否有效 | [RMSNorm](rms-norm.md)、[LayerNorm](layer-norm.md) |
| attention 加速是否覆盖真实形状 | [Flash Attention](flash-attention.md) |
| 多值 sparse lookup/pooling 是否受益 | [Embedding Bag Mean](embedding-bag-mean.md) |
| 下一步优化如何选题 | [算子路线图](operator-roadmap.md) |

## 已声明的 Backend

| 算子 | torch | TileLang | Triton | cuEmbed |
| --- | --- | --- | --- | --- |
| RMSNorm | reference | 有实现 | 有实现 | — |
| LayerNorm | reference | 未接入 | 有实现 | — |
| Flash Attention | SDPA | 有实现 | 未接入 | — |
| Embedding Bag Mean | reference | 有实现 | 有实现 | 仅 forward |

这张表描述源码能力，不代表每种 dtype/shape/mask 都可运行，也不表示模型已默认启用该 backend。

## 读 JSON 的顺序

先看 `status` 与 `resolved_backend`，再检查 `max_abs_error`、`max_rel_error`，最后比较 `step_time_ms_mean` 与离散程度。`compile_sec` 是首次编译成本，独立于稳态时间。

保存命令时加 `--json`，创建输出目录。记录 commit、设备、版本和完整参数；区分 forward、forward+backward、Python launch、数据读取和模型端到端。不要将 unsupported、缺失值或失败结果计为零耗时。
