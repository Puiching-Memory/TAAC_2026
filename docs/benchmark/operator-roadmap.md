---
icon: lucide/cpu
---

# 算子优化路线图

优化候选由真实模型 profile 决定。此页记录已实现范围和待验证方向，优先级是工程建议，不是性能收益承诺。

## 现有能力

| 算子 | 已有 backend | 尚缺范围 |
| --- | --- | --- |
| RMSNorm | torch / TileLang / Triton | 目标模型的融合与端到端收益 |
| LayerNorm | torch / Triton | TileLang 与高价值融合 |
| Flash Attention | torch / TileLang | 独立 Triton backend |
| Embedding Bag Mean | torch / TileLang / Triton / cuEmbed forward | 更完整 sparse/backward 负载 |
| Gated Delta Rule | TileLang kernel 族 | 独立 Triton 对照 |

具体 dtype、mask、autograd 与已知限制看对应[算子页](index.md)。共享配置有 backend 字段不表示所有模型都调用它，接入前先追踪实际调用点。

## 候选顺序

1. 先 profile 完整模型：数据等待、embedding、attention、normalization、optimizer 各占多少时间。
2. 优先消除重复转换和多次 launch，评估 LayerNorm+Linear、residual+norm、SwiGLU 等融合边界。
3. 对已有高占比算子补 backend，对低占比算子不因 microbenchmark 倍数大就默认替换。
4. GDR、dropout 与 sparse backward 涉及复杂状态、随机数和累加，先确立可测试数值契约。

## 每个候选的完成条件

| 层面 | 必需证据 |
| --- | --- |
| 接口 | dtype、shape、layout、mask、device 与不支持情况明确 |
| 正确性 | torch reference，forward/backward、空输入与边界值 |
| 状态 | 随机性、compile、重计算和重复执行行为 |
| 性能 | 代表性形状、多次重复、JIT 与 launch 成本 |
| 集成 | 实际模型消费该路径，比较完整 step/inference 成本 |
| 交付 | CLI、配置、测试、来源与文档同步 |

源码 TODO 只能说明开发意图，不等于已实现能力。新 backend 需更新类型枚举与生产者/消费者，失败时明确报错，不把无效 accelerator 请求伪装成成功。

验证方法见[测试](../guide/testing.md)与[性能测量](../guide/performance-benchmarks.md)。
