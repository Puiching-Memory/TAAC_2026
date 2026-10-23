---
icon: lucide/refresh-cw
---

# 梯度检查点：用重计算降低激活内存

梯度检查点在反向时重算部分前向计算，减少保存 activation 的内存。它与磁盘中的权重 checkpoint 是不同机制，也不会减少 embedding 参数与 optimizer state 的体积。

## 启用和关闭

准备数据后，在训练命令追加：

```text
--model.gradient_checkpointing
```

关闭用 `--model.no_gradient_checkpointing`。实验默认值在 `TRAIN_DEFAULTS.model`，七个 PCVR 模型当前均默认关闭。AMP、compile 位于独立的 `runtime` 配置中。

```bash
bash run.sh train \
  --experiment experiments/baseline \
  --dataset-path data/sample_1000_raw/demo_1000.parquet \
  --schema-path data/sample_1000_raw/schema.json \
  --run-dir outputs/baseline_gc \
  --optimizer.device cuda --runtime.amp \
  --model.gradient_checkpointing
```

GPU 环境准备见[快速开始](../getting-started.md)。

## 覆盖范围

共享 helper `maybe_gradient_checkpoint()` 仅在 `enabled=True` 且 `torch.is_grad_enabled()` 时调用 `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`。模型必须显式包住相应 block，它不会自动重算整个网络。

| 实验 | 重计算边界 |
| --- | --- |
| Baseline / DualQ | HyFormer blocks |
| InterFormer | InterFormer blocks |
| OneTrans / TokenFormer | 各自主干 blocks |
| Symbiosis | unified interaction blocks |
| QueryFormer | CoTransformer blocks |

embedding lookup、输入构造、最终分类头、loss 与 optimizer step 不在统一 block 边界中。评估和推理使用无梯度路径，sidecar 即使记录开启也不会执行反向重算。

## 新模型如何接入

当前统一构造函数是 `model_type(schema, config)`，其中 `config` 是模型子配置（`PCVRModelConfig` 或实验扩展类型）。从 `config.gradient_checkpointing` 读取开关，不要新增不匹配的独立构造参数：

```python
from taac2026.api import maybe_gradient_checkpoint

# 在模型构造函数中保存配置
self.gradient_checkpointing = config.gradient_checkpointing

# 在 forward 中包住可重算的 block
tokens = maybe_gradient_checkpoint(
    block, tokens, padding_mask,
    enabled=self.gradient_checkpointing,
)
```

示例是模型方法内的接入片段。边界内应保持可重算，不依赖一次性 IO 或不可重复的全局副作用。

## 如何判断是否值得

固定模型、batch、序列长度、精度与设备，分别测关闭/开启后的峰值显存和稳定 step time，并验证 loss 与梯度数值。若瓶颈来自参数表或数据加载，内存收益可能很小；重算通常增加计算成本。

AMP、compile、TileLang 与重计算组合需要分别及组合验证。一次一步的 CPU smoke 只能检查 backward 路径，不能证明 GPU 显存收益。

源码 helper 位于 `infrastructure/modeling/sequence.py`，模型调用点在 `experiments/*/model.py`。测试选择见[测试指南](testing.md)。
