---
icon: lucide/refresh-cw
---

# 梯度检查点

梯度检查点用于降低训练时的 activation 显存占用。它会在前向阶段少存一部分中间激活，并在反向传播时重新计算这些片段；代价是训练 step 会变慢一些。

这里说的梯度检查点是 activation recomputation，不是训练产物里的 `model.safetensors`、`schema.json` 或 `train_config.json` checkpoint。

## 快速配置

本地训练时直接打开 CLI 开关：

```bash
bash run.sh train \
  --experiment experiments/baseline_plus \
  --run-dir outputs/baseline_plus_gc \
  --gradient-checkpointing
```

这个参数由 `argparse.BooleanOptionalAction` 处理，因此也支持显式关闭：

```bash
bash run.sh train \
  --experiment experiments/symbiosis \
  --run-dir outputs/symbiosis_no_gc \
  --no-gradient-checkpointing
```

实验默认值写在实验包的 `TRAIN_DEFAULTS` 中：

```python
from taac2026.api import PCVRModelConfig, PCVRTrainConfig


TRAIN_DEFAULTS = PCVRTrainConfig(
    model=PCVRModelConfig(
        gradient_checkpointing=True,
    ),
)
```

CLI 参数会覆盖实验默认值。线上训练 bundle 也走同一套训练 CLI，所以可以把 `--gradient-checkpointing` 追加到 bundle 的 `run.sh` 后面。

## 行为边界

当前实现使用 `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`。共享 helper 是 `maybe_gradient_checkpoint()`：

```python
def maybe_gradient_checkpoint(function, *args, enabled: bool = False, **kwargs):
    if not enabled or not torch.is_grad_enabled():
        return function(*args, **kwargs)
    return checkpoint(function, *args, use_reentrant=False, **kwargs)
```

这意味着：

- 只有 `gradient_checkpointing=True` 且 PyTorch 正在记录梯度时才会触发。
- 评估和推理通常在 no-grad 路径中运行，即使 checkpoint sidecar 里记录了 `gradient_checkpointing=True`，也不会做反向重算。
- 它只覆盖模型里显式包了 `maybe_gradient_checkpoint()` 的 block，不会自动包住整个模型。

## 当前覆盖范围

现有 PCVR 模型包都接受 `gradient_checkpointing` 构造参数，默认值都是 `False`。

| 实验            | 当前 checkpoint 的模块                                          |
| --------------- | --------------------------------------------------------------- |
| `baseline`      | HyFormer block 循环                                             |
| `baseline_plus` | HyFormer block 循环                                             |
| `interformer`   | InterFormer block 循环                                          |
| `onetrans`      | OneTrans block 循环                                             |
| `symbiosis`     | user/item graph blocks、unified blocks、context exchange blocks |

Embedding lookup、输入构造、最终 classifier、loss 计算和 optimizer step 不在这套 helper 的覆盖范围内。

## 什么时候打开

优先在这些情况尝试打开：

- CUDA OOM 主要来自较长序列、较大 batch size、较宽 `d_model` 或更多 block。
- 你愿意用更长训练时间换取更低峰值显存。
- 模型主体是 Transformer / cross-attention / fusion block，activation 占比明显高于参数显存。

不一定有帮助的情况：

- OOM 主要来自 embedding table 参数或 optimizer state。
- batch 已经很小，瓶颈主要是数据加载或 CPU 侧处理。
- 正在做极短 smoke，例如 `--max_steps 1`，此时打开它只能验证通路，不能代表性能收益。

## 和其他运行时开关的关系

梯度检查点是模型配置字段，不属于 `RuntimeExecutionConfig`。它可以和 AMP、`torch.compile`、TileLang backend 同时打开，但改这些组合后应重新做 smoke 和单测。

常见组合：

```bash
bash run.sh train \
  --experiment experiments/baseline_plus \
  --run-dir outputs/baseline_plus_mem \
  --device cuda \
  --amp \
  --amp-dtype bfloat16 \
  --gradient-checkpointing
```

如果你同时打开 `--compile` 后遇到编译慢、图捕获失败或显存没有改善，先分别验证 `--compile` 和 `--gradient-checkpointing`，再组合运行。

## 新实验怎么接入

普通 PCVR 实验如果使用共享模型构造链路，只需要让模型类接收并保存 `gradient_checkpointing`：

```python
class MyModel(nn.Module):
    def __init__(self, *, gradient_checkpointing: bool = False, **kwargs) -> None:
        super().__init__()
        self.gradient_checkpointing = bool(gradient_checkpointing)
```

然后在需要重算的 block 位置包一层 helper：

```python
from taac2026.api import maybe_gradient_checkpoint


for block in self.blocks:
    tokens = maybe_gradient_checkpoint(
        block,
        tokens,
        padding_mask,
        enabled=self.gradient_checkpointing,
    )
```

不要把会修改全局状态、依赖不可重算 side effect、或返回非 Tensor 主体状态的逻辑直接塞进 checkpoint block。优先包纯前向计算的层或 block。

## 源码入口

- 配置字段：`src/taac2026/domain/config.py`
- 训练 CLI 参数：`src/taac2026/application/training/args.py`
- 模型构造传参：`src/taac2026/domain/model_contract.py`
- checkpoint helper：`src/taac2026/infrastructure/modeling/sequence.py`
- API 导出：`src/taac2026/api.py`
- 实验模型调用点：`experiments/*/model.py`
- CLI 覆盖测试：`tests/unit/application/training/test_cli.py`
- backward 覆盖测试：`tests/unit/experiments/test_packages.py`

## 常见问题

| 现象                    | 原因和处理                                                                                     |
| ----------------------- | ---------------------------------------------------------------------------------------------- |
| 打开后训练变慢          | 这是预期代价；反向传播会重新计算被 checkpoint 的 block                                         |
| 打开后显存下降不明显    | 当前模型只 checkpoint 显式包住的 block；embedding 参数、optimizer state 和未包住的激活不会减少 |
| 评估或推理看起来没变化  | no-grad 路径不会触发 activation checkpointing，这是正常行为                                    |
| 新实验传参失败          | 确认模型构造函数接收 `gradient_checkpointing`，或共享模型构造链路能把该字段转发进去            |
| 和 `--compile` 组合出错 | 先分别跑 `--gradient-checkpointing` 和 `--compile`，确认单独开关健康后再组合                   |
