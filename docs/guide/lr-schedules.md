---
icon: lucide/trending-up
---

# 学习率 Warmup 和 Scheduler

共享 PCVR 训练器已经支持 dense optimizer 的学习率 warmup 和调度器。它们可以在实验默认配置里写死，也可以通过训练 CLI 覆盖。

这套调度只作用在 dense optimizer 的参数组上。Sparse embedding optimizer 使用 `sparse_lr` / `sparse_weight_decay` 等单独配置，不会被这里的 `scheduler_type`、`warmup_steps` 或 `min_lr_ratio` 影响。

## 快速配置

本地训练时可以直接传 CLI 参数：

```bash
bash run.sh train \
  --experiment experiments/baseline_plus \
  --run-dir outputs/baseline_plus_cosine \
  --max_steps 10000 \
  --scheduler-type cosine \
  --warmup-steps 500 \
  --min-lr-ratio 0.1
```

训练 CLI 同时接受连字符和下划线参数名，所以 `--scheduler-type` / `--scheduler_type`、`--warmup-steps` / `--warmup_steps`、`--min-lr-ratio` / `--min_lr_ratio` 等价。

实验默认值写在实验包的 `TRAIN_DEFAULTS` 中：

```python
from taac2026.api import PCVROptimizerConfig, PCVRTrainConfig


TRAIN_DEFAULTS = PCVRTrainConfig(
    optimizer=PCVROptimizerConfig(
        lr=1e-4,
        max_steps=10000,
        dense_optimizer_type="adamw",
        scheduler_type="cosine",
        warmup_steps=500,
        min_lr_ratio=0.1,
    ),
)
```

CLI 参数会覆盖实验默认值。线上训练 bundle 也会走同一套训练 CLI，因此这些参数可以追加到 bundle 的 `run.sh` 后面。

## 参数含义

| 参数             | 含义                                                 |
| ---------------- | ---------------------------------------------------- |
| `lr`             | dense optimizer 的 base learning rate                |
| `scheduler_type` | 调度器类型，支持 `none`、`linear`、`cosine`          |
| `warmup_steps`   | warmup 的 optimizer step 数，必须大于等于 0          |
| `min_lr_ratio`   | 衰减后的最低学习率比例，范围是 0.0 到 1.0            |
| `max_steps`      | 衰减区间的终点；`linear` / `cosine` 衰减需要它大于 0 |

默认配置是 `scheduler_type="none"`、`warmup_steps=0`、`min_lr_ratio=0.0`。也就是说，默认不会做 warmup 或 decay。

## 调度规则

训练器会在每个 `_train_step()` 开始前设置 dense optimizer 学习率。第一个 optimizer step 的 step 编号是 1。

Warmup 规则：

```text
step <= warmup_steps 时：lr = base_lr * step / warmup_steps
```

Warmup 结束后，如果 `scheduler_type="none"` 或 `max_steps <= 0`，学习率回到并保持 `base_lr`。

当 `scheduler_type` 是 `linear` 或 `cosine`，并且 `max_steps > 0` 时，衰减进度按下面的区间计算：

```text
decay_steps = max(1, max_steps - warmup_steps)
decay_progress = clamp((step - warmup_steps) / decay_steps, 0.0, 1.0)
```

`linear` 会从 `base_lr` 线性衰减到 `base_lr * min_lr_ratio`。`cosine` 会按半周期 cosine 曲线衰减到同一个最低比例。

## 常用组合

只做 warmup，不做后续衰减：

```bash
bash run.sh train \
  --experiment experiments/baseline \
  --run-dir outputs/baseline_warmup \
  --scheduler-type none \
  --warmup-steps 500
```

做 warmup + cosine decay：

```bash
bash run.sh train \
  --experiment experiments/symbiosis \
  --run-dir outputs/symbiosis_cosine \
  --max_steps 20000 \
  --scheduler-type cosine \
  --warmup-steps 2000 \
  --min-lr-ratio 0.1
```

做 warmup + linear decay：

```bash
bash run.sh train \
  --experiment experiments/baseline_plus \
  --run-dir outputs/baseline_plus_linear \
  --max_steps 10000 \
  --scheduler-type linear \
  --warmup-steps 500 \
  --min-lr-ratio 0.2
```

如果不传 `--max_steps`，当前默认值通常是 0。此时训练总步数会按数据 sweep 推导，但 scheduler 的 decay 会被关闭；如果你期望真的看到 `linear` 或 `cosine` 衰减，需要显式设置 `--max_steps`。

## 现有实验默认值

当前实验包的默认调度大致如下：

| 实验            | 默认调度                                                           |
| --------------- | ------------------------------------------------------------------ |
| `baseline`      | `scheduler_type="none"`，`warmup_steps=0`                          |
| `baseline_plus` | `scheduler_type="none"`，`warmup_steps=0`                          |
| `interformer`   | `scheduler_type="none"`，`warmup_steps=0`                          |
| `onetrans`      | `scheduler_type="none"`，`warmup_steps=0`                          |
| `symbiosis`     | `scheduler_type="none"`，`warmup_steps=0`                          |

## 源码入口

- 配置对象和校验：`src/taac2026/domain/config.py`
- 训练 CLI 参数：`src/taac2026/application/training/args.py`
- workflow 传参：`src/taac2026/application/training/workflow.py`
- 学习率公式：`src/taac2026/infrastructure/optimization/schedules.py`
- trainer 应用学习率：`src/taac2026/infrastructure/runtime/checkpoint_io.py`
- runtime 覆盖测试：`tests/unit/infrastructure/runtime/test_trainer.py`
- CLI 覆盖测试：`tests/unit/application/training/test_cli.py`

训练时如果有 writer，当前 dense 学习率会写到 `LR/dense`。排查调度是否生效时，优先看训练日志里的 trainer 初始化参数，以及 TensorBoard 里的 `LR/dense` 曲线。

## 常见问题

| 现象                                      | 原因和处理                                                                                                           |
| ----------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| 传了 `--scheduler-type cosine` 但没有衰减 | 确认是否设置了 `--max_steps`；`max_steps <= 0` 时只会 warmup，不会 decay                                             |
| warmup 覆盖了整个训练                     | `warmup_steps` 大于或接近总 optimizer step 数；调小 warmup 或调大 `max_steps`                                        |
| sparse embedding 学习率没变化             | 这里的 scheduler 只管 dense optimizer；调整 sparse 参数要用 `sparse_lr` 等配置                                       |
| 参数被拒绝                                | `scheduler_type` 只能是 `none`、`linear`、`cosine`，`warmup_steps` 不能为负数，`min_lr_ratio` 必须在 0.0 到 1.0 之间 |
