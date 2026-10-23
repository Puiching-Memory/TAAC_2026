---
icon: lucide/chart-line
---

# 学习率调度与早停

调度只作用于 dense optimizer，sparse 学习率独立配置。验证间隔与 early stopping 按 optimizer step 计数，不按 epoch 或验证次数计数。

## 配置项

| 参数 | 含义 |
| --- | --- |
| `--optimizer.lr` | dense base learning rate |
| `--optimizer.scheduler_type` | `none`、`linear`、`cosine` |
| `--optimizer.warmup_steps` | 从小学习率升至 base 的步数 |
| `--optimizer.min_lr_ratio` | decay 终点相对 base 的比例，范围 `[0,1]` |
| `--optimizer.max_steps` | 总训练截断；正值时提供 decay 终点 |
| `--data.eval_every_n_steps` | 验证间隔 |
| `--optimizer.patience_steps` | 无改善容忍步数 |
| `--validation.early_stopping_metric` | `auc` 或 `logloss` |

## 实际公式

第一个 optimizer step 为 1。warmup 阶段的乘数是 `step / warmup_steps`。结束后，`none` 或 `max_steps<=0` 保持 base lr；其余情况计算：

```text
p = clamp((step - warmup_steps) / max(1, max_steps - warmup_steps), 0, 1)
linear: multiplier = min_lr_ratio + (1-min_lr_ratio) * (1-p)
cosine: multiplier = min_lr_ratio + (1-min_lr_ratio) * (1+cos(pi*p))/2
```

公式位于 `infrastructure/optimization/schedules.py`。warmup 覆盖整个训练时不会进入后续衰减；不能只设置 `scheduler_type` 就假定能看到完整曲线。

## 示例

完成[数据准备](../getting-started.md)后，给 Baseline 设置 500 步 warmup 与 10000 步 cosine：

```bash
uv run --with torch==2.13.0 taac-train \
  --experiment experiments/baseline \
  --dataset-path data/sample_1000_raw/demo_1000.parquet \
  --schema-path data/sample_1000_raw/schema.json \
  --run-dir outputs/baseline_cosine \
  --optimizer.max_steps 10000 --optimizer.scheduler_type cosine \
  --optimizer.warmup_steps 500 --optimizer.min_lr_ratio 0.1
```

这是调度配置示例，demo 重复采样 10000 步不能用作泛化结论。真实实验应换成固定训练集和独立验证集。

## 当前实验默认值

所有七个模型实验显式设置 `max_steps=100000`，不能把基础配置类型的默认 `0` 当作实验默认值。QueryFormer 为 cosine/无 warmup；Symbiosis 为 none/100 步 warmup；其余模型为 none/无 warmup。

多数实验每 5000 步验证，patience 25000；Symbiosis 为 500/2500。更新默认值时同步实验页与实际配置。

## 确认是否生效

检查训练初始化日志中的实际配置，以及 TensorBoard 的 `LR/dense`。若 sparse lr 未变，属于预期范围；若过早停止，检查验证间隔与 patience 的步数关系，并核对监控指标是越大越好还是越小越好。

测试入口：`tests/unit/infrastructure/runtime/test_trainer.py`、`tests/unit/application/training/test_cli.py`。不要只比较配置文本，还需验证训练步上的实际学习率。
