---
title: CLI 指南
icon: material/console-line
---

# CLI 指南

## 当前真实存在的命令

当前分支里真实可用的 CLI 只有三个：

| 命令            | 作用                     | 最常见输入              | 最常见输出                                                  |
| --------------- | ------------------------ | ----------------------- | ----------------------------------------------------------- |
| `taac-train`    | 训练单个实验             | `--experiment`          | `best.pt`、`summary.json` 等训练产物                        |
| `taac-evaluate` | 评估单个或多个实验       | `single` / `batch`      | `evaluation.json` 或直接终端输出                            |
| `taac-search`   | 基于 Optuna 搜索实验配置 | `--experiment --trials` | `study_summary.json`、`best_experiment.json`、`trial_xxxx/` |

!!! warning "旧命令不在当前分支中"
    旧版文档里出现过的 `taac2026/experiments` 注册目录、`taac-visualize`、`taac-feature-*`、`taac-truncation-sweep` 等命令，在当前分支里都没有对应实现，不应继续写进新文档。

## `taac-train`

最小用法：

```bash
uv run taac-train --experiment config/gen/baseline
```

当前参数非常少：

| 参数           | 说明                       |
| -------------- | -------------------------- |
| `--experiment` | 实验包目录路径或模块路径。 |
| `--run-dir`    | 覆盖默认输出目录。         |
| `--compile`    | 启用 `torch.compile`。     |
| `--compile-backend` | 覆盖 `torch.compile` backend。 |
| `--compile-mode` | 覆盖 `torch.compile` mode。 |
| `--amp`        | 启用 AMP autocast。        |
| `--amp-dtype`  | AMP dtype，当前只支持 `float16` / `bfloat16`。 |

如果只想把输出落到单独目录：

```bash
uv run taac-train --experiment config/gen/oo --run-dir outputs/smoke/oo_manual
```

如果想显式打开运行时优化：

```bash
uv run taac-train \
    --experiment config/gen/baseline \
    --compile \
    --compile-backend inductor \
    --amp \
    --amp-dtype bfloat16
```

!!! info "运行时优化默认关闭"
    当前 `torch.compile` 和 AMP 都是显式 opt-in，默认不会影响现有实验包行为。

!!! info "AMP 的当前边界"
    CUDA 设备上可以使用 `float16` 或 `bfloat16`。CPU 路径当前只会实际启用 `bfloat16` autocast；如果在 CPU 上请求 `float16`，运行时会记录降级原因并继续按普通精度执行。

## `taac-evaluate`

评估入口分成 `single` 和 `batch` 两个子命令。

### `single`

默认会读取实验包 `train.output_dir` 下的 `best.pt`：

```bash
uv run taac-evaluate single --experiment config/gen/baseline
uv run taac-evaluate single --experiment config/gen/oo --run-dir outputs/smoke/oo
```

如果要用和训练一致的运行时优化路径评估 checkpoint：

```bash
uv run taac-evaluate single \
    --experiment config/gen/baseline \
    --compile \
    --amp \
    --amp-dtype bfloat16
```

也可以显式指定 checkpoint 与输出文件：

```bash
uv run taac-evaluate single \
    --experiment config/gen/interformer \
    --checkpoint outputs/smoke/interformer/best.pt \
    --output-path outputs/smoke/interformer/evaluation.json
```

### `batch`

批量评估当前支持：

```bash
uv run taac-evaluate batch --experiment-paths \
    config/gen/baseline \
    config/gen/grok \
    config/gen/ctr_baseline \
    config/gen/deepcontextnet \
    config/gen/interformer \
    config/gen/onetrans \
    config/gen/hyformer \
    config/gen/unirec \
    config/gen/uniscaleformer \
    config/gen/oo
```

`batch` 也支持和 `single` 相同的 `--compile`、`--compile-backend`、`--compile-mode`、`--amp`、`--amp-dtype` 开关；这些参数会统一应用到该批次里的每个实验。

!!! info "当前行为边界"
    `single` 模式只评估一个实验，不会做任何自动并行派发。`batch` 模式也不会自动忽略错误；如果某个实验缺少 `best.pt`，或者 checkpoint 与当前模型定义不兼容，命令会直接失败。

## `taac-search`

搜索入口基于 `optuna`，默认会做通用结构 + 优化器超参数搜索，并施加两条硬约束：

1. 参数量上限：`3 GiB`
2. 基于验证集 latency probe 估算的端到端总推理时长上限：`180 秒`

最小用法：

```bash
uv run taac-search --experiment config/gen/baseline --trials 20
```

常见参数：

| 参数                   | 说明                                                                |
| ---------------------- | ------------------------------------------------------------------- |
| `--study-dir`          | 保存 Optuna study 产物的目录。                                      |
| `--trials`             | 试验次数。                                                          |
| `--timeout-seconds`    | 可选的总墙钟时间限制。                                              |
| `--metric-name`        | `summary.json` 里的目标指标，例如 `best_val_auc` 或 `metrics.auc`。 |
| `--direction`          | `maximize` 或 `minimize`。                                          |
| `--scheduler`          | `auto` 或 `sequential`。                                            |
| `--gpu-indices`        | 自动调度时限制物理卡范围。                                          |
| `--min-free-memory-gb` | 自动模式下新增 worker 所需的最小空闲显存。                          |
| `--max-jobs-per-gpu`   | 单卡并发 worker 上限。                                              |
| `--json`               | 打印完整 JSON 报告，而不是紧凑摘要。                                |

更完整的例子：

```bash
uv run taac-search \
    --experiment config/gen/interformer \
    --study-dir outputs/search/interformer \
    --trials 40 \
    --scheduler auto \
    --gpu-indices 0,1,2,3 \
    --min-free-memory-gb 20 \
    --max-jobs-per-gpu 2 \
    --max-parameter-gb 2.5 \
    --max-end-to-end-inference-seconds 120
```

默认在 GPU 主机上会按当前可见设备空闲显存自动并行派发 trial；如果探测不到 GPU，会自动回退到顺序执行。默认在交互式终端下会用 `rich` 显示 trial 进度条；`--json` 模式会关闭进度条，避免污染机器可读输出。

如果你想跑原来的本地 grok 方案，只需要把 `--experiment config/gen/baseline` 替换成 `--experiment config/gen/grok`。

## 搜索目录里会生成什么

```text
study_summary.json
best_experiment.json
trial_0000/
trial_0001/
...
```

其中每个 `trial_xxxx/` 仍沿用常规训练产物格式，包含 `best.pt`、`summary.json`、`training_curves.json` 与 `training_curves.png`。
