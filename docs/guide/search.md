---
icon: lucide/search
---

# 搜索请求

## 当前状态

当前仓库保留了 `taac-search` 入口，但它还不是完整的 Optuna 调度器。这个命令现在用于记录一次可复现的搜索请求：加载实验包、创建 study 目录，并写出 `study_request.json`。真正的 trial 派发、GPU 调度和搜索空间执行还没有在当前实现中落地。

## 基本用法

```bash
uv run taac-search \
    --experiment config/baseline \
    --study-dir outputs/search/baseline \
    --trials 20 \
    --metric-name metrics.auc \
    --direction maximize \
    --seed 42 \
    --json
```

可用参数以 `src/taac2026/application/search/cli.py` 为准：

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--experiment` | `config/baseline` | 要记录搜索请求的实验包 |
| `--study-dir` | `outputs/search/baseline` | 输出目录 |
| `--trials` | `1` | 计划 trial 数 |
| `--timeout-seconds` | `None` | 计划超时时间 |
| `--metric-name` | `metrics.auc` | 目标指标名 |
| `--direction` | `maximize` | 指标方向，可选 `maximize` 或 `minimize` |
| `--seed` | `42` | 请求种子 |
| `--json` | 关闭 | 以缩进 JSON 打印结果 |

## 输出

命令会写出：

```text
outputs/search/<name>/
└── study_request.json
```

示例 payload：

```json
{
  "experiment_name": "pcvr_baseline",
  "experiment": "config/baseline",
  "trials": 20,
  "timeout_seconds": null,
  "metric_name": "metrics.auc",
  "direction": "maximize",
  "seed": 42,
  "status": "recorded"
}
```

## 和训练入口的关系

`taac-search` 不会自动调用训练 CLI，也不会读取数据集路径。要实际训练某个实验包，继续使用统一入口：

```bash
bash run.sh train --experiment config/baseline \
    --dataset-path /path/to/parquet_or_dataset_dir \
    --schema-path /path/to/schema.json
```

如果后续恢复完整搜索器，文档需要同步补齐调度策略、trial 输出目录、剪枝规则和回归测试。当前页面只描述已经在代码中存在的请求记录能力。
