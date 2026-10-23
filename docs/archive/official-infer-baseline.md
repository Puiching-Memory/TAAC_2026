---
icon: lucide/folder-input
---

# 官方推理 Baseline 源码快照

归档推理实现用于理解旧平台变量、模型重建与输出格式。当前推理包由共享 bootstrap 生成，不直接上传这套快照。

| 文件 | 职责 |
| --- | --- |
| [infer.py](files/official_infer_baseline/infer.py) | 读取环境、加载权重、写 predictions |
| [dataset.py](files/official_infer_baseline/dataset.py) | parquet 和 schema 解析 |
| [model.py](files/official_infer_baseline/model.py) | 自包含模型 |

## 历史行为与当前区别

旧入口读取 `MODEL_OUTPUT_PATH`、`EVAL_DATA_PATH`、`EVAL_RESULT_PATH`。它优先使用 checkpoint schema，缺失时尝试数据目录；旧 train_config 缺失时可使用脚本内置默认值，NS 分组另有可选文件。

当前框架要求完整 `schema.json` 和 `train_config.json` sidecar，显式 schema 无效时立即失败，配置缺字段也会失败。不要把归档中的默认补齐当作当前兼容策略。

两种实现都使用 `predictions.json` 的 `predictions` 映射表达用户概率；相同输出格式不代表模型构造与 checkpoint 可互换。

训练侧对照见[训练快照](official-train-baseline.md)，当前路径见[架构](../architecture.md)和[Bundle](../guide/online-training-bundle.md)。
