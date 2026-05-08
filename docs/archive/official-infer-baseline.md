---
icon: lucide/folder-input
---

# 官方推理 Baseline 快照

这个页面归档了比赛参考推理 baseline 的自包含源码快照，便于后续做推理入口、checkpoint sidecar 契约和 predictions 输出格式的对照。当前仓库长期维护的推理入口仍然位于共享框架代码中；这里的内容仅作为历史参考，不作为当前运行时依赖。

## 源码浏览

| 文件                                                   | 作用                                                                          |
| ------------------------------------------------------ | ----------------------------------------------------------------------------- |
| [infer.py](files/official_infer_baseline/infer.py)     | 推理入口，负责读取环境变量、重建模型、加载 checkpoint 并输出 predictions.json |
| [dataset.py](files/official_infer_baseline/dataset.py) | 原始 Parquet 数据集读取与 schema 解析                                         |
| [model.py](files/official_infer_baseline/model.py)     | 自包含 PCVRHyFormer 模型实现                                                  |

## 关键契约

- infer.py 通过 `schema.json`、可选的 `ns_groups.json` 和 `train_config.json` 重建模型结构；如果 `train_config.json` 缺失，会回退到脚本内置默认值。
- infer.py 读取 `MODEL_OUTPUT_PATH`、`EVAL_DATA_PATH` 和 `EVAL_RESULT_PATH` 三个环境变量。
- schema 优先从 checkpoint 目录读取；如果 checkpoint 目录缺少 `schema.json`，才退回到评测数据目录中的 `schema.json`。
- 权重加载使用严格模式；如果重建出来的模型结构与 checkpoint 不一致，会直接失败。
- 输出文件固定写到 `EVAL_RESULT_PATH/predictions.json`，结构为 `{"predictions": {user_id: probability}}`。

## 与当前仓库的关系

- 当前仓库的长期维护实现位于 experiments/baseline 和 src/taac2026。
- 当前打包流程以 taac-package-infer 生成推理 bundle，不直接使用这份历史源码快照。
- 这份归档更适合用来回答“官方参考推理实现如何解析 sidecar、构建模型和写出 predictions.json”这类问题。