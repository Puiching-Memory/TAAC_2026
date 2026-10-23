---
icon: lucide/folder-archive
---

# 官方训练 Baseline 源码快照

这是归档的自包含训练实现，用来对照当时的数据读取、NS 分组和 checkpoint 写入方式。当前框架不会自动执行这些文件。

## 文件入口

| 文件 | 职责 |
| --- | --- |
| [run.sh](files/official_train_baseline/run.sh) | 启动参数 |
| [train.py](files/official_train_baseline/train.py) | CLI 与环境变量 |
| [dataset.py](files/official_train_baseline/dataset.py) | raw parquet 与 schema |
| [model.py](files/official_train_baseline/model.py) | PCVRHyFormer |
| [trainer.py](files/official_train_baseline/trainer.py) | 更新、验证、保存和早停 |
| [utils.py](files/official_train_baseline/utils.py) | 日志、seed 与 loss helper |
| [ns_groups.json](files/official_train_baseline/ns_groups.json) | 参考字段分组 |

## 对照时关注什么

归档 shell 默认 RankMixer，用户/物品 token 为 5/2、query 为 2、workers 为 8。训练读取 `TRAIN_DATA_PATH`、`TRAIN_CKPT_PATH` 等变量，保存 schema、配置及可选 NS 分组。

当前共享实现把 NS 分组放入 `model.ns`，使用 typed sidecar 与严格重建。旧配置文件和参数不能直接搬进当前 CLI。需要迁移时同时检查输入、模型与权重形状，而不是只替换启动命令。

对应推理行为见[推理快照](official-infer-baseline.md)，当前模型见[Baseline](../experiments/baseline.md)，当前上传方式见[Bundle](../guide/online-training-bundle.md)。
