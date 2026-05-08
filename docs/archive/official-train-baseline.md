---
icon: lucide/folder-archive
---

# 官方训练 Baseline 快照

这个页面归档了比赛参考训练 baseline 的自包含源码快照，便于后续做结构对照、契约追溯和资料保存。当前仓库长期维护的训练入口仍然是 experiments/baseline 及共享框架代码；这里的内容仅作为历史参考，不作为当前运行时依赖。

对应的推理源码快照见 [官方推理 Baseline 快照](official-infer-baseline.md)。

## 源码浏览

| 文件                                                           | 作用                                         |
| -------------------------------------------------------------- | -------------------------------------------- |
| [run.sh](files/official_train_baseline/run.sh)                 | Shell 训练入口，内置默认启动参数             |
| [train.py](files/official_train_baseline/train.py)             | 训练 CLI，解析环境变量和模型参数             |
| [dataset.py](files/official_train_baseline/dataset.py)         | 原始 Parquet 数据集读取与 schema 解析        |
| [model.py](files/official_train_baseline/model.py)             | 自包含 PCVRHyFormer 模型实现                 |
| [trainer.py](files/official_train_baseline/trainer.py)         | 训练循环、验证、checkpoint 与 early stopping |
| [utils.py](files/official_train_baseline/utils.py)             | 日志、随机种子、focal loss、early stopping   |
| [ns_groups.json](files/official_train_baseline/ns_groups.json) | Group tokenizer 的参考分组配置               |

## 关键契约

- run.sh 默认启用 rankmixer NS tokenizer，并固定 user_ns_tokens=5、item_ns_tokens=2、num_queries=2、emb_skip_threshold=1000000、num_workers=8。
- run.sh 还保留了一套 group tokenizer 的备选配置，依赖 ns_groups.json，且用 num_queries=1 满足 d_model 与 token 总数的整除关系。
- train.py 以环境变量为主读取 TRAIN_DATA_PATH、TRAIN_CKPT_PATH、TRAIN_LOG_PATH、TRAIN_TF_EVENTS_PATH。
- dataset.py 明确以多列 raw parquet 加 schema.json 的形式构建数据集，而不是依赖预打包特征张量。
- trainer.py 会在 checkpoint 目录旁写入 schema.json、可选的 ns_groups.json 以及 train_config.json，使训练产物更接近自描述状态。

## 与当前仓库的关系

- 当前仓库的长期维护实现位于 experiments/baseline 和 src/taac2026。
- 当前打包流程以 taac-package-train 和 taac-package-infer 生成 bundle，不直接使用这份历史源码快照。
- 这份归档更适合用来回答“官方参考实现当时怎么组织训练入口、数据集、分组配置和 checkpoint sidecar”这类问题。