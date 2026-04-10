---
title: 项目结构
icon: material/file-tree-outline
---

# 项目结构

## 当前分支的实现边界

当前分支的实现边界比较明确：

1. 共享底座位于 `src/taac2026`。
2. 目录式独立实验包位于 `config/gen`。
3. 当前真实可用的 CLI 是 `taac-train`、`taac-evaluate` 与 `taac-search`。
4. 当前统一回归入口是 `pytest tests -q`。

## 目录职责

| 路径           | 作用                                                                                        |
| -------------- | ------------------------------------------------------------------------------------------- |
| `src/taac2026` | 共享训练/评估/搜索底座，包括实验加载、指标、运行时与调度逻辑，以及可被 CLI / 测试复用的实现模块。 |
| `config/gen`   | 目录式实验包代码，每个实验自己管理数据、模型与优化器装配；配套说明统一放在 `docs/packages`。 |
| `tests`        | 回归测试入口，按 `unit` / `integration` / `smoke` / `fault` 分层覆盖实验包加载、数据管线、profiling、search worker、train/evaluate 闭环与 checkpoint 兼容性。 |
| `tools`        | 仓库维护脚本与薄入口，通常负责 repo 默认路径、批处理入口或一次性辅助任务，不承载核心业务逻辑。 |
| `figures`      | 仓库根目录展示图表的产物。                                                                  |
| `docs`         | 文档站内容本体，包括 `docs/stages`、`docs/packages`、`docs/papers` 等统一文档入口。         |

## `src` 和 `tools` 的判断标准

判断时优先看“是否需要被复用”，而不是看“有没有命令行参数”：

1. 会被 `config/gen`、测试、正式 CLI 或包内其他模块复用的能力，应该放进 `src/taac2026`。
2. 即使某个文件带 `argparse`，只要它是正式入口或内部运行入口，例如 `train.py`、`evaluate.py`、`search.py`、`search_worker.py`，也仍然属于 `src`。
3. 只服务仓库维护、默认绑定当前 repo 路径、未来不打算作为框架 API 复用的脚本，才放 `tools`。
4. 如果 `tools` 脚本里逐渐长出了数据解析、渲染、报告装配这类通用逻辑，应优先下沉到 `src`，脚本本身只保留薄入口。

## `config/gen` 的目录契约

每个可执行实验包都应该满足下面的契约：

1. `__init__.py` 导出 `EXPERIMENT`。
2. `data.py` 私有实现数据管线。
3. `model.py` 私有实现模型主体。
4. `utils.py` 私有实现损失与优化器装配。
5. `docs/packages/<name>.md` 说明来源、映射方式、运行命令与当前验证状态。

当前已登记的可执行实验包现在是十个：

1. `baseline`
2. `grok`
3. `ctr_baseline`
4. `deepcontextnet`
5. `interformer`
6. `onetrans`
7. `hyformer`
8. `unirec`
9. `uniscaleformer`
10. `oo`

`symbiosis` 当前只是概念说明，不是可执行实验包。

## 输出产物契约

训练输出目录当前统一包含四类主要产物：

| 文件                   | 说明                                             |
| ---------------------- | ------------------------------------------------ |
| `best.pt`              | 当前最佳 epoch 的模型参数和指标。                |
| `summary.json`         | 关键指标、latency、profile、完整训练总算力估算。 |
| `training_curves.json` | 逐 epoch 曲线的结构化记录。                      |
| `training_curves.png`  | 训练过程中持续刷新的曲线图。                     |

## 当前未覆盖的能力

下面这些内容不在当前分支实现范围内：

1. 正式比赛线上提交流程。
2. 官方评测环境封装。
3. 可视化、EDA、聚类分析 CLI。
4. truncation sweep / feature engineering 专用脚本入口。

如果后续这些能力重新回到主分支，文档应基于实际代码补齐，而不是继续沿用旧分支说明。
