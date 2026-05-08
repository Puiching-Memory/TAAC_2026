---
icon: lucide/folder-open
---

# 实验包总览

实验包是这个仓库里“可以被训练、评估、推理或打包”的最小单位。框架层负责通用运行时，实验包只回答三个问题：叫什么、默认怎么训、用哪个模型或维护任务。

如果你只是想跑起来，优先从 Baseline 开始；如果你想改模型，再从最接近的实验包复制。

## 怎么选

| 目标                      | 从这里开始                                  | 说明                                                                  |
| ------------------------- | ------------------------------------------- | --------------------------------------------------------------------- |
| 需要一个干净参照          | [Baseline](baseline.md)                     | HyFormer 基线，默认不启用数据增强和 TileLang backend                  |
| 想看当前增强版基线        | [Baseline+](baseline-plus.md)               | 同一类 HyFormer 思路，默认打开 OPT cache、数据增强和 TileLang backend |
| 想做用户-物品交互结构     | [InterFormer](interformer.md)               | Group tokenizer，强调用户与物品分支之间的交互                         |
| 想做统一 Transformer 结构 | [OneTrans](onetrans.md)                     | RankMixer tokenizer，把用户、物品和序列放进更统一的编码路径           |
| 想试原生统一 token 流     | [UniTok](unitok.md)                         | Field-level token、dense packet 和行为事件进入同一个 backbone          |
| 想试 UniRec 融合结构      | [UniRec](unirec.md)                         | Hybrid SiLU attention、MoT、target-aware interest 和 BlockAttnRes      |
| 想试复杂融合方案          | [Symbiosis](symbiosis.md)                   | 带额外 CLI 参数和自定义 hooks，适合做消融                             |
| 想知道线上机器是什么样    | [Host Device Info](host-device-info.md)     | 不需要数据集，采集平台环境快照                                        |
| 想在线上跑数据概览        | [Online Dataset EDA](online-dataset-eda.md) | 需要数据集，流式扫描 parquet 并打印文本报告                           |

## 运行方式

模型实验本地训练：

```bash
bash run.sh train \
  --experiment experiments/baseline \
  --run-dir outputs/baseline_smoke
```

维护实验也走同一个入口：

```bash
bash run.sh train --experiment experiments/host_device_info
```

`run.sh` 只处理 `train`、`val` / `eval` 和 `infer`。线上上传物由独立打包命令生成：

```bash
uv run taac-package-train --experiment experiments/baseline --output-dir outputs/bundles/baseline_training
uv run taac-package-infer --experiment experiments/baseline --output-dir outputs/bundles/baseline_inference
```

维护类实验只支持训练 bundle；它们没有推理模型接口。

## 包长什么样

普通 PCVR 模型实验通常只有两个必需文件：

```text
experiments/<name>/
├── __init__.py
└── model.py
```

`__init__.py` 导出 `EXPERIMENT`，放实验名、模型类名、默认训练配置和必要 hooks。`model.py` 放当前实验自己的模型类。确实属于共享运行时的逻辑应放到 `src/taac2026/`，不要塞回实验包。

维护工具包更轻：

```text
experiments/<tool>/
├── __init__.py
└── runner.py
```

它们使用 `ExperimentSpec`，不需要模型类、checkpoint sidecar 或推理 hooks。

## 改实验时先看哪里

- 实验入口：`experiments/<name>/__init__.py`
- 模型实现：`experiments/<name>/model.py`
- 实验发现与装载：`src/taac2026/application/experiments/`
- 模型输入契约：`src/taac2026/domain/model_contract.py`
- 新增实验流程：[新增实验包](../guide/contributing.md)

不要从 `docs/archive/files/...` 推断当前契约；那里是历史快照。
