---
icon: lucide/flask-conical
---

# 实验包：选择与比较

七个模型实验共用训练、评估、推理和打包流程，两个维护实验用于诊断。实验页说明结构与默认配置；训练接口见[快速开始](../getting-started.md)。

## 选择模型

| 实验 | 主要问题 | 比较时特别控制 |
| --- | --- | --- |
| [Baseline](baseline.md) | HyFormer 参照 | 无增强、无 AMP/compile 的配方 |
| [InterFormer](interformer.md) | 静态与序列交替交互 | group tokenizer 与 query 数 |
| [OneTrans](onetrans.md) | 统一主干与序列收缩 | 每层 token 预算 |
| [TokenFormer](tokenformer.md) | 统一 token、分层 mask 与门控 | RoPE、增强与全局注意力成本 |
| [Symbiosis](symbiosis.md) | 来源、缺失与长历史选择 | 时间切分、memory 预算与配方 |
| [DualQ](dualq.md) | 双侧 query、pair 与时间信号 | 未迁移的数据 profile、pairwise loss |
| [QueryFormer](queryformer.md) | 多列 embedding 与四类交互 | 列数、参数量、Muon 与 compile |

## 默认训练差异

| 实验 | batch | dense optimizer | split | AMP / compile |
| --- | ---: | --- | --- | --- |
| baseline | 256 | adamw | row_group_tail | false / false |
| interformer | 256 | adamw | row_group_tail | false / false |
| onetrans | 256 | adamw | row_group_tail | false / false |
| tokenformer | 256 | adamw | row_group_tail | true / false |
| symbiosis | 128 | muon | timestamp_auto | true / true |
| dualq | 576 | adamw | timestamp_auto | true / true |
| queryformer | 256 | muon | timestamp_auto | true / true |

默认配置面向不同实验目标，不能直接当作受控横向榜单。比较前固定数据、切分、seed、训练预算、backend 与选 checkpoint 规则，另报告参数量、显存、吞吐和预测指标。

## 维护实验能力

| 实验 | 需要数据 | train / val / infer | 输出 |
| --- | --- | --- | --- |
| [Host Device Info](host-device-info.md) | 否 | 支持 / 不支持 / 不支持 | 环境诊断 stdout |
| [Online Dataset EDA](online-dataset-eda.md) | 是 | 支持 / 不支持 / 支持 | profile 与差异 stdout |

EDA 的推理入口用于统计，不写模型提交所需的 `predictions.json`。维护实验能否生成推理包取决于实际 `infer_fn`，不能只按 kind 判断。

## 证据应怎样表述

实验契约测试检查输入、输出与重建；CPU smoke 检查流程；目标 GPU 测量检查特定环境的数值和速度；完整训练与独立验证才支持质量结论。来源论文和团队文章的成绩单独引用，不标作此仓库实测。

新增包遵循[接入指南](../guide/contributing.md)。
