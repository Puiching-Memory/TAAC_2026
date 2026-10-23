---
icon: lucide/archive
---

# 归档：参考 Schema 与官方快照

这里保存已有参考资产，便于下载、对照和追溯。归档内容不参与当前共享运行时的自动选择，也不代表最新平台契约。

| 内容 | 阅读入口 | 用途 |
| --- | --- | --- |
| demo 与 online schema | [Schema 归档](schema.md) | 对照字段、长度和词表 |
| 官方训练源码 | [训练 Baseline 快照](official-train-baseline.md) | 理解历史训练和 sidecar 行为 |
| 官方推理源码 | [推理 Baseline 快照](official-infer-baseline.md) | 对照环境变量与预测格式 |

归档正文可重写说明，但 `files/` 下的源码与 JSON 应保持原始快照内容。当前训练、评估和推理使用 `experiments/` 与 `src/taac2026/`，新上传物由打包 CLI 生成。

需要运行 demo 时，按[快速开始](../getting-started.md)显式复制参考 schema 并下载 parquet；不要假设文档目录自动成为数据目录。
