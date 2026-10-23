---
icon: lucide/file-json
---

# Schema 归档

两份 JSON 是文档站发布的静态参考。选择 schema 应由实际 parquet 与 checkpoint 决定，不按“更接近线上”随意互换。

| 文件 | 来源语义 | 下载 |
| --- | --- | --- |
| `sample_1000_raw.schema.json` | 公开 demo 的参考声明 | [JSON](files/schema/sample_1000_raw.schema.json) |
| `online_schema.schema.json` | 历史线上声明快照 | [JSON](files/schema/online_schema.schema.json) |

## 结构和差异

两者均为 `raw_parquet`，包含 user_int、item_int、user_dense 和四个序列域。部分字段长度、词表上界与时间特征表示不同；字段同名不保证张量和 embedding 可交换。

| 序列域 | prefix | 时间 fid |
| --- | --- | ---: |
| seq_a | domain_a_seq | 39 |
| seq_b | domain_b_seq | 67 |
| seq_c | domain_c_seq | 27 |
| seq_d | domain_d_seq | 26 |

## 运行时如何使用

训练显式传 `--schema-path`，否则读取数据目录的 `schema.json`；默认 demo 路径另有仓库位置要求。评估/推理未显式指定时读取 checkpoint 同目录 schema，详见[架构](../architecture.md)。

```bash
mkdir -p data/sample_1000_raw
cp docs/archive/files/schema/sample_1000_raw.schema.json data/sample_1000_raw/schema.json
```

这条命令只准备 schema，不下载数据。parquet 准备见[快速开始](../getting-started.md)。新增归档使用独立 `<name>.schema.json` 并注明来源，避免覆盖已有历史版本。
