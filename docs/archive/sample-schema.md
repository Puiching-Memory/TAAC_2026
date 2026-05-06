---
icon: lucide/file-json
---

# 样例 Schema

这个页面归档了 sample_1000_raw 的默认 schema。它是本仓库本地 demo 数据集最常用的特征声明文件，也是很多 smoke run、样例命令和合成数据生成流程的基准输入。

## 下载

- [sample_1000_raw.schema.json](files/schema/sample_1000_raw.schema.json)

## 结构摘要

| 区块       | 条目数 | 说明                                     |
| ---------- | ------ | ---------------------------------------- |
| format     | 1      | 当前值为 raw_parquet                     |
| user_int   | 46     | 用户离散特征，包含多值字段               |
| item_int   | 14     | 物品离散特征                             |
| user_dense | 10     | 用户连续特征                             |
| seq        | 4 个域 | 序列域包含 prefix、ts_fid 和逐域特征列表 |

## 序列域

| 域    | prefix       | ts_fid | 特征数 |
| ----- | ------------ | ------ | ------ |
| seq_a | domain_a_seq | 39     | 9      |
| seq_b | domain_b_seq | 67     | 14     |
| seq_c | domain_c_seq | 27     | 12     |
| seq_d | domain_d_seq | 26     | 10     |

## 使用说明

- 文档站里的这份文件是静态归档副本，便于查看和下载。
- 文档站里的 schema 归档统一放在 files/schema 下，这份文件使用名称 sample_1000_raw.schema.json。
- 当前仓库运行时默认仍直接读取 data/sample_1000_raw/schema.json。
- 线上训练与 infer 共用的 schema 快照已单独归档在 [Online Schema 快照](training-schema-snapshot.md)。
- 如果后续需要保留更多 schema 版本，建议继续放在 data/schema_snapshots 下，再按需要补充到文档站归档区。