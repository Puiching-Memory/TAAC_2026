---
icon: lucide/file-lock-2
---

# Online Schema 快照

这个页面归档了线上使用的共享 schema，训练和 infer 当前都复用这一份声明式 schema。源文件来自 data/schema_snapshots/infer_2026-05-06_ams/schema.json。文档站现在把所有 schema 都统一发布到 files/schema 目录下，并用文件名区分来源，这份共享 schema 的发布名是 online_schema.schema.json。

## 下载

- [online_schema.schema.json](files/schema/online_schema.schema.json)

## 当前约定

- 仓库原始快照文件名仍是 schema.json，路径保留在 data/schema_snapshots/infer_2026-05-06_ams/schema.json。
- 文档站里的统一发布名改成 online_schema.schema.json，放在 files/schema 下。
- 当前线上训练和 infer 归档不再各放一份重复文件，而是共用这一份 online schema。
- 如果未来还要发布更多 schema，继续沿用 files/schema/<name>.schema.json 的格式即可。

## 结构摘要

| 区块       | 条目数 | 说明                                          |
| ---------- | ------ | --------------------------------------------- |
| format     | 1      | 当前值为 raw_parquet                          |
| user_int   | 46     | 用户离散特征，部分字段长度比 demo schema 更大 |
| item_int   | 14     | 物品离散特征                                  |
| user_dense | 10     | 用户连续特征                                  |
| seq        | 4 个域 | 序列域包含 prefix、ts_fid 和逐域特征列表      |

## 序列域

| 域    | prefix       | ts_fid | 特征数 |
| ----- | ------------ | ------ | ------ |
| seq_a | domain_a_seq | 39     | 9      |
| seq_b | domain_b_seq | 67     | 14     |
| seq_c | domain_c_seq | 27     | 12     |
| seq_d | domain_d_seq | 26     | 10     |

## 与默认样例 Schema 的区别

- 这份文件是线上归档快照，不是本地 demo 路径下的默认 schema。
- 多个 user_int 和 user_dense 字段的长度、词表上界与 sample_1000_raw schema 不同，更接近当时训练所见到的数据声明。
- 部分序列时间特征在 schema 中的 vocab 值记录为 0，这和 demo schema 的写法不同，因此不应直接拿两份文件互相覆盖。