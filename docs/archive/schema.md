---
icon: lucide/file-json
---

# Schema 归档

这里统一归档当前文档站发布的 schema 静态副本。它们用于查阅、下载和对比，不改变仓库本地训练、线上 bundle 或推理入口的真实读取路径。

现在保留两份 schema：

| 文件                          | 场景                                      | 下载                                             |
| ----------------------------- | ----------------------------------------- | ------------------------------------------------ |
| `sample_1000_raw.schema.json` | 本地 demo 数据集与 smoke run 的默认参考   | [下载](files/schema/sample_1000_raw.schema.json) |
| `online_schema.schema.json`   | 线上训练与 infer 共用的声明式 schema 快照 | [下载](files/schema/online_schema.schema.json)   |

## 怎么用

本地 demo 和很多样例命令默认围绕 `sample_1000_raw` 组织。它适合用来理解特征结构、写 smoke test、生成小规模合成数据。

线上归档快照更接近真实训练 / infer 当时使用的数据声明。它适合做 bundle 排障、线上特征范围确认，以及和 demo schema 对比差异。

不要把这两份文件随意互相覆盖。它们的结构相似，但部分字段长度、词表上界和序列时间特征写法不同。

## 共同结构

两份 schema 的大结构一致：

| 区块         | 条目数 | 说明                                     |
| ------------ | ------ | ---------------------------------------- |
| `format`     | 1      | 当前值为 `raw_parquet`                   |
| `user_int`   | 46     | 用户离散特征，包含多值字段               |
| `item_int`   | 14     | 物品离散特征                             |
| `user_dense` | 10     | 用户连续特征                             |
| `seq`        | 4 个域 | 序列域包含 prefix、ts_fid 和逐域特征列表 |

序列域也保持一致：

| 域      | prefix         | ts_fid | 特征数 |
| ------- | -------------- | ------ | ------ |
| `seq_a` | `domain_a_seq` | 39     | 9      |
| `seq_b` | `domain_b_seq` | 67     | 14     |
| `seq_c` | `domain_c_seq` | 27     | 12     |
| `seq_d` | `domain_d_seq` | 26     | 10     |

## 主要差异

- `sample_1000_raw.schema.json` 是本地 demo 数据的参考副本。
- `online_schema.schema.json` 是线上训练与 infer 共用的归档副本。
- online schema 中多个 `user_int` 和 `user_dense` 字段长度 / 词表上界与 demo schema 不同。
- online schema 里部分序列时间特征的 vocab 值记录为 0，这和 demo schema 的写法不同。

这些差异意味着：看结构可以放在一起看，跑代码时仍应让当前入口解析自己该用的 schema。

## 路径约定

文档站发布路径统一放在：

```text
docs/archive/files/schema/
```

当前发布文件：

```text
docs/archive/files/schema/
├── online_schema.schema.json
└── sample_1000_raw.schema.json
```

仓库运行时仍以真实入口为准。本地 demo 通常读取仓库的数据快照；线上 bundle 通过 `TAAC_SCHEMA_PATH` 或 checkpoint sidecar 解析 schema。

如果后续继续归档更多版本，沿用：

```text
files/schema/<name>.schema.json
```
