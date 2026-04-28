---
icon: lucide/search
---

# 搜索请求

超参数搜索功能。

## 当前状态

`taac-search` 命令已实现基础框架，用于生成超参数搜索请求。

## 基本用法

```bash
uv run taac-search \
  --experiment config/symbiosis \
  --dataset-path data/sample_1000_raw/demo_1000.parquet \
  --schema-path data/sample_1000_raw/schema.json
```

## 输出

搜索请求输出到终端或文件，包含：

- 实验包名称
- 搜索空间定义
- 建议的超参数组合

## 和训练入口的关系

`taac-search` 生成的搜索请求可以传递给 `taac-train` 的 `--extra-args` 参数：

```bash
# 生成搜索请求
uv run taac-search --experiment config/symbiosis > search_params.json

# 使用搜索结果训练
uv run taac-train --experiment config/symbiosis \
  --dataset-path data/sample_1000_raw/demo_1000.parquet \
  --schema-path data/sample_1000_raw/schema.json \
  --learning-rate 0.001 --batch-size 128
```
