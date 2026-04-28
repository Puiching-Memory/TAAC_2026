---
icon: lucide/wrench
---

# 仓库缓存清理

清理 Python 缓存文件和构建产物。

## 目标

删除 `__pycache__`、`.pyc`、`.pyo`、`*.egg-info`、`dist/`、`build/` 等缓存目录，保持仓库干净。

## 命令入口

```bash
# 使用 CLI 命令
uv run taac-clean-pycache

# 使用 run.sh
./run.sh clean

# 使用 Makefile（如果存在）
make clean
```

## 常用命令

```bash
# 清理当前目录及子目录的所有 __pycache__
find . -type d -name "__pycache__" -exec rm -rf {} +

# 清理 .pyc 文件
find . -type f -name "*.pyc" -delete

# 清理 egg-info
find . -type d -name "*.egg-info" -exec rm -rf {} +
```

## 脚本参数

`taac-clean-pycache` 无额外参数，默认清理仓库根目录下的所有 Python 缓存。

## 输出解释

- 删除目录数和文件数会打印到终端
- 返回码 0 表示成功

## 推荐使用时机

- 切换 Python 版本后
- 提交代码前
- CI 构建前
- 磁盘空间不足时

## 返回码约定

| 返回码 | 含义                   |
| ------ | ---------------------- |
| 0      | 清理成功               |
| 1      | 发生错误（权限不足等） |
