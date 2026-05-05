---
icon: lucide/wrench
---

# 仓库缓存清理

清理仓库里的 Python 缓存目录、测试/工具缓存和常见构建产物。

## 目标

删除 `__pycache__`、`.ruff_cache/`、`.pytest_cache/`、`.benchmarks/`、`.cache/`、`*.egg-info`、`.coverage`、`.coverage.cpu`、`dist/`、`build/` 等缓存与产物，保持仓库干净。

## 脚本入口

```bash
# 预览将要删除的内容
bash tools/cache-cleanup.sh --dry-run

# 执行清理
bash tools/cache-cleanup.sh
```

## 常用命令

```bash
# 清理当前目录及子目录的所有 __pycache__
find . -type d -name "__pycache__" -exec rm -rf {} +

# 清理 .pyc 文件
find . -type f -name "*.pyc" -delete

# 清理 egg-info
find . -type d -name "*.egg-info" -exec rm -rf {} +

# 清理常见测试和工具缓存
find . -type d \( -name ".ruff_cache" -o -name ".pytest_cache" -o -name ".benchmarks" -o -name ".cache" \) -exec rm -rf {} +

# 清理覆盖率文件
find . -type f \( -name ".coverage" -o -name ".coverage.cpu" \) -delete
```

## 脚本参数

- `--root <path>`：指定清理根目录，默认是仓库根目录
- `--dry-run`：只打印将删除的目录和文件，不真正删除
- `--include-env-dirs`：连 `.venv`、`venv`、`env`、`node_modules`、`.tox`、`.mypy_cache` 里的缓存目录和覆盖率文件也一起清理

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
