---
icon: lucide/wrench
---

# 仓库缓存清理

这个脚本只清理本地缓存和构建产物，用来让工作区变轻一点。它不处理实验输出目录，也不替你判断哪些结果值得保留。

## 先预览

```bash
bash tools/cache-cleanup.sh --dry-run
```

确认列表合理后再执行：

```bash
bash tools/cache-cleanup.sh
```

## 会清什么

默认会清理：

- `__pycache__/` 和 `.pyc`
- `.ruff_cache/`
- `.pytest_cache/`
- `.benchmarks/`
- `.cache/`
- `*.egg-info`
- `.coverage` 和 `.coverage.cpu`
- `build/` 和 `dist/`

默认不会深入 `.venv`、`venv`、`env`、`node_modules`、`.tox` 和 `.mypy_cache` 这些环境目录。

## 常用参数

```bash
# 指定清理根目录
bash tools/cache-cleanup.sh --root /path/to/repo --dry-run

# 连环境目录里的缓存也一起清
bash tools/cache-cleanup.sh --include-env-dirs --dry-run
```

`--include-env-dirs` 建议先 dry-run，因为它会扩大扫描范围。

## 什么时候用

- 切换 Python 版本后。
- 跑完大量测试或 benchmark 后。
- 提交前想移除工具缓存。
- 本地磁盘空间紧张，但还不想删 `outputs/`。

## 脚本入口

- `tools/cache-cleanup.sh`
