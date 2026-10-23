---
icon: lucide/wrench
---

# 清理本地缓存

`tools/cache-cleanup.sh` 删除可再生缓存和构建文件，不处理实验结果的取舍。先预览目标，再执行同一范围。

```bash
bash tools/cache-cleanup.sh --dry-run
bash tools/cache-cleanup.sh
```

## 清理范围

默认匹配 `__pycache__`、`.pyc`、Ruff/Pytest/benchmark 缓存、`.cache`、egg-info、coverage 数据及 `build/`、`dist/`。脚本不以清空训练结果为目标，重要产物仍应单独备份。

默认不深入 `.venv`、`venv`、`env`、`node_modules`、`.tox`、`.mypy_cache`。扩大范围或切换根目录时重新预览：

```bash
bash tools/cache-cleanup.sh --root /path/to/repo --dry-run
bash tools/cache-cleanup.sh --include-env-dirs --dry-run
```

## 执行前后

确认目标是当前仓库，停止正在写这些缓存的进程。删除 native/JIT 缓存后首次运行可能重新编译；重新测量时不要把冷启动与原来的稳态结果直接比较。

站点产物由 Zensical `--clean` 重建，依赖环境由 uv 管理。远端 Actions 日志的整理见[日志管理](repo-log-management.md)，不是本脚本职责。
