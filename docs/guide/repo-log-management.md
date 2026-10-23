---
icon: lucide/wrench
---

# Actions 日志与部署记录管理

日志是失败复现和部署追溯的证据。先记录任务、commit、运行 ID 和所需日志，再决定是否清理远端记录。

## 查询

```bash
gh run list --repo Puiching-Memory/TAAC_2026 --limit 20
gh run view <run_id> --repo Puiching-Memory/TAAC_2026 --log
```

部署状态可在仓库 [Actions](https://github.com/Puiching-Memory/TAAC_2026/actions) 与 Pages 设置中核对。文档更新的触发与 commit 选择见[站点指南](local-site.md)。

## 本地脚本的实际作用

```bash
bash tools/github-cleanup.sh --repo Puiching-Memory/TAAC_2026 --dry-run
```

此脚本只解析并校验清理请求，输出 repo、模式、dry-run 与 token 是否存在；即使去掉 `--dry-run`，也不会调用 GitHub API 删除内容。不要将脚本退出成功解释成远端已清理。

`--actions-only` 和 `--pages-only` 互斥。认证可使用已有 gh 登录态；若脚本需要 token，使用环境注入，避免把凭据写入文档、命令历史或日志。

## 需要删除时

对照 GitHub 当前 API 或 UI 逐个确认精确 ID、对象类型和删除影响。失败排查、历史部署对比或性能回归还需要的记录应先导出。无筛选的批量删除不利于追溯。

本地缓存清理见[缓存指南](cache-cleanup.md)。该操作与远端 workflow runs、日志或 deployment 记录互不替代。
