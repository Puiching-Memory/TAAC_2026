---
icon: lucide/wrench
---

# 仓库日志管理

这页用于处理 GitHub Actions 日志和 Pages 部署记录。这里的操作会影响远端仓库，请先 dry-run 和列清楚目标，再删除。

## 本仓库脚本做什么

`tools/github-cleanup.sh` 只负责收集和校验参数，不会直接删除 GitHub 远端数据。

```bash
export GITHUB_REPO=Puiching-Memory/TAAC_2026
export GITHUB_TOKEN=ghp_xxxxx

bash tools/github-cleanup.sh --dry-run
```

输出里会告诉你目标仓库、模式和 token 是否提供。真正删除要使用 `gh api` 或 GitHub UI。

## 建议流程

1. 用 GitHub UI 或 `gh` 列出要删除的 workflow runs / Pages deployments。
2. 记录 ID，确认不是正在排查的失败任务。
3. 运行 `tools/github-cleanup.sh --dry-run` 记录目标和模式。
4. 对单个 ID 执行删除命令。
5. 回到 Actions / Pages 页面确认状态。

## 常用查询

列出最近的 workflow runs：

```bash
gh run list --repo Puiching-Memory/TAAC_2026 --limit 20
```

列出 Pages deployments：

```bash
gh api repos/Puiching-Memory/TAAC_2026/pages/deployments --paginate
```

删除某个 workflow run 的日志：

```bash
gh api --method DELETE \
  repos/Puiching-Memory/TAAC_2026/actions/runs/<run_id>/logs
```

删除某个 workflow run：

```bash
gh api --method DELETE \
  repos/Puiching-Memory/TAAC_2026/actions/runs/<run_id>
```

Pages deployment 的删除和 inactive 操作请先用 `gh api repos/<owner>/<repo>/pages/deployments` 查到具体 ID，再按 GitHub API 当前文档执行。不要写一个没有筛选条件的批量删除管道。

## 什么时候不该清

- 失败任务还在排查。
- 正在比较 Pages 部署差异。
- 不确定 token 权限范围。
- 只是本地仓库变脏，这种情况用 [仓库缓存清理](cache-cleanup.md)。

## 脚本入口

- `tools/github-cleanup.sh`
