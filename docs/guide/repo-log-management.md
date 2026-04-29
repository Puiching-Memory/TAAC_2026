---
icon: lucide/wrench
---

# 仓库日志管理

通过 shell 脚本收集 GitHub 清理请求参数，或直接使用 `gh api` 清理 CI 工作流日志和 Pages 部署记录。

## 目标

减少仓库存储占用，清理过期的 CI 日志和部署记录。

## 覆盖范围

- GitHub Actions Workflow Runs 日志
- GitHub Pages Deployments 记录

## 使用前准备

需要 GitHub Token：

```bash
export GITHUB_TOKEN=ghp_xxxxx
export GITHUB_REPO=Puiching-Memory/TAAC_2026
```

## 脚本参数

```bash
# 使用环境变量
bash tools/github-cleanup.sh --dry-run

# 或显式传参
bash tools/github-cleanup.sh --repo Puiching-Memory/TAAC_2026 --actions-only
```

当前仓库内脚本只负责参数校验和请求记录，不会直接调用 GitHub API 删除日志。需要真正执行清理时，请使用下文列出的 `gh api` 命令。

## 推荐执行流程

1. 设置环境变量
2. 用 `bash tools/github-cleanup.sh ...` 校验目标仓库和模式
3. 按需执行下方 `gh api` 命令
4. 检查输出

## 常用命令

```bash
# 清理所有 workflow run 日志
gh api --method DELETE repos/{owner}/{repo}/actions/runs

# 清理超过 30 天的 workflow runs
gh api repos/{owner}/{repo}/actions/runs --paginate \
  | jq '.workflow_runs[] | select(.created_at < "2026-03-01") | .id' \
  | xargs -I {} gh api --method DELETE repos/{owner}/{repo}/actions/runs/{}
```

## 输出解释

- 目标仓库
- 是否 dry-run
- 返回码 0 表示参数合法
- 返回码 0 表示成功

## 返回码约定

| 返回码 | 含义               |
| ------ | ------------------ |
| 0      | 参数校验通过       |
| 1      | 参数缺失或校验失败 |
| 2      | 参数组合非法       |

## GitHub API 接口清单

### 1) 列出 workflow runs

```
GET /repos/{owner}/{repo}/actions/runs
```

### 2) 删除 workflow run 日志

```
DELETE /repos/{owner}/{repo}/actions/runs/{run_id}/logs
```

### 3) 列出 Pages deployments

```
GET /repos/{owner}/{repo}/pages/deployments
```

### 4) 将 deployment 标记为 inactive

```
POST /repos/{owner}/{repo}/pages/deployments/{deployment_id}
```

### 5) 删除 deployment

```
DELETE /repos/{owner}/{repo}/pages/deployments/{deployment_id}
```
