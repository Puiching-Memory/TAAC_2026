---
icon: lucide/wrench
---

# 仓库日志管理

通过 GitHub API 清理 CI 工作流日志和 Pages 部署记录。

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
uv run taac-clean-github-logs
```

## 推荐执行流程

1. 设置环境变量
2. 运行清理命令
3. 检查输出

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

- 删除的 workflow run 数量
- 删除的 deployment 数量
- 返回码 0 表示成功

## 返回码约定

| 返回码 | 含义               |
| ------ | ------------------ |
| 0      | 清理成功           |
| 1      | API 错误或权限不足 |

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
