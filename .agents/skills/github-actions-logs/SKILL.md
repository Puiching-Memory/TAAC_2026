---
name: github-actions-logs
description: >
  Retrieve and analyze GitHub Actions CI/CD logs. Use when the user provides a
  GitHub Actions run URL (e.g. https://github.com/OWNER/REPO/actions/runs/RUN_ID)
  and needs to diagnose CI failures, view raw logs, or summarize build results.
  Works with or without the raw logs direct URL.
---

# GitHub Actions Log Retrieval Skill

## When to Use

The user provides one of:
- A GitHub Actions **run** URL: `https://github.com/OWNER/REPO/actions/runs/RUN_ID`
- A GitHub Actions **job** URL: `https://github.com/OWNER/REPO/actions/runs/RUN_ID/job/JOB_ID`
- A raw logs blob URL (signed SAS URL from `productionresultssa7.blob.core.windows.net`)

## Prerequisites

- `gh` CLI must be installed and authenticated (`gh auth login` or set `GH_TOKEN`)
- If `gh` is not installed, install it first:
  ```bash
  (type -p wget >/dev/null || (apt update && apt-get install wget -y)) \
    && mkdir -p -m 755 /etc/apt/keyrings \
    && wget -qO- https://cli.github.com/packages/githubcli-archive-keyring.gpg \
       | tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null \
    && chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
       | tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
    && apt update && apt install gh -y
  ```

## Retrieval Methods

### Method 1: `gh` CLI (Preferred)

Extract `OWNER/REPO` and `RUN_ID` from the URL, then:

```bash
# List all jobs and their status
gh api repos/{OWNER}/{REPO}/actions/runs/{RUN_ID}/jobs?per_page=100 | \
  python3 -c "
import json, sys
data = json.load(sys.stdin)
for job in data.get('jobs', []):
    print(f\"Job: {job['name']} | Status: {job['conclusion']} | ID: {job['id']}\")
    for step in job.get('steps', []):
        print(f\"  Step: {step['name']} | Status: {step['conclusion']}\")
"

# Get logs for a failed job (replace JOB_ID with the numeric job ID)
gh api repos/{OWNER}/{REPO}/actions/jobs/{JOB_ID}/logs

# Get logs only for failed steps (pipe through grep/tail)
gh api repos/{OWNER}/{REPO}/actions/jobs/{JOB_ID}/logs 2>&1 | grep -A5 "error:"
```

**Important**: `gh run view --log-failed` only works when the run is **completed**.
If the run is still in progress, it will say "logs will be available when it is complete".
In that case, use `gh api .../actions/jobs/{JOB_ID}/logs` instead — this works even
for in-progress or recently completed runs.

### Method 2: Raw Logs URL (Direct Blob)

If the user provides a `productionresultssa7.blob.core.windows.net` URL:

```
https://productionresultssa7.blob.core.windows.net/actions-results/{GUID}/workflow-job-run-{JOB_RUN_GUID}/logs/job/job-logs.txt?{SAS_PARAMS}
```

- These URLs are **time-limited signed URLs** (SAS tokens). They expire after ~1 hour.
- Use `fetch_webpage` tool to retrieve content, or `curl` in terminal:
  ```bash
  curl -sL "RAW_LOGS_URL"
  ```
- If expired, fall back to Method 1.

### Method 3: GitHub API (No `gh` CLI)

If `gh` is not available, use `curl` directly:

```bash
# List jobs
curl -sL "https://api.github.com/repos/{OWNER}/{REPO}/actions/runs/{RUN_ID}/jobs" | python3 -c "..."

# Get job logs (requires auth for private repos)
curl -sL -H "Authorization: token ${GH_TOKEN}" \
  "https://api.github.com/repos/{OWNER}/{REPO}/actions/jobs/{JOB_ID}/logs"
```

### Method 4: Web Page Scraping (Last Resort)

Use `fetch_webpage` on the Actions page URL. GitHub requires sign-in for full logs,
but annotations (error/warning summaries) are visible on public repos without login.

## URL Parsing Guide

| URL Pattern | Extract |
|---|---|
| `github.com/OWNER/REPO/actions/runs/RUN_ID` | `OWNER`, `REPO`, `RUN_ID` |
| `github.com/OWNER/REPO/actions/runs/RUN_ID/job/JOB_ID` | + `JOB_ID` |
| `productionresultssa7.blob.core.windows.net/actions-results/GUID/...` | Direct blob (SAS) |

## Common CI Failure Patterns

After retrieving logs, look for:
- `##[error]` — GitHub Actions error markers
- `error:` — compiler/clang-tidy errors
- `fatal:` — git or build fatal errors
- `FAILED` — test or build failures
- `not found` — missing dependencies
- `Process completed with exit code 1` — step failure boundary

## Example Workflow

1. User gives: `https://github.com/foo/bar/actions/runs/12345/job/67890`
2. Parse: `OWNER=foo`, `REPO=bar`, `RUN_ID=12345`, `JOB_ID=67890`
3. Run: `gh api repos/foo/bar/actions/jobs/67890/logs 2>&1 | tail -80`
4. Analyze the error output and report findings to the user
