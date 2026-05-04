---
name: taac-docs-pages-pipeline
description: 'Use when reviewing or editing TAAC_2026 docs/, zensical.toml, site/, or GitHub Pages workflows. Covers the real GitHub Pages build and deploy path, CI gating, docs-only vs code-change behavior, and how to validate docs changes without treating stale site/ artifacts as a Pages regression.'
argument-hint: 'docs deploy, pages workflow, site mismatch, docs validation'
user-invocable: true
---

# TAAC Docs Pages Pipeline

## When to Use

- Review changes under docs/, zensical.toml, .github/workflows/, or site/.
- Decide whether a site/ mismatch is a real GitHub Pages problem or just a stale local build artifact.
- Explain how GitHub Pages is deployed from this repository.
- Check whether a docs change will run CI, deploy Pages, both, or neither.
- Validate doc-site related review comments before raising them.

## Files To Inspect First

Read these files before making claims about docs deployment or testing:

- .github/workflows/deploy-docs.yml
- .github/workflows/ci.yml
- docs/guide/local-site.md
- docs/getting-started.md
- zensical.toml

These files are the source of truth for GitHub Pages behavior. Do not infer deployment behavior from the current contents of site/ alone.

## Core Facts

### 1. site/ is a generated output, not the Pages source of truth

- Local docs build writes to site/ via uv run zensical build.
- docs/guide/local-site.md explicitly documents site/ as the build output and suggests rm -rf site/ for cleanup.
- .gitignore includes site/.
- GitHub Pages deployment does not publish the checked-in site/ tree directly. The Action rebuilds the site on the runner, uploads a fresh Pages artifact from site/, then deploys that artifact.

Practical rule:

- A stale site/ directory in the repository is not, by itself, evidence of a GitHub Pages regression.
- Only treat site/ drift as user-facing if the user explicitly says the committed static export matters, or another workflow outside GitHub Pages consumes it.

### 2. GitHub Pages is rebuilt online from source

The deploy-docs workflow performs this sequence:

1. Configure GitHub Pages.
2. Check out the repository at the push SHA or the CI workflow_run head SHA.
3. Set up Python 3.13.
4. Set up uv.
5. Install a lightweight project environment.
6. Run uv run --no-project --isolated --with zensical zensical build --clean.
7. Upload the freshly built site/ as a Pages artifact.
8. Deploy that artifact with actions/deploy-pages.

This means GitHub Pages always reflects a fresh online build, not whatever static files happened to be committed locally.

## Online Flow Matrix

### Docs-only push to main

Trigger source:

- deploy-docs.yml listens to pushes on main that touch:
  - docs/**
  - zensical.toml
  - .github/workflows/deploy-docs.yml

Behavior:

- classify-push allows direct deployment as long as the same push did not also touch:
  - .github/workflows/ci.yml
  - experiments/**
  - src/**
  - tests/**
  - pyproject.toml
  - uv.lock
- If allowed, GitHub Pages deploys immediately from deploy-docs without waiting for CI.

Implication:

- Docs-only pushes to main can deploy without any CI test job running.

### Code-related push to main

Trigger source:

- ci.yml runs on pushes to main that touch:
  - .github/workflows/ci.yml
  - experiments/**
  - src/**
  - tests/**
  - pyproject.toml
  - uv.lock
- deploy-docs.yml also listens to workflow_run for CI completion on main.

Behavior:

- deploy-docs only deploys on workflow_run when the CI conclusion is success.
- If a push touches both docs and code paths, classify-push blocks direct push deployment.
- After CI passes, workflow_run becomes the deployment path.

Implication:

- Code-affecting pushes on main gate Pages deployment on successful CI.

### Pull requests

Behavior:

- ci.yml runs for pull requests only when code/test/build paths change.
- deploy-docs.yml does not run on pull_request.
- There is no Pages preview workflow for pull requests.

Implication:

- Docs-only PRs do not get CI and do not deploy Pages.
- A docs review should not assume “PR passed online docs validation” unless the reviewer can point to another explicit workflow.

## Online Test Flow

CI is the repository's online code-validation path, not the docs deployment path.

### style job

- Matrix: Python 3.10, 3.11, 3.12, 3.13.
- Command: uv run --with ruff ruff check .

### test job

- Matrix: Python 3.10, 3.11, 3.12, 3.13.
- Command: uv sync --locked --extra cuda126 --no-install-package torchrec --no-install-package fbgemm-gpu
- Test command: uv run --with coverage coverage run --data-file=.coverage.cpu -m pytest -m unit -v

### coverage job

- Runs after test.
- Uses the canonical Python 3.13 lane.
- Restores uploaded coverage data.
- Enforces fail-under=70 for a limited CPU-safe include set.

### Docs-specific online validation

- There is no dedicated docs lint or docs test job inside ci.yml.
- For docs-only pushes to main, the effective online validation is whether deploy-docs can build the site successfully.
- For code pushes, Pages deployment depends on CI success first, then a successful docs build inside deploy-docs.

## Local Validation Rules

When editing docs or zensical.toml locally:

1. Use uv run zensical serve for local preview.
2. Use uv run zensical build --strict for a tighter one-shot validation when you need a publish-like check.
3. Treat site/ as disposable local output unless the user explicitly wants it refreshed.

When reviewing documentation changes:

1. Check whether the change is docs-only or code-affecting.
2. Map it to the correct online path:
   - docs-only main push -> deploy-docs directly
   - code push to main -> CI, then workflow_run deploy-docs
   - PR -> maybe CI, never Pages deploy
3. Only raise a GitHub Pages issue if the workflow-defined path would fail or produce wrong content.
4. Do not raise “site/ is stale” as a Pages bug unless the review scope explicitly includes committed static exports.

## Review Heuristics

Use these heuristics to avoid false positives:

- If the concern is only that docs/ and site/ differ, verify deploy-docs.yml before flagging it.
- If the user asks whether GitHub Pages is broken, inspect the workflow triggers and build commands first.
- If a docs-only PR lacks CI, that is expected under the current workflow configuration.
- If a docs-only push to main deploys without CI, that is also expected under the current workflow configuration.
- If code and docs changed together, expect Pages deployment to come from workflow_run after CI success, not from the direct push path.

## Quick Answers

- Does GitHub Pages publish the repository's current site/ tree directly?
  - No. It rebuilds site/ online, uploads the artifact, and deploys that artifact.

- Should a stale checked-in site/ directory block a Pages review by default?
  - No.

- Do docs-only PRs get online docs validation?
  - No, not from the current workflows.

- Do docs-only pushes to main wait for CI?
  - No.

- Do code pushes to main gate Pages deployment on CI success?
  - Yes.