---
name: taac-docs-pages-pipeline
description: Use when editing or reviewing TAAC docs, zensical configuration, generated site artifacts, GitHub Pages deployment workflows, docs CI behavior, or questions about whether documentation changes deploy correctly.
---

# TAAC Docs Pages Pipeline

Use this skill to avoid stale-docs and Pages false positives. Treat workflow files and docs config as the source of truth.

## Read First

- `.github/workflows/deploy-docs.yml`
- `.github/workflows/ci.yml`
- `zensical.toml`
- `docs/guide/local-site.md`
- `docs/getting-started.md`
- touched files under `docs/`

## Non-Obvious Context

- `site/` is generated local output. Do not treat drift between `docs/` and `site/` as a GitHub Pages regression unless the user explicitly cares about committed static output.
- Pages behavior is defined by the workflows. Inspect triggers and build commands before claiming docs-only changes do or do not run CI/deploy.
- Pull request behavior and main-branch deployment behavior can differ; verify the event path in workflow YAML instead of assuming one pipeline.

## Validation

For docs content or `zensical.toml` changes:

```bash
uv run zensical build --strict
```

For workflow changes:

```bash
uv run --with ruff ruff check .github src tests
```

Only refresh or remove `site/` when the user asks for generated output handling, or when a local docs command intentionally produced it.
