---
name: taac-docs-pages-pipeline
description: Use when editing or reviewing TAAC docs, zensical configuration, generated site artifacts, GitHub Pages deployment workflows, docs CI behavior, or questions about whether documentation changes deploy correctly.
---

# TAAC Docs Pages Pipeline

Use this skill to avoid stale-docs and Pages false positives. Treat workflow files, docs config, and touched docs as the source of truth.

## Read First

- Touched files under `docs/`
- The nearest section `index.md` for changed docs, when changing navigation or page roles
- `zensical.toml`, when changing navigation, assets, theme, or page paths
- `.github/workflows/deploy-docs.yml` and `.github/workflows/ci.yml`, when making claims about CI or Pages behavior
- `docs/guide/local-site.md`, when changing local build, deployment, or generated-site behavior

## Non-Obvious Context

- Section `index.md` pages are directional guides. Leaf docs should carry implementation details: commands, inputs/outputs, environment variables, source entrypoints, and failure modes.
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
