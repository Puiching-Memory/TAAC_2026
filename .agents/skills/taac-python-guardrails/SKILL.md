---
name: taac-python-guardrails
description: Enforce repository-specific Python workflow rules for this TAAC 2026 codebase. Use when Codex edits Python modules, dependency declarations, docs with runnable commands, shell snippets, or tooling guidance and must avoid try-import dependency fallbacks while standardizing all environment and command usage around uv and uv run.
---

# Taac Python Guardrails

## Overview

Use this skill to keep Python changes aligned with two repository rules: do not write `try import` fallback patterns, and use `uv` plus `uv run` for dependency and command management.
Apply it to code changes, docs, scripts, review comments, and refactors that touch Python imports or execution instructions.

## Rules

### Ban try-import dependency fallbacks

Do not write patterns such as:

- `try: import x`
- `except ModuleNotFoundError`
- `except ImportError` used to degrade features or hide undeclared dependencies
- helper wrappers whose only purpose is deferred optional imports

If code needs a package, declare the dependency and import it normally.
If a feature is truly separate, isolate it behind a separate command, module, or task whose environment explicitly includes that package instead of soft-failing at runtime.

Dynamic loading for real product behavior is still allowed.
Using `importlib` to load experiment packages by path or module name is different from using `try import` to paper over missing dependencies.

### Standardize on uv and uv run

Use:

- `uv sync --locked` to materialize the project environment
- `uv run ...` for project commands
- `uv run --with package ...` for one-off tools that should not become permanent project dependencies
- `uv add`, `uv remove`, and `uv lock` when dependency declarations need to change

Avoid:

- `pip install ...`
- `python -m venv ...`
- bare `python script.py`
- bare `pytest`, `zensical`, or similar commands in docs and review guidance

When writing docs, issue templates, or code review suggestions, normalize examples to `uv` commands even if other tools would work.

### Editing workflow

When touching Python code:

1. Check whether the imported package is already declared in `pyproject.toml`.
2. If not, update dependencies with the expectation that the environment will be managed by `uv`.
3. Replace optional-import wrappers with direct imports or a cleaner module boundary.
4. Update nearby docs and commands so they use `uv run`.
5. Call out any existing violations you did not fix.

### Review stance

During review, flag:

- optional dependency fallbacks via `try import`
- new commands that bypass `uv run`
- docs that tell users to use `pip` or bare Python executables
- dependency additions that are not reflected in `pyproject.toml` and the lockfile

If the user wants the rule to hold automatically, recommend pairing this skill with CI or lint checks.
The skill improves consistency, but hard enforcement belongs in automation.

## Reference

Read `references/rules-and-hotspots.md` for approved patterns, disallowed patterns, and current cleanup candidates in this repository.
