---
name: taac-experiment-onboard
description: Onboard, migrate, or scaffold TAAC 2026 experiment packages in this repository. Use when Codex needs to adapt a paper or public repo into a package under `config/gen`, decide whether a proposal should stay concept-only, update package docs/tests/experiment tables together, or align a package with the current `EXPERIMENT` contract and train/evaluate/search workflow.
---

# Taac Experiment Onboard

## Overview

Use this skill to turn an external idea into a repo-native TAAC experiment package without breaking the repository contract.
Prefer this skill when the task spans code, tests, docs, experiment registry pages, and validation artifacts together.

## Workflow

### Build context first

Open the current repository contract before editing:

- `docs/dev.md`
- `docs/project-layout.md`
- `tests/test_experiment_packages.py`
- one nearby experiment package under `config/gen`
- any touched runtime contract file such as `src/taac2026/experiment.py`, `src/taac2026/experiment_loader.py`, or `src/taac2026/search.py`

Trust the live branch over stale docs, old READMEs, or upstream source layouts.
If the branch is mid-migration, mirror the active pattern used by the files you are editing instead of performing an unrelated repo-wide conversion.

### Classify the target

Choose one of two paths early:

- **Executable experiment package**: the model can be wired into the current train/evaluate stack and deserves `config/gen/<name>`.
- **Concept-only draft**: the idea is not runnable yet; keep it in docs only and do not add it to executable experiment rosters.

If the source material is incomplete, tightly coupled to unavailable infra, or too speculative for current validation, prefer the concept-only path.

### Implement the package contract

For an executable package, create or update:

- `config/gen/<name>/__init__.py`
- `config/gen/<name>/data.py`
- `config/gen/<name>/model.py`
- `config/gen/<name>/utils.py`

Export `EXPERIMENT` from `__init__.py`.
Keep experiment-private data/model/loss wiring inside the package.
Move reusable framework logic into `src/taac2026` only when it is genuinely shared by multiple packages or CLI/test code.
Do not copy an upstream repository layout verbatim if it fights the local contract.

### Update docs and indexes together

When the package is executable, update the visible registry together with the code:

- `docs/packages/<name>.md` is required.
- `docs/papers/<name>.md` is optional and only worth adding for a longer engineering-oriented paper digest.
- Update `docs/packages/index.md`, `docs/experiments.md`, `README.md`, and `zensical.toml` when the new package becomes part of the documented roster.

When the work is concept-only, keep it out of executable lists and smoke-result tables.
Never document capabilities, results, or validation states that do not exist in the current workspace.

### Validate before closing

Run the smallest useful checks first, then the full regression when feasible:

- targeted package build / forward test
- `uv run pytest tests -q`
- `uv run taac-train --experiment config/gen/<name>` for smoke training when feasible
- `uv run taac-evaluate single --experiment config/gen/<name>` when a checkpoint exists
- `uv run taac-search --experiment config/gen/<name> --trials <n>` only when search support matters for the task
- `uv run --no-project --isolated --with zensical zensical build --clean` when docs or nav changed

Summarize what was validated, what was intentionally skipped, and any deliberate deviations from the upstream paper or codebase.

## Reference

Read `references/onboarding-checklist.md` for the file-by-file checklist and validation matrix.
Keep that reference synchronized with the repository contract as it evolves.
