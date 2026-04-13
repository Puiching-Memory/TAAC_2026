# Onboarding Checklist

## Source Of Truth

Read these files before making structural decisions:

- `docs/dev.md`
- `docs/project-layout.md`
- `docs/packages/index.md`
- `docs/experiments.md`
- `tests/test_experiment_packages.py`
- `tests/training_stack_support.py` when adding or expanding training-stack tests
- the closest existing package under `config/gen`

If repository interfaces disagree, follow the pattern used by the touched package and tests.
Do not mix unrelated migrations into a package-onboarding task unless the user asks for it.

## Executable Package Path

Add or update these files:

- `config/gen/<name>/__init__.py`
- `config/gen/<name>/data.py`
- `config/gen/<name>/model.py`
- `config/gen/<name>/utils.py`
- `docs/packages/<name>.md`

Usually update these files too:

- `docs/packages/index.md`
- `docs/experiments.md`
- `README.md`
- `zensical.toml`
- `tests/test_experiment_packages.py`

Conditionally update these files:

- `docs/papers/<name>.md` when a long-form paper digest is worth maintaining
- `docs/papers/index.md` when a new paper page is added
- shared modules in `src/taac2026` only for truly reusable framework logic
- search-related files or tests when the package needs custom search behavior

Keep these rules:

- Export `EXPERIMENT` from `config/gen/<name>/__init__.py`.
- Keep package-private modeling and data glue inside `config/gen/<name>`.
- Prefer adapting the architecture to the local batch/runtime contract over preserving upstream file layout.
- Record only validation evidence that can be opened in the current workspace.

## Concept-Only Path

Use the concept-only path when the source cannot be integrated cleanly yet.

Usually add or update:

- `docs/packages/<name>.md`
- `docs/packages/index.md`
- `zensical.toml` when the page is added to nav
- `docs/TODO.md` when follow-up engineering work should stay visible

Avoid these changes for concept-only work:

- do not add `config/gen/<name>` as an executable package
- do not add the package to README executable tables
- do not add it to `docs/experiments.md` executable package or smoke-result tables
- do not imply train/evaluate/search support exists if it does not

## Validation Matrix

Prefer the smallest set of checks that proves the claimed state:

- Executable package scaffold only: targeted import and forward test
- Shared runtime change: `uv run pytest tests -q`
- Docs or nav change: `uv run --no-project --isolated --with zensical zensical build --clean`
- Smoke training claim: `uv run taac-train --experiment config/gen/<name>`
- Evaluation claim: `uv run taac-evaluate single --experiment config/gen/<name>`
- Search claim: `uv run taac-search --experiment config/gen/<name> --trials <n>`

When validation is skipped, state why and avoid overstating package readiness.
