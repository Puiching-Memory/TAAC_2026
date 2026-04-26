"""Remove Python bytecode cache directories from the workspace."""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence


_ENV_DIR_NAMES = {".venv", "venv", "env", "node_modules", ".tox", ".mypy_cache"}


@dataclass(slots=True)
class CleanResult:
    root: Path
    matched_dirs: list[Path]
    matched_files: int
    total_bytes: int
    failures: list[str]
    dry_run: bool = False


def _is_inside_env(path: Path, root: Path) -> bool:
    try:
        relative_parts = path.relative_to(root).parts
    except ValueError:
        return False
    return any(part in _ENV_DIR_NAMES for part in relative_parts)


def find_pycache_dirs(root: Path, *, include_env_dirs: bool = False) -> list[Path]:
    resolved_root = root.expanduser().resolve()
    matches = [path for path in resolved_root.rglob("__pycache__") if path.is_dir()]
    if not include_env_dirs:
        matches = [path for path in matches if not _is_inside_env(path, resolved_root)]
    return sorted(matches)


def clean_pycache(root: Path, *, dry_run: bool = False, include_env_dirs: bool = False) -> CleanResult:
    resolved_root = root.expanduser().resolve()
    matched_dirs = find_pycache_dirs(resolved_root, include_env_dirs=include_env_dirs)
    matched_files = 0
    total_bytes = 0
    failures: list[str] = []
    for cache_dir in matched_dirs:
        for child in cache_dir.rglob("*"):
            if child.is_file():
                matched_files += 1
                total_bytes += child.stat().st_size
        if dry_run:
            continue
        try:
            shutil.rmtree(cache_dir)
        except OSError as error:
            failures.append(f"{cache_dir}: {error}")
    return CleanResult(resolved_root, matched_dirs, matched_files, total_bytes, failures, dry_run=dry_run)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Remove __pycache__ directories")
    parser.add_argument("--root", default=".")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--include-env-dirs", action="store_true")
    args = parser.parse_args(argv)
    result = clean_pycache(Path(args.root), dry_run=args.dry_run, include_env_dirs=args.include_env_dirs)
    print(
        f"root={result.root} dirs={len(result.matched_dirs)} files={result.matched_files} "
        f"bytes={result.total_bytes} dry_run={result.dry_run} include_env_dirs={args.include_env_dirs}"
    )
    for cache_dir in result.matched_dirs:
        print(cache_dir)
    if result.failures:
        for failure in result.failures:
            print(failure)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
