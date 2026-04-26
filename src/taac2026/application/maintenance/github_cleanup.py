"""Thin placeholder for repository-hosted cleanup automation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from collections.abc import Sequence


@dataclass(slots=True)
class CleanupCounter:
    listed: int = 0
    targeted: int = 0
    deleted: int = 0
    failed: int = 0


def _validate_args(args: argparse.Namespace) -> str | None:
    if getattr(args, "actions_only", False) and getattr(args, "pages_only", False):
        return "--actions-only and --pages-only cannot be used together"
    return None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Clean GitHub Actions logs or Pages deployments")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--token", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--actions-only", action="store_true")
    parser.add_argument("--pages-only", action="store_true")
    args = parser.parse_args(argv)
    validation_error = _validate_args(args)
    if validation_error:
        parser.error(validation_error)
    print(f"cleanup request recorded for {args.repo}; dry_run={args.dry_run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
