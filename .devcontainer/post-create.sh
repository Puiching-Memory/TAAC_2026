#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT_DIR"

uv python install 3.13
uv sync --locked --python 3.13
uv run python scripts/verify_gpu_env.py --json
