#!/usr/bin/env bash
set -euo pipefail

cd /workspace

if [[ ! -f pyproject.toml ]]; then
    echo "pyproject.toml not found under /workspace" >&2
    exit 2
fi

if [[ "${AUTO_SYNC:-1}" == "1" ]]; then
    sync_args=(--locked --extra "${UV_EXTRA:-cpu}")
    if [[ "${ENABLE_TE:-0}" == "1" ]]; then
        sync_args+=(--extra te)
    fi
    uv sync "${sync_args[@]}"
fi

exec "$@"
