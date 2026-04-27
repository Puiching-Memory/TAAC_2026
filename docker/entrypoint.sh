#!/usr/bin/env bash
set -euo pipefail

cd /workspace

if [[ ! -f pyproject.toml ]]; then
    echo "pyproject.toml not found under /workspace" >&2
    exit 2
fi

actual_uv_extra="${UV_EXTRA:-}"
expected_uv_extra="${EXPECTED_UV_EXTRA:-cuda128}"
if [[ "${actual_uv_extra}" != "${expected_uv_extra}" ]]; then
    echo "Unsupported UV_EXTRA: expected ${expected_uv_extra}, got ${actual_uv_extra:-<unset>}" >&2
    exit 64
fi

actual_python_version="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
expected_python_version="${EXPECTED_PYTHON_VERSION:-3.13}"
if [[ "${actual_python_version}" != "${expected_python_version}" ]]; then
    echo "Unsupported Python version: expected ${expected_python_version}, got ${actual_python_version}" >&2
    exit 65
fi

actual_base_image="${BASE_IMAGE:-${IMAGE_BASE_NAME:-}}"
expected_base_image="${EXPECTED_BASE_IMAGE:-nvidia/cuda:12.8.0-devel-ubuntu24.04}"
if [[ -n "${actual_base_image}" && "${actual_base_image}" != "${expected_base_image}" ]]; then
    echo "Unsupported base image: expected ${expected_base_image}, got ${actual_base_image}" >&2
    exit 66
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    cuda_version="$(nvidia-smi --query-gpu=cuda_version --format=csv,noheader | head -n 1 | tr -d '[:space:]')"
    if [[ -n "${cuda_version}" && "${cuda_version}" != 12.8* ]]; then
        echo "Unsupported CUDA runtime: expected 12.8.x, got ${cuda_version}" >&2
        exit 67
    fi
fi

if [[ "${AUTO_SYNC:-1}" == "1" ]]; then
    sync_args=(--locked --extra "${actual_uv_extra}")
    if [[ "${ENABLE_TE:-0}" == "1" ]]; then
        sync_args+=(--extra te)
    fi
    uv sync "${sync_args[@]}"
fi

exec "$@"
