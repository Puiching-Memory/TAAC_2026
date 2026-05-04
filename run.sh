#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="${TAAC_BUNDLE_WORKDIR:-${SCRIPT_DIR}/.taac_bundle}"
PROJECT_DIR="${SCRIPT_DIR}"
CODE_PACKAGE="${TAAC_CODE_PACKAGE:-${SCRIPT_DIR}/code_package.zip}"
BUNDLE_MODE=0

# Delegated to taac2026.infrastructure.platform.run_sh: TAAC_INSTALL_PROJECT_DEPS,
# TAAC_BUNDLE_PIP_EXTRAS, TAAC_PIP_EXTRAS, and the taac-train entrypoint.

case "${1:-}" in
	test)
		echo "run.sh no longer supports 'test'; use 'uv run pytest ...' directly" >&2
		exit 2
		;;
	package-infer)
		echo "unknown command: package-infer" >&2
		exit 2
		;;
	package)
		echo "unknown command: package" >&2
		exit 2
		;;
esac

find_python() {
	if [[ -n "${TAAC_PYTHON:-}" ]]; then
		command -v "${TAAC_PYTHON}"
		return $?
	fi
	if command -v python3 >/dev/null 2>&1; then
		command -v python3
		return
	fi
	if command -v python >/dev/null 2>&1; then
		command -v python
		return
	fi
	return 1
}

extract_code_package() {
	local package_path="$1"
	local target_dir="$2"
	local python_bin
	if python_bin="$(find_python)"; then
		"${python_bin}" - "${package_path}" "${target_dir}" <<'PY'
import sys
import zipfile

zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])
PY
		return
	fi
	if command -v unzip >/dev/null 2>&1; then
		unzip -q "${package_path}" -d "${target_dir}"
		return
	fi
	echo "python3, python, or unzip is required to unpack code_package.zip" >&2
	exit 127
}

if [[ -f "${CODE_PACKAGE}" ]]; then
	BUNDLE_MODE=1
	PROJECT_DIR="${WORKDIR}/project"
	if [[ "${TAAC_FORCE_EXTRACT:-0}" == "1" || ! -f "${PROJECT_DIR}/pyproject.toml" ]]; then
		rm -rf "${PROJECT_DIR}"
		mkdir -p "${WORKDIR}"
		extract_code_package "${CODE_PACKAGE}" "${WORKDIR}"
	fi
fi

if [[ ! -f "${PROJECT_DIR}/pyproject.toml" ]]; then
	echo "pyproject.toml not found. Upload run.sh together with code_package.zip, or run from the repository root." >&2
	exit 2
fi

export TAAC_PROJECT_DIR="${PROJECT_DIR}"
export TAAC_BUNDLE_MODE="${BUNDLE_MODE}"
export PYTHONPATH="${PROJECT_DIR}/src:${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ "${BUNDLE_MODE}" == "1" || "${TAAC_RUNNER:-}" == "python" ]]; then
	PYTHON_BIN="$(find_python)" || {
		echo "python3 or python is required to run the TAAC runtime" >&2
		exit 127
	}
	exec "${PYTHON_BIN}" -m taac2026.infrastructure.platform.run_sh "$@"
fi

if ! command -v uv >/dev/null 2>&1; then
	echo "uv is required but not found in PATH" >&2
	exit 127
fi

exec uv run python -m taac2026.infrastructure.platform.run_sh "$@"