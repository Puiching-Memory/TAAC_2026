#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
TARGET_ROOT="${ROOT_DIR}"
DRY_RUN=0
INCLUDE_ENV_DIRS=0

usage() {
	cat <<'EOF'
Usage: bash tools/cache-cleanup.sh [--root <path>] [--dry-run] [--include-env-dirs]

Remove __pycache__ directories and common build artifacts from the repository.
EOF
}

while [[ $# -gt 0 ]]; do
	case "$1" in
		--root)
			TARGET_ROOT="$2"
			shift 2
			;;
		--dry-run)
			DRY_RUN=1
			shift
			;;
		--include-env-dirs)
			INCLUDE_ENV_DIRS=1
			shift
			;;
		-h|--help)
			usage
			exit 0
			;;
		*)
			echo "unknown argument: $1" >&2
			usage >&2
			exit 2
			;;
	esac
done

TARGET_ROOT="$(cd "${TARGET_ROOT}" && pwd)"
ENV_DIR_PATTERN='/(\.venv|venv|env|node_modules|\.tox|\.mypy_cache)/'

mapfile -t PYCACHE_DIRS < <(
	find "${TARGET_ROOT}" -type d -name '__pycache__' | sort | {
		if [[ "${INCLUDE_ENV_DIRS}" == "1" ]]; then
			cat
		else
			grep -Ev "${ENV_DIR_PATTERN}" || true
		fi
	}
)

BUILD_TARGETS=(
	"${TARGET_ROOT}/build"
	"${TARGET_ROOT}/dist"
)

mapfile -t EGG_INFO_DIRS < <(find "${TARGET_ROOT}" -type d -name '*.egg-info' | sort)

TARGETS=()
for path in "${PYCACHE_DIRS[@]}" "${EGG_INFO_DIRS[@]}"; do
	if [[ -n "${path}" ]]; then
		TARGETS+=("${path}")
	fi
done
for path in "${BUILD_TARGETS[@]}"; do
	if [[ -e "${path}" ]]; then
		TARGETS+=("${path}")
	fi
done

FILES=0
BYTES=0
FAILURES=0

for target in "${TARGETS[@]}"; do
	while IFS= read -r file_path; do
		[[ -z "${file_path}" ]] && continue
		((FILES += 1))
		if size=$(stat -c '%s' "${file_path}" 2>/dev/null); then
			((BYTES += size))
		fi
	done < <(find "${target}" -type f 2>/dev/null || true)
	done

printf 'root=%s dirs=%s files=%s bytes=%s dry_run=%s include_env_dirs=%s\n' \
	"${TARGET_ROOT}" "${#TARGETS[@]}" "${FILES}" "${BYTES}" "${DRY_RUN}" "${INCLUDE_ENV_DIRS}"

for target in "${TARGETS[@]}"; do
	printf '%s\n' "${target}"
	if [[ "${DRY_RUN}" == "1" ]]; then
		continue
	fi
	if ! rm -rf "${target}"; then
		printf 'failed: %s\n' "${target}" >&2
		FAILURES=1
	fi
	done

exit "${FAILURES}"