#!/usr/bin/env bash
set -euo pipefail

REPO="${GITHUB_REPO:-}"
TOKEN="${GITHUB_TOKEN:-}"
DRY_RUN=0
ACTIONS_ONLY=0
PAGES_ONLY=0

usage() {
	cat <<'EOF'
Usage: bash tools/github-cleanup.sh [--repo <owner/repo>] [--token <token>] [--dry-run] [--actions-only | --pages-only]

Validate a GitHub cleanup request for workflow logs or Pages deployments.
Environment fallbacks:
  GITHUB_REPO   default for --repo
  GITHUB_TOKEN  default for --token
EOF
}

while [[ $# -gt 0 ]]; do
	case "$1" in
		--repo)
			REPO="$2"
			shift 2
			;;
		--token)
			TOKEN="$2"
			shift 2
			;;
		--dry-run)
			DRY_RUN=1
			shift
			;;
		--actions-only)
			ACTIONS_ONLY=1
			shift
			;;
		--pages-only)
			PAGES_ONLY=1
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

if [[ -z "${REPO}" ]]; then
	echo "--repo is required (or set GITHUB_REPO)" >&2
	exit 1
fi

if [[ "${ACTIONS_ONLY}" == "1" && "${PAGES_ONLY}" == "1" ]]; then
	echo "--actions-only and --pages-only cannot be used together" >&2
	exit 2
fi

MODE="all"
if [[ "${ACTIONS_ONLY}" == "1" ]]; then
	MODE="actions"
elif [[ "${PAGES_ONLY}" == "1" ]]; then
	MODE="pages"
fi

if [[ -n "${TOKEN}" ]]; then
	TOKEN_SOURCE="provided"
else
	TOKEN_SOURCE="missing"
fi

printf 'cleanup request recorded for %s; dry_run=%s mode=%s token=%s\n' \
	"${REPO}" "${DRY_RUN}" "${MODE}" "${TOKEN_SOURCE}"