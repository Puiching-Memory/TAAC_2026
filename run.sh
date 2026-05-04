#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="${TAAC_BUNDLE_WORKDIR:-${SCRIPT_DIR}/.taac_bundle}"
PROJECT_DIR="${SCRIPT_DIR}"
CODE_PACKAGE="${TAAC_CODE_PACKAGE:-${SCRIPT_DIR}/code_package.zip}"
BUNDLE_MODE=0
SUPPORTED_CUDA_PROFILE="cuda126"
TENCENT_PYPI_INDEX_URL="https://mirrors.cloud.tencent.com/pypi/simple/"
PROJECT_PIP_DEPENDENCIES_READY=0

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

read_bundled_experiment() {
	local manifest_path="$1"
	local python_bin
	if python_bin="$(find_python)"; then
		"${python_bin}" - "${manifest_path}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    manifest = json.load(handle)
print(manifest.get("bundled_experiment_path", ""))
PY
		return
	fi
	sed -n 's/.*"bundled_experiment_path"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' "${manifest_path}" | head -n 1
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

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}/src:${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

RUNNER_MODE="${TAAC_RUNNER:-}"
if [[ -z "${RUNNER_MODE}" ]]; then
	if [[ "${BUNDLE_MODE}" == "1" ]]; then
		RUNNER_MODE="python"
	else
		RUNNER_MODE="uv"
	fi
fi

MANIFEST_EXPERIMENT=""
MANIFEST_PATH="${PROJECT_DIR}/.taac_training_manifest.json"
if [[ -f "${MANIFEST_PATH}" ]]; then
	MANIFEST_EXPERIMENT="$(read_bundled_experiment "${MANIFEST_PATH}")"
fi

ensure_uv() {
	if command -v uv >/dev/null 2>&1; then
		return
	fi
	echo "uv is required but not found in PATH" >&2
	exit 127
}

ensure_python() {
	local python_bin
	if python_bin="$(find_python)"; then
		PYTHON_BIN="${python_bin}"
		return
	fi
	echo "python3 or python is required to run the training code" >&2
	exit 127
}

should_install_project_pip_dependencies() {
	if [[ "${TAAC_SKIP_PIP_INSTALL:-0}" == "1" ]]; then
		return 1
	fi
	if [[ -n "${TAAC_INSTALL_PROJECT_DEPS:-}" ]]; then
		[[ "${TAAC_INSTALL_PROJECT_DEPS}" != "0" ]]
		return
	fi
	[[ "${BUNDLE_MODE}" == "1" ]]
}

ensure_project_pip_dependencies() {
	if [[ "${RUNNER_MODE}" != "python" ]]; then
		return
	fi
	if [[ "${PROJECT_PIP_DEPENDENCIES_READY}" == "1" ]]; then
		return
	fi
	if ! should_install_project_pip_dependencies; then
		PROJECT_PIP_DEPENDENCIES_READY=1
		return
	fi

	ensure_python
	TAAC_DEFAULT_PIP_INDEX_URL="${TENCENT_PYPI_INDEX_URL}" \
	TAAC_PIP_EXTRAS_ENV_NAME="$([[ "${BUNDLE_MODE}" == "1" ]] && printf '%s' "TAAC_BUNDLE_PIP_EXTRAS" || printf '%s' "TAAC_PIP_EXTRAS")" \
	"${PYTHON_BIN}" - <<'PY'
import os
import shlex
import subprocess
import sys

extra_args = shlex.split(os.environ.get("TAAC_PIP_EXTRA_ARGS", ""))
extras = shlex.split(
    os.environ.get(os.environ.get("TAAC_PIP_EXTRAS_ENV_NAME", "TAAC_PIP_EXTRAS"), "")
)
target = "."
if extras:
    target = f".[{','.join(extras)}]"

command = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check"]
index_url = os.environ.get(
    "TAAC_PIP_INDEX_URL",
    os.environ.get("TAAC_DEFAULT_PIP_INDEX_URL", "https://mirrors.cloud.tencent.com/pypi/simple/"),
)
if index_url:
    command.extend(["-i", index_url])
command.extend(extra_args)
command.append(target)

print("Installing TAAC project dependencies from pyproject.toml", file=sys.stderr)
subprocess.check_call(command)
PY
	PROJECT_PIP_DEPENDENCIES_READY=1
}

extract_cuda_profile() {
	local default_profile="$1"
	shift
	CUDA_PROFILE="${TAAC_CUDA_PROFILE:-${default_profile}}"
	REMAINING_ARGS=()
	while [[ $# -gt 0 ]]; do
		case "$1" in
			--cuda-profile)
				CUDA_PROFILE="$2"
				shift 2
				;;
			--cuda-profile=*)
				CUDA_PROFILE="${1#--cuda-profile=}"
				shift
				;;
			*)
				REMAINING_ARGS+=("$1")
				shift
				;;
		esac
	done
	if [[ "${CUDA_PROFILE}" != "${SUPPORTED_CUDA_PROFILE}" ]]; then
		echo "unsupported TAAC_CUDA_PROFILE/--cuda-profile: ${CUDA_PROFILE}; only '${SUPPORTED_CUDA_PROFILE}' is supported" >&2
		exit 2
	fi
}

sync_runtime() {
	local profile="$1"
	shift
	if [[ "${RUNNER_MODE}" != "uv" ]]; then
		return
	fi
	if [[ "${TAAC_SKIP_UV_SYNC:-0}" == "1" ]]; then
		return
	fi
	ensure_uv
	uv sync --locked --extra "${profile}" "$@"
}

run_python_module() {
	local module_name="$1"
	shift
	ensure_python
	"${PYTHON_BIN}" -m "${module_name}" "$@"
}

run_console_script() {
	local script_name="$1"
	local module_name="$2"
	shift 2
	case "${RUNNER_MODE}" in
		uv)
			ensure_uv
			uv run "${script_name}" "$@"
			;;
		python)
			ensure_project_pip_dependencies
			run_python_module "${module_name}" "$@"
			;;
		*)
			echo "unsupported TAAC_RUNNER: ${RUNNER_MODE}; expected 'python' or 'uv'" >&2
			exit 2
			;;
	esac
}

COMMAND="train"
if [[ $# -gt 0 ]]; then
	case "$1" in
		train|val|eval|infer)
			COMMAND="$1"
			shift
			;;
		--*)
			COMMAND="train"
			;;
		test)
			echo "run.sh no longer supports 'test'; use 'uv run pytest ...' directly" >&2
			exit 2
			;;
		*)
			COMMAND="$1"
			shift
			;;
	esac
fi

case "${COMMAND}" in
	train)
		extract_cuda_profile "${SUPPORTED_CUDA_PROFILE}" "$@"
		sync_runtime "${CUDA_PROFILE}"
		TRAIN_ARGS=()
		if [[ -n "${TAAC_EXPERIMENT:-}" ]]; then
			TRAIN_ARGS+=(--experiment "${TAAC_EXPERIMENT}")
		elif [[ -n "${MANIFEST_EXPERIMENT}" ]]; then
			TRAIN_ARGS+=(--experiment "${MANIFEST_EXPERIMENT}")
		fi
		DATASET_PATH="${TRAIN_DATA_PATH:-}"
		SCHEMA_PATH="${TAAC_SCHEMA_PATH:-}"
		OUTPUT_DIR="${TRAIN_CKPT_PATH:-}"
		if [[ -n "${DATASET_PATH}" ]]; then
			TRAIN_ARGS+=(--dataset-path "${DATASET_PATH}")
		fi
		if [[ -n "${SCHEMA_PATH}" ]]; then
			TRAIN_ARGS+=(--schema-path "${SCHEMA_PATH}")
		fi
		if [[ -n "${OUTPUT_DIR}" ]]; then
			TRAIN_ARGS+=(--run-dir "${OUTPUT_DIR}")
		fi
		run_console_script taac-train taac2026.application.training.cli "${TRAIN_ARGS[@]}" "${REMAINING_ARGS[@]}"
		;;
	val|eval)
		extract_cuda_profile "${SUPPORTED_CUDA_PROFILE}" "$@"
		sync_runtime "${CUDA_PROFILE}"
		EVAL_ARGS=(single)
		if [[ -n "${TAAC_EXPERIMENT:-}" ]]; then
			EVAL_ARGS+=(--experiment "${TAAC_EXPERIMENT}")
		elif [[ -n "${MANIFEST_EXPERIMENT}" ]]; then
			EVAL_ARGS+=(--experiment "${MANIFEST_EXPERIMENT}")
		fi
		DATASET_PATH="${TRAIN_DATA_PATH:-}"
		SCHEMA_PATH="${TAAC_SCHEMA_PATH:-}"
		OUTPUT_DIR="${TRAIN_CKPT_PATH:-}"
		if [[ -n "${DATASET_PATH}" ]]; then
			EVAL_ARGS+=(--dataset-path "${DATASET_PATH}")
		fi
		if [[ -n "${SCHEMA_PATH}" ]]; then
			EVAL_ARGS+=(--schema-path "${SCHEMA_PATH}")
		fi
		if [[ -n "${OUTPUT_DIR}" ]]; then
			EVAL_ARGS+=(--run-dir "${OUTPUT_DIR}")
		fi
		run_console_script taac-evaluate taac2026.application.evaluation.cli "${EVAL_ARGS[@]}" "${REMAINING_ARGS[@]}"
		;;
	infer)
		extract_cuda_profile "${SUPPORTED_CUDA_PROFILE}" "$@"
		sync_runtime "${CUDA_PROFILE}"
		INFER_ARGS=(infer)
		if [[ -n "${TAAC_EXPERIMENT:-}" ]]; then
			INFER_ARGS+=(--experiment "${TAAC_EXPERIMENT}")
		elif [[ -n "${MANIFEST_EXPERIMENT}" ]]; then
			INFER_ARGS+=(--experiment "${MANIFEST_EXPERIMENT}")
		fi
		DATASET_PATH="${EVAL_DATA_PATH:-}"
		SCHEMA_PATH="${TAAC_SCHEMA_PATH:-}"
		RESULT_DIR="${EVAL_RESULT_PATH:-}"
		CHECKPOINT_PATH="${MODEL_OUTPUT_PATH:-}"
		if [[ -n "${DATASET_PATH}" ]]; then
			INFER_ARGS+=(--dataset-path "${DATASET_PATH}")
		fi
		if [[ -n "${SCHEMA_PATH}" ]]; then
			INFER_ARGS+=(--schema-path "${SCHEMA_PATH}")
		fi
		if [[ -n "${RESULT_DIR}" ]]; then
			INFER_ARGS+=(--result-dir "${RESULT_DIR}")
		fi
		if [[ -n "${CHECKPOINT_PATH}" ]]; then
			INFER_ARGS+=(--checkpoint "${CHECKPOINT_PATH}")
		fi
		run_console_script taac-evaluate taac2026.application.evaluation.cli "${INFER_ARGS[@]}" "${REMAINING_ARGS[@]}"
		;;
	*)
		echo "unknown command: ${COMMAND}" >&2
		exit 2
		;;
esac
