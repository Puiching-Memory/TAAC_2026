#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

REPO_ROOT="$DEFAULT_REPO_ROOT"
REQUESTED_PROFILE="${TAAC_REQUESTED_PROFILE:-${TAAC_CUDA_PROFILE:-}}"
REQUESTED_PYTHON="${TAAC_REQUESTED_PYTHON:-}"
UV_INSTALL_URL="${TAAC_UV_INSTALL_URL:-https://astral.sh/uv/install.sh}"
PYPI_INDEX_URL="${TAAC_PYPI_INDEX_URL:-https://pypi.org/simple}"
PYTORCH_CPU_INDEX_URL="${TAAC_PYTORCH_CPU_INDEX_URL:-https://download.pytorch.org/whl/cpu}"
PYTORCH_CUDA126_INDEX_URL="${TAAC_PYTORCH_CUDA126_INDEX_URL:-https://download.pytorch.org/whl/cu126}"
PYTORCH_CUDA128_INDEX_URL="${TAAC_PYTORCH_CUDA128_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
PYTORCH_CUDA130_INDEX_URL="${TAAC_PYTORCH_CUDA130_INDEX_URL:-https://download.pytorch.org/whl/cu130}"
PROBE_TIMEOUT_SECONDS="${TAAC_NETWORK_PROBE_TIMEOUT:-10}"
PROBE_DETAIL_LIMIT="${TAAC_NETWORK_PROBE_DETAIL_LIMIT:-240}"
SITE_PROBE_TARGETS="${TAAC_SITE_PROBE_TARGETS:-example=https://example.com github=https://github.com python=https://www.python.org pypi=https://pypi.org/simple astral=https://astral.sh/uv/install.sh pytorch_cpu=https://download.pytorch.org/whl/cpu}"
ENABLE_PROXY_MATRIX="${TAAC_ENABLE_PROXY_MATRIX:-1}"
ENABLE_PIP_DOWNLOAD_PROBE="${TAAC_ENABLE_PIP_DOWNLOAD_PROBE:-1}"
PIP_DOWNLOAD_PACKAGE="${TAAC_PIP_DOWNLOAD_PACKAGE:-sampleproject==4.0.0}"
PIP_DOWNLOAD_INDEX_URL="${TAAC_PIP_DOWNLOAD_INDEX_URL:-$PYPI_INDEX_URL}"
OUTPUT_PATH=""

usage() {
    cat <<'EOF'
Usage: bash tools/log_host_device_info.sh [options]

Options:
  --repo-root PATH           Override repo root shown in the log.
  --requested-profile NAME   Record the requested runtime profile.
  --requested-python VER     Record the requested Python version.
    --uv-install-url URL       Override the uv installer URL probe target.
  --output PATH              Tee the log to PATH while printing to stdout.
  -h, --help                 Show this help message.

Examples:
  bash tools/log_host_device_info.sh --requested-profile cpu --requested-python 3.13
  bash tools/log_host_device_info.sh --output /tmp/host-device.log
EOF
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --repo-root)
                [[ $# -ge 2 ]] || {
                    echo "Missing value for $1" >&2
                    exit 2
                }
                REPO_ROOT="$2"
                shift 2
                ;;
            --requested-profile)
                [[ $# -ge 2 ]] || {
                    echo "Missing value for $1" >&2
                    exit 2
                }
                REQUESTED_PROFILE="$2"
                shift 2
                ;;
            --requested-python)
                [[ $# -ge 2 ]] || {
                    echo "Missing value for $1" >&2
                    exit 2
                }
                REQUESTED_PYTHON="$2"
                shift 2
                ;;
            --output)
                [[ $# -ge 2 ]] || {
                    echo "Missing value for $1" >&2
                    exit 2
                }
                OUTPUT_PATH="$2"
                shift 2
                ;;
            --uv-install-url)
                [[ $# -ge 2 ]] || {
                    echo "Missing value for $1" >&2
                    exit 2
                }
                UV_INSTALL_URL="$2"
                shift 2
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                echo "Unknown argument: $1" >&2
                usage >&2
                exit 2
                ;;
        esac
    done
}

timestamp() {
    date '+%Y-%m-%dT%H:%M:%S%z'
}

log_line() {
    printf '[%s] %s\n' "$(timestamp)" "$*"
}

sanitize_proxy_value() {
    local value="$1"
    local scheme=""
    local rest=""

    if [[ "$value" == *"://"* ]]; then
        scheme="${value%%://*}"
        rest="${value#*://}"
        if [[ "$rest" == *"@"* ]]; then
            printf '%s://***@%s' "$scheme" "${rest#*@}"
            return
        fi
    fi
    printf '%s' "$value"
}

start_capture() {
    if [[ -z "$OUTPUT_PATH" ]]; then
        return
    fi
    mkdir -p "$(dirname "$OUTPUT_PATH")"
    exec > >(tee -a "$OUTPUT_PATH") 2>&1
    log_line "device log: $OUTPUT_PATH"
}

run_logged_command() {
    local title="$1"
    shift
    local command_name="$1"
    if command -v "$command_name" >/dev/null 2>&1; then
        log_line "---- $title ----"
        "$@" || log_line "$title failed with exit code $?"
    else
        log_line "---- $title unavailable: $command_name not found ----"
    fi
}

log_os_release() {
    if [[ -r /etc/os-release ]]; then
        log_line "---- os-release ----"
        sed -n 's/^PRETTY_NAME=//p; s/^VERSION=//p' /etc/os-release | tr -d '"'
    else
        log_line "---- os-release unavailable ----"
    fi
}

log_network_info() {
    if command -v ip >/dev/null 2>&1; then
        log_line "---- network ----"
        ip -br addr || log_line "network failed with exit code $?"
    else
        log_line "---- network unavailable: ip not found ----"
    fi
}

log_proxy_environment() {
    log_line "---- proxy environment ----"

    local variable_name=""
    local variable_value=""
    for variable_name in HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY http_proxy https_proxy all_proxy no_proxy; do
        variable_value="${!variable_name-}"
        if [[ -n "$variable_value" ]]; then
            log_line "${variable_name}=$(sanitize_proxy_value "$variable_value")"
        else
            log_line "${variable_name}=<unset>"
        fi
    done
}

log_nvidia_device_nodes() {
    if compgen -G "/dev/nvidia*" >/dev/null; then
        log_line "---- nvidia device nodes ----"
        ls -l /dev/nvidia* || true
    else
        log_line "nvidia device nodes: none"
    fi
}

log_dri_nodes() {
    if [[ -e /dev/dri ]]; then
        log_line "---- /dev/dri ----"
        ls -l /dev/dri || true
    else
        log_line "/dev/dri: none"
    fi
}

log_python_info() {
    if command -v python3 >/dev/null 2>&1; then
        log_line "---- python3 ----"
        python3 --version
        python3 - <<'PY'
import os
import platform
import sys

print(f"python_executable={sys.executable}")
print(f"python_version={sys.version.replace(chr(10), ' ')}")
print(f"platform={platform.platform()}")
print(f"cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
print(f"nvidia_visible_devices={os.environ.get('NVIDIA_VISIBLE_DEVICES', '<unset>')}")
PY
    else
        log_line "---- python3 unavailable: python3 not found ----"
    fi
}

log_python_packages() {
    if ! command -v python3 >/dev/null 2>&1; then
        log_line "---- python packages unavailable: python3 not found ----"
        return
    fi

    log_line "---- python packages ----"
    python3 - <<'PY'
from importlib import metadata

packages = []
for dist in metadata.distributions():
    name = dist.metadata.get("Name") or dist.metadata.get("Summary") or dist.metadata.get("name") or dist.name
    version = dist.version
    packages.append((str(name), str(version)))

packages.sort(key=lambda item: item[0].lower())
print(f"installed_python_packages={len(packages)}")
for name, version in packages:
    print(f"{name}=={version}")
PY
}

compact_probe_detail() {
    local detail="$1"

    detail="$(printf '%s' "$detail" | tr '\r\n' '  ' | sed 's/[[:space:]]\+/ /g; s/^ //; s/ $//')"
    if (( ${#detail} > PROBE_DETAIL_LIMIT )); then
        detail="${detail:0:PROBE_DETAIL_LIMIT-3}..."
    fi
    printf '%s' "$detail"
}

run_with_proxy_mode() {
    local proxy_mode="$1"
    shift

    case "$proxy_mode" in
        inherited)
            "$@"
            ;;
        no_proxy)
            env \
                -u HTTP_PROXY \
                -u HTTPS_PROXY \
                -u ALL_PROXY \
                -u NO_PROXY \
                -u http_proxy \
                -u https_proxy \
                -u all_proxy \
                -u no_proxy \
                "$@"
            ;;
        *)
            printf 'Unsupported proxy mode: %s\n' "$proxy_mode" >&2
            return 2
            ;;
    esac
}

url_host() {
    local url="$1"
    local without_scheme="${url#*://}"

    without_scheme="${without_scheme%%/*}"
    without_scheme="${without_scheme%%\?*}"
    without_scheme="${without_scheme%%#*}"
    without_scheme="${without_scheme%%:*}"
    printf '%s' "$without_scheme"
}

log_dns_probe() {
    local label="$1"
    local host="$2"

    [[ -n "$host" ]] || return
    if command -v getent >/dev/null 2>&1; then
        local dns_output=""
        local dns_status=0
        dns_output="$(getent hosts "$host" 2>&1)" || dns_status=$?
        if [[ $dns_status -eq 0 ]]; then
            log_line "${label}_dns=resolved"
            log_line "${label}_dns_detail=$(compact_probe_detail "$dns_output")"
        else
            local dns_detail=""

            log_line "${label}_dns=failed"
            log_line "${label}_dns_exit_code=$dns_status"
            dns_detail="$(compact_probe_detail "$dns_output")"
            if [[ -n "$dns_detail" ]]; then
                log_line "${label}_dns_detail=$dns_detail"
            fi
        fi
        return
    fi

    log_line "${label}_dns=unavailable"
}

classify_curl_probe_failure() {
    local exit_code="$1"
    local probe_detail="$2"

    if printf '%s\n' "$probe_detail" | grep -Eiq 'proxy tunneling failed|connect tunnel failed|received http code [0-9]+ from proxy|proxy error|service unavailable'; then
        printf 'proxy_tunnel_failure'
        return
    fi
    if printf '%s\n' "$probe_detail" | grep -Eiq 'temporary failure in name resolution|name or service not known|no address associated'; then
        printf 'dns_failure'
        return
    fi
    if printf '%s\n' "$probe_detail" | grep -Eiq 'certificate verify failed|ssl|tls'; then
        printf 'tls_failure'
        return
    fi

    case "$exit_code" in
        5|6)
            printf 'dns_failure'
            ;;
        7)
            printf 'connect_failure'
            ;;
        28)
            printf 'timeout'
            ;;
        35|51|58|60|77|83|90|91|92)
            printf 'tls_failure'
            ;;
        52|55|56)
            printf 'transport_failure'
            ;;
        *)
            printf 'unknown_failure'
            ;;
    esac
}

classify_wget_probe_failure() {
    local exit_code="$1"
    local probe_detail="$2"

    if printf '%s\n' "$probe_detail" | grep -Eiq 'proxy tunneling failed|connect tunnel failed|received http code [0-9]+ from proxy|proxy error|service unavailable'; then
        printf 'proxy_tunnel_failure'
        return
    fi
    if printf '%s\n' "$probe_detail" | grep -Eiq 'temporary failure in name resolution|name or service not known|no address associated'; then
        printf 'dns_failure'
        return
    fi
    if printf '%s\n' "$probe_detail" | grep -Eiq 'certificate verify failed|ssl|tls'; then
        printf 'tls_failure'
        return
    fi

    case "$exit_code" in
        4)
            printf 'network_failure'
            ;;
        5)
            printf 'tls_failure'
            ;;
        6)
            printf 'auth_failure'
            ;;
        7)
            printf 'protocol_failure'
            ;;
        8)
            printf 'http_error'
            ;;
        *)
            printf 'unknown_failure'
            ;;
    esac
}

log_failed_url_probe() {
    local label="$1"
    local host="$2"
    local exit_code="$3"
    local failure_class="$4"
    local http_code="$5"
    local probe_detail="$6"

    log_line "${label}_probe=failed"
    log_line "${label}_probe_exit_code=$exit_code"
    log_line "${label}_failure_class=$failure_class"
    if [[ -n "$http_code" ]]; then
        log_line "${label}_http_code=$http_code"
    fi
    probe_detail="$(compact_probe_detail "$probe_detail")"
    if [[ -n "$probe_detail" ]]; then
        log_line "${label}_probe_detail=$probe_detail"
    fi
    log_dns_probe "$label" "$host"
}

log_url_probe_with_mode() {
    local label="$1"
    local url="$2"
    local proxy_mode="$3"
    local host=""

    log_line "${label}_url=$url"
    host="$(url_host "$url")"
    if [[ -n "$host" ]]; then
        log_line "${label}_host=$host"
    fi
    log_line "${label}_proxy_mode=$proxy_mode"
    if command -v curl >/dev/null 2>&1; then
        log_line "${label}_probe_tool=curl"
        local probe_output=""
        local probe_status=0
        local http_code=""
        local probe_detail=""

        probe_output="$(run_with_proxy_mode "$proxy_mode" curl -I -L -sS --max-time "$PROBE_TIMEOUT_SECONDS" -o /dev/null -w $'\n__HTTP_CODE__=%{http_code}' "$url" 2>&1)" || probe_status=$?
        http_code="$(printf '%s\n' "$probe_output" | sed -n 's/^__HTTP_CODE__=//p' | tail -n 1)"
        probe_detail="$(printf '%s\n' "$probe_output" | sed '/^__HTTP_CODE__=/d')"
        if [[ $probe_status -eq 0 && -n "$http_code" && "$http_code" != "000" ]]; then
            log_line "${label}_probe=reachable"
            log_line "${label}_http_code=$http_code"
        else
            log_failed_url_probe "$label" "$host" "$probe_status" "$(classify_curl_probe_failure "$probe_status" "$probe_detail")" "$http_code" "$probe_detail"
        fi
        return
    fi

    if command -v wget >/dev/null 2>&1; then
        log_line "${label}_probe_tool=wget"
        local probe_output=""
        local probe_status=0
        local http_code=""

        probe_output="$(run_with_proxy_mode "$proxy_mode" wget --server-response --spider --timeout="$PROBE_TIMEOUT_SECONDS" --tries=1 "$url" -O /dev/null 2>&1)" || probe_status=$?
        http_code="$(printf '%s\n' "$probe_output" | awk '/^[[:space:]]*HTTP\// { code=$2 } END { if (code != "") print code }')"
        if [[ $probe_status -eq 0 ]]; then
            log_line "${label}_probe=reachable"
            if [[ -n "$http_code" ]]; then
                log_line "${label}_http_code=$http_code"
            fi
        else
            log_failed_url_probe "$label" "$host" "$probe_status" "$(classify_wget_probe_failure "$probe_status" "$probe_output")" "$http_code" "$probe_output"
        fi
        return
    fi

    log_line "${label}_probe_tool=none"
    log_line "${label}_probe=unavailable"
}

log_url_probe() {
    local label="$1"
    local url="$2"

    log_url_probe_with_mode "$label" "$url" inherited
}

log_dual_mode_url_probe() {
    local label="$1"
    local url="$2"

    log_url_probe_with_mode "${label}_inherited" "$url" inherited
    log_url_probe_with_mode "${label}_no_proxy" "$url" no_proxy
}

log_connectivity_matrix() {
    [[ "$ENABLE_PROXY_MATRIX" == "1" ]] || return

    log_line "---- connectivity matrix ----"

    local target_entry=""
    local target_name=""
    local target_url=""
    local raw_targets="${SITE_PROBE_TARGETS//,/ }"

    for target_entry in $raw_targets; do
        target_name="${target_entry%%=*}"
        target_url="${target_entry#*=}"
        if [[ -z "$target_name" || -z "$target_url" || "$target_name" == "$target_entry" ]]; then
            log_line "site_probe_target_invalid=$target_entry"
            continue
        fi
        log_dual_mode_url_probe "site_${target_name}" "$target_url"
    done
}

classify_pip_download_failure() {
    local exit_code="$1"
    local probe_detail="$2"

    if printf '%s\n' "$probe_detail" | grep -Eiq 'proxy tunneling failed|tunnel connection failed|received http code [0-9]+ from proxy|proxyerror|proxy error|service unavailable'; then
        printf 'proxy_tunnel_failure'
        return
    fi
    if printf '%s\n' "$probe_detail" | grep -Eiq 'temporary failure in name resolution|name or service not known|no address associated'; then
        printf 'dns_failure'
        return
    fi
    if printf '%s\n' "$probe_detail" | grep -Eiq 'certificate verify failed|ssl|tls'; then
        printf 'tls_failure'
        return
    fi
    if printf '%s\n' "$probe_detail" | grep -Eiq 'timed out|read timed out|connect timeout'; then
        printf 'timeout'
        return
    fi
    if printf '%s\n' "$probe_detail" | grep -Eiq 'no matching distribution found|could not find a version that satisfies the requirement'; then
        printf 'package_resolution_failure'
        return
    fi
    if printf '%s\n' "$probe_detail" | grep -Eiq 'connection refused|connection error|failed to establish a new connection'; then
        printf 'connect_failure'
        return
    fi
    case "$exit_code" in
        0)
            printf 'success'
            ;;
        *)
            printf 'unknown_failure'
            ;;
    esac
}

log_pip_download_probe_with_mode() {
    local label="$1"
    local proxy_mode="$2"
    local pip_executable=""
    local probe_status=0
    local probe_output=""
    local temp_dir=""

    if command -v pip >/dev/null 2>&1; then
        pip_executable="pip"
    elif command -v pip3 >/dev/null 2>&1; then
        pip_executable="pip3"
    else
        log_line "${label}_probe=unavailable"
        log_line "${label}_probe_detail=pip executable not found"
        return
    fi

    log_line "${label}_package=$PIP_DOWNLOAD_PACKAGE"
    log_line "${label}_index_url=$PIP_DOWNLOAD_INDEX_URL"
    log_line "${label}_tool=$pip_executable"
    log_line "${label}_proxy_mode=$proxy_mode"

    temp_dir="$(mktemp -d 2>/dev/null || mktemp -d -t taac-pip-probe)"
    probe_output="$(run_with_proxy_mode "$proxy_mode" "$pip_executable" download --disable-pip-version-check --no-cache-dir --no-deps --retries 0 --timeout "$PROBE_TIMEOUT_SECONDS" --dest "$temp_dir" --index-url "$PIP_DOWNLOAD_INDEX_URL" "$PIP_DOWNLOAD_PACKAGE" 2>&1)" || probe_status=$?
    rm -rf "$temp_dir"

    if [[ $probe_status -eq 0 ]]; then
        log_line "${label}_probe=reachable"
        return
    fi

    log_line "${label}_probe=failed"
    log_line "${label}_probe_exit_code=$probe_status"
    log_line "${label}_failure_class=$(classify_pip_download_failure "$probe_status" "$probe_output")"
    log_line "${label}_probe_detail=$(compact_probe_detail "$probe_output")"
}

log_pip_download_probes() {
    [[ "$ENABLE_PIP_DOWNLOAD_PROBE" == "1" ]] || return

    log_line "---- pip download probes ----"
    log_pip_download_probe_with_mode "pip_download_inherited" inherited
    log_pip_download_probe_with_mode "pip_download_no_proxy" no_proxy
}

pytorch_index_url_for_profile() {
    case "$1" in
        cpu)
            printf '%s' "$PYTORCH_CPU_INDEX_URL"
            ;;
        cuda126)
            printf '%s' "$PYTORCH_CUDA126_INDEX_URL"
            ;;
        cuda128)
            printf '%s' "$PYTORCH_CUDA128_INDEX_URL"
            ;;
        cuda130)
            printf '%s' "$PYTORCH_CUDA130_INDEX_URL"
            ;;
        *)
            return 1
            ;;
    esac
}

log_uv_bootstrap_status() {
    log_line "---- uv bootstrap ----"
    log_line "uv_install_url=$UV_INSTALL_URL"
    if command -v uv >/dev/null 2>&1; then
        log_line "uv_present=1"
        uv --version
    else
        log_line "uv_present=0"
    fi

    log_url_probe "uv_download" "$UV_INSTALL_URL"
}

log_dependency_index_status() {
    log_line "---- dependency indexes ----"
    log_url_probe "pypi_index" "$PYPI_INDEX_URL"

    if [[ -n "$REQUESTED_PROFILE" ]]; then
        local requested_url
        if requested_url="$(pytorch_index_url_for_profile "$REQUESTED_PROFILE")"; then
            log_line "pytorch_probe_profile=$REQUESTED_PROFILE"
            log_url_probe "pytorch_index_${REQUESTED_PROFILE}" "$requested_url"
        else
            log_line "pytorch_probe_profile=$REQUESTED_PROFILE"
            log_line "pytorch_probe=unsupported-profile"
        fi
        return
    fi

    log_line "pytorch_probe_profile=all"
    local profile
    local profile_url
    for profile in cpu cuda126 cuda128 cuda130; do
        profile_url="$(pytorch_index_url_for_profile "$profile")"
        log_url_probe "pytorch_index_${profile}" "$profile_url"
    done
}

log_build_tools() {
    log_line "---- build tools ----"
    local tool
    for tool in gcc g++ make cmake ninja pkg-config cc c++; do
        if command -v "$tool" >/dev/null 2>&1; then
            log_line "$tool=present"
            "$tool" --version || log_line "$tool --version failed with exit code $?"
        else
            log_line "$tool=missing"
        fi
    done
}

main() {
    parse_args "$@"
    start_capture

    log_line "==== Host and device information ===="
    log_line "repo_root=$REPO_ROOT"
    if [[ -n "$REQUESTED_PROFILE" ]]; then
        log_line "requested_profile=$REQUESTED_PROFILE"
    fi
    if [[ -n "$REQUESTED_PYTHON" ]]; then
        log_line "requested_python=$REQUESTED_PYTHON"
    fi
    log_line "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
    log_line "NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-<unset>}"

    log_os_release
    log_proxy_environment
    run_logged_command "hostname" hostname
    run_logged_command "uptime" uptime
    run_logged_command "kernel" uname -a
    run_logged_command "cpu" lscpu
    run_logged_command "memory" free -h
    run_logged_command "block devices" lsblk -o NAME,SIZE,TYPE,MOUNTPOINT,MODEL
    run_logged_command "disk usage" df -h "$REPO_ROOT" /tmp
    log_network_info
    log_nvidia_device_nodes
    log_dri_nodes
    run_logged_command "nvidia-smi list" nvidia-smi -L
    run_logged_command "nvidia-smi" nvidia-smi
    run_logged_command "nvcc" nvcc --version
    run_logged_command "uv" uv --version
    log_uv_bootstrap_status
    log_dependency_index_status
    log_connectivity_matrix
    log_pip_download_probes
    log_build_tools
    log_python_info
    log_python_packages
}

main "$@"