"""Collect host and device diagnostics without relying on shell wrapper scripts."""

from __future__ import annotations

import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from importlib import metadata
from pathlib import Path


DEFAULT_UV_INSTALL_URL = "https://astral.sh/uv/install.sh"
DEFAULT_PYPI_INDEX_URL = "https://pypi.org/simple"
DEFAULT_TENCENT_PYPI_INDEX_URL = "https://mirrors.cloud.tencent.com/pypi/simple/"
DEFAULT_PYTORCH_CPU_INDEX_URL = "https://download.pytorch.org/whl/cpu"
DEFAULT_PYTORCH_CUDA126_INDEX_URL = "https://download.pytorch.org/whl/cu126"
DEFAULT_CONDA_SUBDIR = "linux-64"
DEFAULT_CONDA_MAIN_CHANNEL_BASE_URL = "https://repo.anaconda.com/pkgs/main"
DEFAULT_CONDA_FORGE_CHANNEL_BASE_URL = "https://conda.anaconda.org/conda-forge"
DEFAULT_TENCENT_CONDA_MAIN_CHANNEL_URL = "http://mirrors.cloud.tencent.com/anaconda/pkgs/main/"
DEFAULT_TENCENT_CONDA_FREE_CHANNEL_URL = "http://mirrors.cloud.tencent.com/anaconda/pkgs/free/"
DEFAULT_PIP_DOWNLOAD_PACKAGE = "sampleproject==4.0.0"
DEFAULT_CONDA_PROBE_SPEC = "python=3.10"
DEFAULT_SITE_PROBE_TARGETS: dict[str, str] = {
    "example": "https://example.com",
    "github": "https://github.com",
    "python": "https://www.python.org",
}
DEFAULT_PROXY_VARIABLES = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
)
DEFAULT_BUILD_TOOLS = ("gcc", "g++", "make", "cmake", "ninja", "pkg-config", "cc", "c++")
TIMEOUT_EXIT_CODE = 124


@dataclass(slots=True)
class HostDeviceInfoConfig:
    repo_root: Path
    requested_profile: str | None = None
    requested_python: str | None = None
    uv_install_url: str = DEFAULT_UV_INSTALL_URL
    pypi_index_url: str = DEFAULT_PYPI_INDEX_URL
    tencent_pypi_index_url: str = DEFAULT_TENCENT_PYPI_INDEX_URL
    pytorch_cpu_index_url: str = DEFAULT_PYTORCH_CPU_INDEX_URL
    pytorch_cuda126_index_url: str = DEFAULT_PYTORCH_CUDA126_INDEX_URL
    conda_subdir: str = DEFAULT_CONDA_SUBDIR
    conda_main_channel_base_url: str = DEFAULT_CONDA_MAIN_CHANNEL_BASE_URL
    conda_forge_channel_base_url: str = DEFAULT_CONDA_FORGE_CHANNEL_BASE_URL
    tencent_conda_main_channel_url: str = DEFAULT_TENCENT_CONDA_MAIN_CHANNEL_URL
    tencent_conda_free_channel_url: str = DEFAULT_TENCENT_CONDA_FREE_CHANNEL_URL
    probe_timeout_seconds: int = 10
    probe_detail_limit: int = 240
    enable_proxy_matrix: bool = True
    enable_pip_download_probe: bool = True
    pip_download_package: str = DEFAULT_PIP_DOWNLOAD_PACKAGE
    pip_download_index_url: str = DEFAULT_PYPI_INDEX_URL
    enable_conda_search_probe: bool = True
    conda_search_channel_url: str = DEFAULT_CONDA_FORGE_CHANNEL_BASE_URL
    conda_probe_spec: str = DEFAULT_CONDA_PROBE_SPEC
    site_probe_targets: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_SITE_PROBE_TARGETS))

    @property
    def conda_main_channel_url(self) -> str:
        return f"{self.conda_main_channel_base_url}/{self.conda_subdir}/repodata.json"

    @property
    def conda_forge_channel_url(self) -> str:
        return f"{self.conda_forge_channel_base_url}/{self.conda_subdir}/repodata.json"


class LogSink:
    def close(self) -> None:
        return None

    def log(self, message: str) -> None:
        line = f"[{datetime.now().astimezone().strftime('%Y-%m-%dT%H:%M:%S%z')}] {message}"
        print(line)


def _sanitize_proxy_value(value: str) -> str:
    if "://" not in value:
        return value
    scheme, rest = value.split("://", 1)
    if "@" not in rest:
        return value
    return f"{scheme}://***@{rest.split('@', 1)[1]}"


def _compact_detail(detail: str, limit: int) -> str:
    normalized = " ".join(detail.replace("\r", " ").replace("\n", " ").split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(limit - 3, 0)] + "..."


def _command_env(proxy_mode: str) -> Mapping[str, str]:
    env = dict(os.environ)
    if proxy_mode == "no_proxy":
        for variable_name in DEFAULT_PROXY_VARIABLES:
            env.pop(variable_name, None)
    return env


def _run_command(
    command: Sequence[str],
    *,
    proxy_mode: str = "inherited",
    timeout: int | None = None,
    cwd: Path | None = None,
) -> tuple[int, str]:
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            cwd=cwd,
            env=_command_env(proxy_mode),
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        combined = "\n".join(
            part for part in (error.stdout, error.stderr, str(error)) if part
        )
        return TIMEOUT_EXIT_CODE, combined
    combined = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
    return completed.returncode, combined


def _log_command(sink: LogSink, title: str, command: Sequence[str], *, timeout: int | None = None) -> None:
    executable = shutil.which(command[0])
    if executable is None:
        sink.log(f"---- {title} unavailable: {command[0]} not found ----")
        return
    sink.log(f"---- {title} ----")
    return_code, output = _run_command(command, timeout=timeout)
    if output:
        for line in output.splitlines():
            sink.log(line)
    if return_code != 0:
        sink.log(f"{title} failed with exit code {return_code}")


def _url_host(url: str) -> str:
    return urllib.parse.urlsplit(url).hostname or ""


def _classify_url_failure(message: str) -> str:
    lowered = message.lower()
    if any(token in lowered for token in ("proxy", "tunnel connection failed", "proxyerror")):
        return "proxy_tunnel_failure"
    if any(token in lowered for token in ("name or service not known", "temporary failure in name resolution", "nodename nor servname")):
        return "dns_failure"
    if any(token in lowered for token in ("certificate", "ssl", "tls")):
        return "tls_failure"
    if "timed out" in lowered:
        return "timeout"
    if any(token in lowered for token in ("connection refused", "failed to establish a new connection", "could not connect")):
        return "connect_failure"
    return "unknown_failure"


def _open_url(url: str, *, timeout: int, proxy_mode: str) -> tuple[bool, int | None, str]:
    proxy_handler = urllib.request.ProxyHandler({}) if proxy_mode == "no_proxy" else urllib.request.ProxyHandler()
    opener = urllib.request.build_opener(proxy_handler)
    for method in ("HEAD", "GET"):
        request = urllib.request.Request(url, method=method, headers={"User-Agent": "taac2026-host-device-info/1.0"})
        try:
            with opener.open(request, timeout=timeout) as response:
                return True, response.getcode(), ""
        except urllib.error.HTTPError as error:
            if error.code == 405 and method == "HEAD":
                continue
            return False, error.code, str(error)
        except urllib.error.URLError as error:
            reason = getattr(error, "reason", error)
            return False, None, str(reason)
        except TimeoutError as error:
            return False, None, str(error)
    return False, None, "request failed"


def _log_dns_probe(sink: LogSink, label: str, host: str) -> None:
    if not host:
        return
    try:
        resolved = socket.getaddrinfo(host, None)
    except OSError as error:
        sink.log(f"{label}_dns=failed")
        sink.log(f"{label}_dns_detail={_compact_detail(str(error), 240)}")
        return
    sink.log(f"{label}_dns=resolved")
    sink.log(f"{label}_dns_detail={_compact_detail(str(resolved[0][4]), 240)}")


def _log_url_probe(sink: LogSink, label: str, url: str, *, config: HostDeviceInfoConfig, proxy_mode: str = "inherited") -> None:
    host = _url_host(url)
    sink.log(f"{label}_url={url}")
    if host:
        sink.log(f"{label}_host={host}")
    sink.log(f"{label}_proxy_mode={proxy_mode}")
    ok, http_code, detail = _open_url(url, timeout=config.probe_timeout_seconds, proxy_mode=proxy_mode)
    if ok:
        sink.log(f"{label}_probe=reachable")
        if http_code is not None:
            sink.log(f"{label}_http_code={http_code}")
        return
    sink.log(f"{label}_probe=failed")
    if http_code is not None:
        sink.log(f"{label}_http_code={http_code}")
    sink.log(f"{label}_failure_class={_classify_url_failure(detail)}")
    compact = _compact_detail(detail, config.probe_detail_limit)
    if compact:
        sink.log(f"{label}_probe_detail={compact}")
    _log_dns_probe(sink, label, host)


def _log_proxy_environment(sink: LogSink) -> None:
    sink.log("---- proxy environment ----")
    for variable_name in DEFAULT_PROXY_VARIABLES:
        value = os.environ.get(variable_name)
        sink.log(f"{variable_name}={_sanitize_proxy_value(value) if value else '<unset>'}")


def _log_os_release(sink: LogSink) -> None:
    os_release = Path("/etc/os-release")
    if not os_release.exists():
        sink.log("---- os-release unavailable ----")
        return
    sink.log("---- os-release ----")
    raw = os_release.read_text(encoding="utf-8")
    for line in raw.splitlines():
        if line.startswith(("PRETTY_NAME=", "VERSION=")):
            sink.log(line.split("=", 1)[1].strip('"'))


def _log_network_info(sink: LogSink) -> None:
    _log_command(sink, "network", ["ip", "-br", "addr"])


def _log_device_nodes(sink: LogSink, *, pattern: str, title: str, missing_message: str) -> None:
    matches = sorted(Path("/").glob(pattern))
    if not matches:
        sink.log(missing_message)
        return
    sink.log(f"---- {title} ----")
    for path in matches:
        try:
            stats = path.stat()
            sink.log(f"{path} mode={oct(stats.st_mode)} size={stats.st_size}")
        except OSError as error:
            sink.log(f"{path} stat failed: {error}")


def _log_python_info(sink: LogSink) -> None:
    sink.log("---- python ----")
    sink.log(f"python_executable={sys.executable}")
    sink.log(f"python_version={sys.version.replace(chr(10), ' ')}")
    sink.log(f"platform={platform.platform()}")
    sink.log(f"cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    sink.log(f"nvidia_visible_devices={os.environ.get('NVIDIA_VISIBLE_DEVICES', '<unset>')}")


def _log_python_packages(sink: LogSink) -> None:
    sink.log("---- python packages ----")
    packages = sorted(
        (
            (distribution.metadata.get("Name") or distribution.name or "<unknown>", distribution.version)
            for distribution in metadata.distributions()
        ),
        key=lambda item: item[0].lower(),
    )
    sink.log(f"installed_python_packages={len(packages)}")
    for name, version in packages:
        sink.log(f"{name}=={version}")


def _log_uv_bootstrap_status(sink: LogSink, config: HostDeviceInfoConfig) -> None:
    sink.log("---- uv bootstrap ----")
    sink.log(f"uv_install_url={config.uv_install_url}")
    uv_path = shutil.which("uv")
    if uv_path is None:
        sink.log("uv_present=0")
    else:
        sink.log("uv_present=1")
        _log_command(sink, "uv", [uv_path, "--version"])
    _log_url_probe(sink, "uv_download", config.uv_install_url, config=config)


def _pytorch_index_url_for_profile(config: HostDeviceInfoConfig, profile: str) -> str | None:
    if profile == "cpu":
        return config.pytorch_cpu_index_url
    if profile == "cuda126":
        return config.pytorch_cuda126_index_url
    return None


def _log_dependency_index_status(sink: LogSink, config: HostDeviceInfoConfig) -> None:
    sink.log("---- dependency indexes ----")
    _log_url_probe(sink, "pypi_index", config.pypi_index_url, config=config)
    _log_url_probe(sink, "tencent_pypi_index", config.tencent_pypi_index_url, config=config)
    _log_url_probe(sink, "conda_main_channel", config.conda_main_channel_url, config=config)
    _log_url_probe(sink, "conda_forge_channel", config.conda_forge_channel_url, config=config)
    _log_url_probe(sink, "tencent_conda_main_channel", config.tencent_conda_main_channel_url, config=config)
    _log_url_probe(sink, "tencent_conda_free_channel", config.tencent_conda_free_channel_url, config=config)

    if config.requested_profile:
        sink.log(f"pytorch_probe_profile={config.requested_profile}")
        url = _pytorch_index_url_for_profile(config, config.requested_profile)
        if url is None:
            sink.log("pytorch_probe=unsupported-profile")
        else:
            _log_url_probe(sink, f"pytorch_index_{config.requested_profile}", url, config=config)
        return

    sink.log("pytorch_probe_profile=all")
    for profile_name in ("cpu", "cuda126"):
        url = _pytorch_index_url_for_profile(config, profile_name)
        if url is not None:
            _log_url_probe(sink, f"pytorch_index_{profile_name}", url, config=config)


def _log_connectivity_matrix(sink: LogSink, config: HostDeviceInfoConfig) -> None:
    if not config.enable_proxy_matrix:
        return
    sink.log("---- connectivity matrix ----")
    extra_targets = {
        "pypi": config.pypi_index_url,
        "tencent_pypi": config.tencent_pypi_index_url,
        "astral": config.uv_install_url,
        "pytorch_cpu": config.pytorch_cpu_index_url,
        "conda_main": config.conda_main_channel_url,
        "conda_forge": config.conda_forge_channel_url,
        "tencent_conda_main": config.tencent_conda_main_channel_url,
        "tencent_conda_free": config.tencent_conda_free_channel_url,
    }
    for label, url in {**config.site_probe_targets, **extra_targets}.items():
        for proxy_mode in ("inherited", "no_proxy"):
            _log_url_probe(sink, f"{label}_{proxy_mode}", url, config=config, proxy_mode=proxy_mode)


def _classify_process_failure(output: str) -> str:
    lowered = output.lower()
    if "proxy" in lowered:
        return "proxy_tunnel_failure"
    if any(token in lowered for token in ("certificate", "ssl", "tls")):
        return "tls_failure"
    if any(token in lowered for token in ("timed out", "timeout")):
        return "timeout"
    if any(token in lowered for token in ("name or service not known", "temporary failure in name resolution", "could not resolve")):
        return "dns_failure"
    if any(token in lowered for token in ("connection refused", "failed to establish", "could not connect")):
        return "connect_failure"
    if any(token in lowered for token in ("not found for channel", "packagesnotfounderror", "resolvepackagenotfound")):
        return "package_resolution_failure"
    return "unknown_failure"


def _log_pip_download_probe(sink: LogSink, config: HostDeviceInfoConfig, *, label: str, index_url: str, proxy_mode: str) -> None:
    sink.log(f"{label}_package={config.pip_download_package}")
    sink.log(f"{label}_index_url={index_url}")
    sink.log(f"{label}_proxy_mode={proxy_mode}")
    with tempfile.TemporaryDirectory() as tmp_dir:
        command = [
            sys.executable,
            "-m",
            "pip",
            "download",
            "--no-deps",
            "--disable-pip-version-check",
            "--dest",
            tmp_dir,
            "-i",
            index_url,
            config.pip_download_package,
        ]
        return_code, output = _run_command(command, proxy_mode=proxy_mode, timeout=config.probe_timeout_seconds)
    if return_code == 0:
        sink.log(f"{label}_probe=reachable")
        return
    sink.log(f"{label}_probe=failed")
    sink.log(f"{label}_failure_class={_classify_process_failure(output)}")
    compact = _compact_detail(output, config.probe_detail_limit)
    if compact:
        sink.log(f"{label}_probe_detail={compact}")


def _log_pip_download_probes(sink: LogSink, config: HostDeviceInfoConfig) -> None:
    if not config.enable_pip_download_probe:
        return
    sink.log("---- pip download probes ----")
    for label, index_url in (
        ("pip_download_inherited", config.pip_download_index_url),
        ("pip_download_no_proxy", config.pip_download_index_url),
        ("pip_download_tencent_inherited", config.tencent_pypi_index_url),
        ("pip_download_tencent_no_proxy", config.tencent_pypi_index_url),
    ):
        proxy_mode = "no_proxy" if label.endswith("no_proxy") else "inherited"
        _log_pip_download_probe(sink, config, label=label, index_url=index_url, proxy_mode=proxy_mode)


def _log_conda_search_probe(sink: LogSink, config: HostDeviceInfoConfig, *, label: str, channel_url: str, proxy_mode: str) -> None:
    conda_path = shutil.which("conda")
    if conda_path is None:
        sink.log(f"{label}_probe=unavailable")
        sink.log(f"{label}_probe_detail=conda executable not found")
        return
    sink.log(f"{label}_spec={config.conda_probe_spec}")
    sink.log(f"{label}_channel_url={channel_url}")
    sink.log(f"{label}_tool={conda_path}")
    sink.log(f"{label}_proxy_mode={proxy_mode}")
    command = [
        conda_path,
        "search",
        "--json",
        "--override-channels",
        "--channel",
        channel_url,
        config.conda_probe_spec,
    ]
    return_code, output = _run_command(
        command,
        proxy_mode=proxy_mode,
        timeout=config.probe_timeout_seconds,
    )
    if return_code == 0:
        sink.log(f"{label}_probe=reachable")
        return
    sink.log(f"{label}_probe=failed")
    sink.log(f"{label}_failure_class={_classify_process_failure(output)}")
    compact = _compact_detail(output, config.probe_detail_limit)
    if compact:
        sink.log(f"{label}_probe_detail={compact}")


def _log_conda_search_probes(sink: LogSink, config: HostDeviceInfoConfig) -> None:
    if not config.enable_conda_search_probe:
        return
    sink.log("---- conda search probes ----")
    _log_conda_search_probe(sink, config, label="conda_search_inherited", channel_url=config.conda_search_channel_url, proxy_mode="inherited")
    _log_conda_search_probe(sink, config, label="conda_search_no_proxy", channel_url=config.conda_search_channel_url, proxy_mode="no_proxy")
    _log_conda_search_probe(sink, config, label="conda_search_tencent_main_inherited", channel_url=config.tencent_conda_main_channel_url, proxy_mode="inherited")
    _log_conda_search_probe(sink, config, label="conda_search_tencent_main_no_proxy", channel_url=config.tencent_conda_main_channel_url, proxy_mode="no_proxy")


def _log_build_tools(sink: LogSink) -> None:
    sink.log("---- build tools ----")
    for tool_name in DEFAULT_BUILD_TOOLS:
        executable = shutil.which(tool_name)
        if executable is None:
            sink.log(f"{tool_name}=missing")
            continue
        sink.log(f"{tool_name}=present")
        return_code, output = _run_command([executable, "--version"], timeout=5)
        if output:
            for line in output.splitlines():
                sink.log(line)
        if return_code != 0:
            sink.log(f"{tool_name} --version failed with exit code {return_code}")


def collect_host_device_info(config: HostDeviceInfoConfig) -> dict[str, object]:
    sink = LogSink()
    try:
        sink.log("==== Host and device information ====")
        sink.log(f"repo_root={config.repo_root}")
        if config.requested_profile:
            sink.log(f"requested_profile={config.requested_profile}")
        if config.requested_python:
            sink.log(f"requested_python={config.requested_python}")
        sink.log(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
        sink.log(f"NVIDIA_VISIBLE_DEVICES={os.environ.get('NVIDIA_VISIBLE_DEVICES', '<unset>')}")

        _log_os_release(sink)
        _log_proxy_environment(sink)
        _log_command(sink, "hostname", ["hostname"])
        _log_command(sink, "uptime", ["uptime"])
        _log_command(sink, "kernel", ["uname", "-a"])
        _log_command(sink, "cpu", ["lscpu"])
        _log_command(sink, "memory", ["free", "-h"])
        _log_command(sink, "block devices", ["lsblk", "-o", "NAME,SIZE,TYPE,MOUNTPOINT,MODEL"])
        _log_command(sink, "disk usage", ["df", "-h", str(config.repo_root), "/tmp"])
        _log_network_info(sink)
        _log_device_nodes(sink, pattern="dev/nvidia*", title="nvidia device nodes", missing_message="nvidia device nodes: none")
        _log_device_nodes(sink, pattern="dev/dri/*", title="/dev/dri", missing_message="/dev/dri: none")
        _log_command(sink, "nvidia-smi list", ["nvidia-smi", "-L"])
        _log_command(sink, "nvidia-smi", ["nvidia-smi"])
        _log_command(sink, "nvcc", ["nvcc", "--version"])
        _log_uv_bootstrap_status(sink, config)
        _log_dependency_index_status(sink, config)
        _log_connectivity_matrix(sink, config)
        _log_pip_download_probes(sink, config)
        _log_conda_search_probes(sink, config)
        _log_build_tools(sink)
        _log_python_info(sink)
        _log_python_packages(sink)
    finally:
        sink.close()
    return {
        "repo_root": str(config.repo_root),
        "requested_profile": config.requested_profile,
        "requested_python": config.requested_python,
    }