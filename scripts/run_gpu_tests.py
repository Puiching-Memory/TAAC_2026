#!/usr/bin/env python3
"""Local GPU test runner.

Run GPU tests on a local machine with CUDA hardware.  This script provides
the same verification sequence as the GPU test job in ``.github/workflows/ci.yml``
but runs entirely locally, producing a human-readable summary and optional
JUnit XML / coverage reports.

Usage
-----
Quick smoke test (GPU marker only)::

    uv run python scripts/run_gpu_tests.py

Full suite with coverage::

    uv run python scripts/run_gpu_tests.py --coverage

Verbose with JUnit output::

    uv run python scripts/run_gpu_tests.py --verbose --junit

Select specific marker expression::

    uv run python scripts/run_gpu_tests.py -m "gpu and not slow"
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import textwrap
import time


def _header(title: str) -> None:
    width = 60
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def _run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, text=True)
    if check and result.returncode != 0:
        print(f"  [FAIL] exit code {result.returncode}")
        sys.exit(result.returncode)
    return result


def _check_cuda() -> bool:
    _header("CUDA Environment Check")
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent("""\
            import torch
            available = torch.cuda.is_available()
            print(f"CUDA available : {available}")
            if available:
                print(f"Device count   : {torch.cuda.device_count()}")
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    free, total = torch.cuda.mem_get_info(i)
                    print(f"  [{i}] {props.name}  "
                          f"compute={props.major}.{props.minor}  "
                          f"mem={total / 1024**3:.1f}GB  "
                          f"free={free / 1024**3:.1f}GB")
            else:
                print("No CUDA devices detected — GPU tests will be skipped.")
        """)],
        text=True,
    )
    if result.returncode != 0:
        print("  [WARN] Could not query CUDA devices")
        return False

    # Quick boolean check
    check = subprocess.run(
        [sys.executable, "-c", "import torch; exit(0 if torch.cuda.is_available() else 1)"],
    )
    return check.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GPU tests locally")
    parser.add_argument("-m", "--marker", default="gpu", help="Pytest marker expression (default: gpu)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose pytest output")
    parser.add_argument("--coverage", action="store_true", help="Collect coverage data")
    parser.add_argument("--junit", action="store_true", help="Write JUnit XML report")
    parser.add_argument("--failfast", "-x", action="store_true", help="Stop on first failure")
    args = parser.parse_args()

    start = time.monotonic()

    # 1. Check CUDA
    has_cuda = _check_cuda()
    if not has_cuda:
        print("\n  Skipping GPU tests — no CUDA device available.")
        print("  Run CPU tests instead: uv run pytest -m unit -q")
        sys.exit(0)

    # 2. Torch lint
    _header("Torch Lint Rules")
    _run([sys.executable, "scripts/lint_torch.py"])

    # 3. GPU tests
    _header("GPU Tests")
    pytest_cmd = ["uv", "run"]
    if args.coverage:
        pytest_cmd.extend(["--with", "coverage", "coverage", "run", "-m"])

    pytest_cmd.extend(["pytest", "-m", args.marker])
    if args.verbose:
        pytest_cmd.append("-v")
    else:
        pytest_cmd.append("-q")
    if args.failfast:
        pytest_cmd.append("-x")
    pytest_cmd.extend(["--tb=short"])
    if args.junit:
        pytest_cmd.extend(["--junit-xml=gpu-test-results.xml"])

    result = _run(pytest_cmd, check=False)

    # 4. Coverage report (if enabled)
    if args.coverage:
        _header("Coverage Report")
        _run(
            ["uv", "run", "--with", "coverage", "coverage", "report", "--include=src/taac2026/**"],
            check=False,
        )

    # 5. Memory summary
    _header("CUDA Memory Summary")
    subprocess.run(
        [sys.executable, "-c", textwrap.dedent("""\
            import torch
            if torch.cuda.is_available():
                print(torch.cuda.memory_summary(device='cuda:0', abbreviated=True))
        """)],
        text=True,
    )

    elapsed = time.monotonic() - start
    _header("Summary")
    status = "PASSED" if result.returncode == 0 else "FAILED"
    print(f"  Status  : {status}")
    print(f"  Elapsed : {elapsed:.1f}s")
    print(f"  Marker  : {args.marker}")
    if args.junit:
        print("  JUnit   : gpu-test-results.xml")

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
