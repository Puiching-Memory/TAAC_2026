#!/usr/bin/env python3
"""Static lint rules for common PyTorch footguns.

Checks model code under ``config/`` for patterns that have caused real
bugs in code review:

  T001  torch.zeros / torch.ones without inheriting dtype/device from an
        existing tensor.  Prefer ``tensor.new_zeros()`` / ``tensor.new_ones()``
        under AMP to avoid silent fp32 upcasts.

  T002  torch.cuda.Stream() without an explicit ``device=`` keyword.
        Streams created on the wrong device silently fail to parallelize.

  T003  Bare ``.item()``, ``.cpu()``, or ``.numpy()`` inside a ``forward``
        method, which forces a GPU→CPU synchronization every call.

Exit code: 0 if clean, 1 if any violations found.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

SCAN_ROOTS = [
    Path("config"),
    Path("src/taac2026"),
]

Violation = tuple[Path, int, str, str]  # (file, lineno, rule_id, message)


class _TorchLintVisitor(ast.NodeVisitor):
    """AST visitor that collects torch-specific lint violations."""

    def __init__(self, filepath: Path) -> None:
        self.filepath = filepath
        self.violations: list[Violation] = []
        self._in_forward = False
        self._nn_parameter_nodes: set[int] = set()

    # -- context tracking ----------------------------------------------------

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        was = self._in_forward
        if node.name == "forward":
            self._in_forward = True
        self.generic_visit(node)
        self._in_forward = was

    visit_AsyncFunctionDef = visit_FunctionDef

    # -- T001: torch.zeros / torch.ones without tensor inheritance -----------

    def visit_Call(self, node: ast.Call) -> None:
        # Track nn.Parameter(...) wrappers so T001 can skip their arguments
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "Parameter"
            and isinstance(func.value, ast.Name)
            and func.value.id == "nn"
        ):
            for arg in node.args:
                self._nn_parameter_nodes.add(id(arg))

        self._check_t001(node)
        self._check_t002(node)
        self._check_t003(node)
        self.generic_visit(node)

    def _check_t001(self, node: ast.Call) -> None:
        """Flag torch.zeros(...) / torch.ones(...) — suggest new_zeros/new_ones."""
        func = node.func
        # Match torch.zeros(...) or torch.ones(...)
        if not (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == "torch"
            and func.attr in ("zeros", "ones")
        ):
            return

        # Allow if dtype= is explicitly passed
        for kw in node.keywords:
            if kw.arg == "dtype":
                return

        # Allow inside nn.Parameter(...) — parameter init is dtype-safe
        if id(node) in self._nn_parameter_nodes:
            return

        self.violations.append((
            self.filepath,
            node.lineno,
            "T001",
            f"torch.{func.attr}() without explicit dtype — prefer tensor.new_{func.attr}() for AMP safety",
        ))

    def _check_t002(self, node: ast.Call) -> None:
        """Flag torch.cuda.Stream() without device=."""
        func = node.func
        # Match torch.cuda.Stream(...)
        if not (
            isinstance(func, ast.Attribute)
            and func.attr == "Stream"
            and isinstance(func.value, ast.Attribute)
            and func.value.attr == "cuda"
            and isinstance(func.value.value, ast.Name)
            and func.value.value.id == "torch"
        ):
            return

        for kw in node.keywords:
            if kw.arg == "device":
                return

        # Also allow positional first arg (device is the first positional param)
        if node.args:
            return

        self.violations.append((
            self.filepath,
            node.lineno,
            "T002",
            "torch.cuda.Stream() without device= — stream may bind to wrong GPU",
        ))

    def _check_t003(self, node: ast.Call) -> None:
        """Flag .item() / .cpu() / .numpy() inside forward() — GPU sync."""
        if not self._in_forward:
            return

        func = node.func
        if not (
            isinstance(func, ast.Attribute)
            and func.attr in ("item", "cpu", "numpy")
        ):
            return

        self.violations.append((
            self.filepath,
            node.lineno,
            "T003",
            f".{func.attr}() inside forward() forces GPU→CPU sync every call",
        ))


def lint_file(filepath: Path) -> list[Violation]:
    source = filepath.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(filepath))
    except SyntaxError:
        return []
    visitor = _TorchLintVisitor(filepath)
    visitor.visit(tree)
    return visitor.violations


def main() -> int:
    files: list[Path] = []
    for root in SCAN_ROOTS:
        if root.exists():
            files.extend(root.rglob("*.py"))

    all_violations: list[Violation] = []
    for f in sorted(files):
        all_violations.extend(lint_file(f))

    if not all_violations:
        print("lint_torch: all clean ✓")
        return 0

    for filepath, lineno, rule_id, message in all_violations:
        print(f"{filepath}:{lineno}: {rule_id} {message}")

    print(f"\n{len(all_violations)} violation(s) found.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
