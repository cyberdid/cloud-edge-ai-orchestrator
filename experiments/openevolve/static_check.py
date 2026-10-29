"""Static re-check rule: a candidate must not reach files, processes, the network or
the interpreter's own machinery (used by the independent re-checks only)."""
from __future__ import annotations

import ast

FORBIDDEN_MODULES = frozenset({
    "os", "sys", "subprocess", "socket", "inspect", "importlib",
    "builtins", "ctypes", "pathlib", "shutil",
})
FORBIDDEN_CALLS = frozenset({
    "open", "eval", "exec", "compile", "__import__",
    "getattr", "setattr", "globals", "locals", "vars",
})


def forbidden_primitives(program_code: str) -> list[str]:
    """Return one finding per forbidden import or call; empty means clean."""
    try:
        tree = ast.parse(program_code)
    except SyntaxError as exc:
        return [f"syntax error: {exc.msg}"]
    findings = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            names = []
        for name in names:
            if name.split(".")[0] in FORBIDDEN_MODULES:
                findings.append(f"forbidden import {name} (line {node.lineno})")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
            findings.append(f"forbidden call {node.func.id} (line {node.lineno})")
    return findings
