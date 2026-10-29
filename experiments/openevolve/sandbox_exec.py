"""Run one candidate program inside the Docker sandbox and collect its JSON result.

The workspace holds only ``candidate.py``, ``driver.py`` and ``payload.json`` and is
mounted read-only; the driver writes ``/output/result.json``. Evaluators, re-checks
and holdout data never enter the workspace.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from antigravity_hybrid_orchestrator.runtime.sandbox import DockerSandbox

MAX_RESULT_BYTES = 8 * 1024 * 1024
ERROR_TAIL_CHARS = 2000


@dataclass(frozen=True)
class SandboxResult:
    ok: bool
    output: dict | list | str | int | float | None
    error: str
    elapsed_s: float


def run_candidate(
    program_code: str,
    driver_code: str,
    payload: dict,
    timeout_s: float,
    image: str = "antigravity-orion:local",
) -> SandboxResult:
    with tempfile.TemporaryDirectory(prefix="oe-ws-") as ws, tempfile.TemporaryDirectory(prefix="oe-out-") as out:
        workspace, output = Path(ws), Path(out)
        (workspace / "candidate.py").write_text(program_code)
        (workspace / "driver.py").write_text(driver_code)
        (workspace / "payload.json").write_text(json.dumps(payload))
        sandbox = DockerSandbox(workspace, image, output_directory=output)
        command = sandbox.build_command("python3 driver.py", read_only=True)
        start = time.monotonic()
        proc = subprocess.Popen(
            command, env=DockerSandbox.client_environment(),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            sandbox.cleanup()
            proc.kill()
            proc.communicate()
            return SandboxResult(False, None, "timeout", time.monotonic() - start)
        finally:
            sandbox.cleanup()
        elapsed = time.monotonic() - start
        result = output / "result.json"
        if proc.returncode or not result.exists():
            tail = (stderr or stdout or f"exit code {proc.returncode}")[-ERROR_TAIL_CHARS:]
            return SandboxResult(False, None, tail, elapsed)
        if result.stat().st_size > MAX_RESULT_BYTES:
            return SandboxResult(False, None, "output_too_large", elapsed)
        try:
            return SandboxResult(True, json.loads(result.read_text()), "", elapsed)
        except json.JSONDecodeError as exc:
            return SandboxResult(False, None, f"invalid_json: {exc}", elapsed)
