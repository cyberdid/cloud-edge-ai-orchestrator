"""Run one candidate program in isolation and collect its JSON result.

The workspace holds only ``candidate.py``, ``driver.py`` and ``payload.json``; the
driver writes ``/output/result.json``. Evaluators, re-checks and holdout data never
enter the workspace.

Executor (``OE_PILOT_EXECUTOR``): ``docker`` (default, the pilot's sandbox) or
``subprocess`` for hosts without Docker such as Colab VMs: a child process with
memory and file-size limits; as root it runs as uid/gid 65534 and, when the host
allows it, in its own network namespace.
"""
from __future__ import annotations

import json
import os
import resource
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from antigravity_hybrid_orchestrator.runtime.sandbox import DockerSandbox

MAX_RESULT_BYTES = 8 * 1024 * 1024
ERROR_TAIL_CHARS = 2000
SUBPROCESS_MEMORY_BYTES = 4 * 1024**3
SUBPROCESS_FILE_BYTES = 16 * 1024**2
NOBODY = 65534
_UNSHARE_NET: bool | None = None


@dataclass(frozen=True)
class SandboxResult:
    ok: bool
    output: dict | list | str | int | float | None
    error: str
    elapsed_s: float


def _collect(returncode: int, stdout: str, stderr: str, result: Path, elapsed: float) -> SandboxResult:
    if returncode or not result.exists():
        tail = (stderr or stdout or f"exit code {returncode}")[-ERROR_TAIL_CHARS:]
        return SandboxResult(False, None, tail, elapsed)
    if result.stat().st_size > MAX_RESULT_BYTES:
        return SandboxResult(False, None, "output_too_large", elapsed)
    try:
        return SandboxResult(True, json.loads(result.read_text()), "", elapsed)
    except json.JSONDecodeError as exc:
        return SandboxResult(False, None, f"invalid_json: {exc}", elapsed)


def _can_unshare_net() -> bool:
    global _UNSHARE_NET
    if _UNSHARE_NET is None:
        try:
            _UNSHARE_NET = subprocess.run(["unshare", "-n", "true"], capture_output=True, timeout=10).returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            _UNSHARE_NET = False
    return _UNSHARE_NET


def _limits() -> None:
    resource.setrlimit(resource.RLIMIT_AS, (SUBPROCESS_MEMORY_BYTES, SUBPROCESS_MEMORY_BYTES))
    resource.setrlimit(resource.RLIMIT_FSIZE, (SUBPROCESS_FILE_BYTES, SUBPROCESS_FILE_BYTES))


def _run_subprocess(program_code: str, driver_code: str, payload: dict, timeout_s: float) -> SandboxResult:
    with tempfile.TemporaryDirectory(prefix="oe-ws-") as ws:
        workspace = Path(ws)
        output = workspace / "output"
        output.mkdir()
        (workspace / "candidate.py").write_text(program_code)
        (workspace / "driver.py").write_text(driver_code.replace("/output/", f"{output}/"))
        (workspace / "payload.json").write_text(json.dumps(payload))
        command = [os.environ.get("OE_PILOT_CANDIDATE_PYTHON", sys.executable), "driver.py"]
        if os.geteuid() == 0:
            for path in (workspace, output):
                os.chown(path, NOBODY, NOBODY)
            os.chmod(workspace, 0o755)
            command = ["setpriv", f"--reuid={NOBODY}", f"--regid={NOBODY}", "--clear-groups", *command]
            if _can_unshare_net():
                command = ["unshare", "-n", "--", *command]
        env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": str(workspace), "PYTHONDONTWRITEBYTECODE": "1",
               "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        start = time.monotonic()
        proc = subprocess.Popen(command, cwd=workspace, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, preexec_fn=_limits, start_new_session=True)
        try:
            stdout, stderr = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.communicate()
            return SandboxResult(False, None, "timeout", time.monotonic() - start)
        return _collect(proc.returncode, stdout, stderr, output / "result.json", time.monotonic() - start)


def run_candidate(
    program_code: str,
    driver_code: str,
    payload: dict,
    timeout_s: float,
    image: str = "antigravity-orion:local",
) -> SandboxResult:
    if os.environ.get("OE_PILOT_EXECUTOR", "docker") == "subprocess":
        return _run_subprocess(program_code, driver_code, payload, timeout_s)
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
        return _collect(proc.returncode, stdout, stderr, output / "result.json", time.monotonic() - start)
