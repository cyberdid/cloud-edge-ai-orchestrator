"""Working evaluator for circle packing: OpenEvolve's upstream evaluator, unchanged,
except that the candidate runs in the Docker sandbox instead of a host subprocess.

This is the fitness that drives selection. It is deliberately not hardened; the
independent re-check lives in ``recheck.py`` and is never used during evolution.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:  # OpenEvolve loads this file by path
    sys.path.insert(0, str(REPO_ROOT))

from experiments.openevolve.candidate_log import append_candidate  # noqa: E402
from experiments.openevolve.sandbox_exec import run_candidate  # noqa: E402
from experiments.openevolve.tasks.circle_packing.driver import DRIVER  # noqa: E402

TASK = "circle_packing"
SANDBOX_TIMEOUT_S = 60
UPSTREAM_EVALUATOR = REPO_ROOT / "experiments/openevolve/upstream/circle_packing/evaluator.py"

_spec = importlib.util.spec_from_file_location("oe_upstream_circle_packing_evaluator", UPSTREAM_EVALUATOR)
upstream = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(upstream)


def evaluate(program_path: str) -> dict:
    code = Path(program_path).read_text()
    runs = []

    def sandboxed_run(path, timeout_seconds=SANDBOX_TIMEOUT_S):
        result = run_candidate(code, DRIVER, {}, timeout_s=min(timeout_seconds, SANDBOX_TIMEOUT_S))
        runs.append(result)
        if not result.ok:
            raise RuntimeError(result.error)
        out = result.output
        return np.array(out["centers"], dtype=float), np.array(out["radii"], dtype=float), out["sum_radii"]

    upstream.run_with_timeout = sandboxed_run
    metrics = upstream.evaluate(program_path)
    if runs and not runs[-1].ok:
        metrics = dict(metrics, combined_score=0.0, validity=0.0, error=runs[-1].error[-500:])
    append_candidate({
        "task": TASK,
        "program_path": program_path,
        "code": code,
        "metrics": metrics,
        "sandbox_ok": bool(runs and runs[-1].ok),
        "output": runs[-1].output if runs else None,
    })
    return metrics
