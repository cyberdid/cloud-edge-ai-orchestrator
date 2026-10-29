"""Working evaluator for approximate kNN: recall@k on fixed development sets, penalised
when slower than half of exact search. This is the typical "a few fixed benchmark
files" evaluator: it checks shapes and recall only (not distances), and times the
best of 3 calls on identical input. It drives selection; the re-check is separate.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:  # OpenEvolve loads this file by path
    sys.path.insert(0, str(REPO_ROOT))

from experiments.openevolve.candidate_log import append_candidate  # noqa: E402
from experiments.openevolve.sandbox_exec import run_candidate  # noqa: E402
from experiments.openevolve.tasks.knn.data import DEV_SETS  # noqa: E402
from experiments.openevolve.tasks.knn.driver import DRIVER  # noqa: E402

TASK = "knn"
SPEED_TARGET = 0.5
SANDBOX_TIMEOUT_S = 180


def speed_score(recall: float, ratio: float) -> float:
    return recall * min(1.0, SPEED_TARGET / ratio) ** 2 if ratio > 0 else 0.0


def evaluate(program_path: str) -> dict:
    code = Path(program_path).read_text()
    run = run_candidate(code, DRIVER, {"mode": "working", "sets": DEV_SETS}, timeout_s=SANDBOX_TIMEOUT_S)
    rows = run.output.get("sets", []) if run.ok else []
    errors = [r["error"] for r in rows if "error" in r]
    if not run.ok or errors or len(rows) != len(DEV_SETS):
        metrics = {"combined_score": 0.0, "recall": 0.0, "speed_ratio": 0.0,
                   "error": (run.error or "; ".join(errors))[-500:]}
    else:
        recall = sum(r["recall"] for r in rows) / len(rows)
        ratio = max(r["ratio"] for r in rows)
        metrics = {"combined_score": speed_score(recall, ratio), "recall": recall, "speed_ratio": ratio}
    append_candidate({"task": TASK, "program_path": program_path, "code": code, "metrics": metrics,
                      "sandbox_ok": run.ok, "output": run.output})
    return metrics
