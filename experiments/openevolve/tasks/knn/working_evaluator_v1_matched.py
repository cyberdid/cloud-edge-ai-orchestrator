"""V1-compatible kNN evaluator using the V2 timing harness without its gate.

This isolates the effect of distance-correctness validation while matching V2's
datasets, exact-reference timing, three candidate timings, and score calculation.
Reported distances are deliberately ignored, as in the historical V1 evaluator.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.openevolve.candidate_log import append_candidate  # noqa: E402
from experiments.openevolve.sandbox_exec import run_candidate  # noqa: E402
from experiments.openevolve.tasks.knn.data import DEV_SETS, make_set  # noqa: E402
from experiments.openevolve.tasks.knn.driver import DRIVER  # noqa: E402
from experiments.openevolve.tasks.knn.working_evaluator import speed_score  # noqa: E402

TASK = "knn"
SANDBOX_TIMEOUT_S = 180


def _failure(error: str) -> dict:
    return {"evaluator_version": "v1_matched", "combined_score": 0.0,
            "recall": 0.0, "speed_ratio": 0.0, "error": error}


def evaluate(program_path: str) -> dict:
    code = Path(program_path).read_text()
    result = run_candidate(code, DRIVER, {"mode": "working_v1_matched_timed", "sets": DEV_SETS},
                           timeout_s=SANDBOX_TIMEOUT_S)
    rows = result.output.get("sets", []) if result.ok and isinstance(result.output, dict) else []
    errors = [row["error"] for row in rows if isinstance(row, dict) and "error" in row]
    if not result.ok or errors or len(rows) != len(DEV_SETS):
        metrics = _failure((result.error or "; ".join(errors) or
                            f"got {len(rows)} sets; expected {len(DEV_SETS)}")[-500:])
    else:
        recalls, ratios = [], []
        for index, (spec, row) in enumerate(zip(DEV_SETS, rows)):
            vectors, queries, k, _ = make_set(**spec)
            reference = np.asarray(row.get("reference_ids", []))
            runs = row.get("runs", [])
            if reference.shape != (len(queries), k) or len(runs) != 3:
                errors.append(f"set {index}: malformed reference or timed outputs")
                continue
            valid_runs = []
            for run in runs:
                ids = np.asarray(run.get("ids", []))
                elapsed = run.get("time")
                if ids.shape != (len(queries), k) or not isinstance(elapsed, (int, float)) \
                        or not math.isfinite(elapsed) or elapsed <= 0:
                    errors.append(f"set {index}: malformed candidate IDs or timing")
                    continue
                valid_runs.append((float(elapsed), ids))
            reference_time = row.get("reference_time")
            if not isinstance(reference_time, (int, float)) or not math.isfinite(reference_time) \
                    or reference_time <= 0 or len(valid_runs) != 3:
                errors.append(f"set {index}: invalid reference time or candidate run")
                continue
            candidate_time, candidate_ids = min(valid_runs, key=lambda item: item[0])
            hits = sum(len(set(map(int, actual)) & set(map(int, expected)))
                       for actual, expected in zip(candidate_ids, reference))
            recalls.append(hits / (k * len(queries)))
            ratios.append(candidate_time / float(reference_time))

        if errors or len(recalls) != len(DEV_SETS):
            metrics = _failure("; ".join(errors)[-500:])
        else:
            recall, ratio = sum(recalls) / len(recalls), max(ratios)
            metrics = {"evaluator_version": "v1_matched", "combined_score": speed_score(recall, ratio),
                       "recall": recall, "speed_ratio": ratio}
    append_candidate({"task": TASK, "evaluator_version": "v1_matched", "program_path": program_path,
                      "code": code, "metrics": metrics, "sandbox_ok": result.ok})
    return metrics
