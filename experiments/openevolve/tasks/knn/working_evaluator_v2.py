"""kNN evaluator V2: validate returned neighbor distances before measuring fitness."""
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
from experiments.openevolve.tasks.knn.validation import validate_knn_output  # noqa: E402
from experiments.openevolve.tasks.knn.working_evaluator import speed_score  # noqa: E402

TASK = "knn"
SANDBOX_TIMEOUT_S = 180


def _failure(error: str) -> dict:
    return {"evaluator_version": "v2", "valid": False, "combined_score": 0.0, "recall": 0.0,
            "speed_ratio": 0.0, "validation_failures": [error]}


def evaluate(program_path: str) -> dict:
    code = Path(program_path).read_text()
    preflight = run_candidate(code, DRIVER, {"mode": "working_v2_preflight", "sets": DEV_SETS},
                              timeout_s=SANDBOX_TIMEOUT_S)
    if not preflight.ok:
        metrics = _failure(f"preflight failed: {preflight.error[-300:]}")
    else:
        rows = preflight.output.get("sets", []) if isinstance(preflight.output, dict) else []
        failures = []
        if len(rows) != len(DEV_SETS):
            failures.append(f"preflight returned {len(rows)} sets; expected {len(DEV_SETS)}")
        for index, (spec, output) in enumerate(zip(DEV_SETS, rows)):
            if isinstance(output, dict) and "error" in output:
                failures.append(f"set {index}: {output['error']}")
                continue
            vectors, queries, k, _ = make_set(**spec)
            failures.extend(f"set {index}: {error}" for error in
                            validate_knn_output(vectors, queries, output, k))

        if failures:
            metrics = dict(_failure("preflight correctness check failed"), validation_failures=failures)
        else:
            timed = run_candidate(code, DRIVER, {"mode": "working_v2_timed", "sets": DEV_SETS},
                                  timeout_s=SANDBOX_TIMEOUT_S)
            if not timed.ok:
                metrics = _failure(f"timed evaluation failed: {timed.error[-300:]}")
            else:
                rows = timed.output.get("sets", []) if isinstance(timed.output, dict) else []
                failures = []
                recalls, ratios = [], []
                if len(rows) != len(DEV_SETS):
                    failures.append(f"timed evaluation returned {len(rows)} sets; expected {len(DEV_SETS)}")
                for index, (spec, output) in enumerate(zip(DEV_SETS, rows)):
                    if isinstance(output, dict) and "error" in output:
                        failures.append(f"set {index}: {output['error']}")
                        continue
                    vectors, queries, k, _ = make_set(**spec)
                    reference_ids = np.asarray(output.get("reference_ids", []))
                    reference_time = output.get("reference_time")
                    runs = output.get("runs", [])
                    if reference_ids.shape != (len(queries), k) or not np.issubdtype(reference_ids.dtype, np.integer):
                        failures.append(f"set {index}: invalid exact-reference IDs")
                        continue
                    if not isinstance(reference_time, (int, float)) or not math.isfinite(reference_time) or reference_time <= 0:
                        failures.append(f"set {index}: invalid exact-reference time")
                        continue
                    if len(runs) != 3:
                        failures.append(f"set {index}: got {len(runs)} timed runs; expected 3")
                        continue

                    validated_runs = []
                    for run_index, run in enumerate(runs):
                        errors = validate_knn_output(vectors, queries, (run.get("ids"), run.get("dists")), k)
                        failures.extend(f"set {index} run {run_index}: {error}" for error in errors)
                        elapsed = run.get("time")
                        if not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed <= 0:
                            failures.append(f"set {index} run {run_index}: invalid elapsed time")
                            continue
                        if not errors:
                            validated_runs.append((float(elapsed), np.asarray(run["ids"])))
                    if len(validated_runs) == 3:
                        best_time, best_ids = min(validated_runs, key=lambda row: row[0])
                        hits = sum(len(set(map(int, a)) & set(map(int, b)))
                                   for a, b in zip(best_ids, reference_ids))
                        recalls.append(hits / (k * len(queries)))
                        ratios.append(best_time / reference_time)

                if failures or len(recalls) != len(DEV_SETS):
                    metrics = dict(_failure("timed correctness check failed"), validation_failures=failures)
                else:
                    recall = sum(recalls) / len(recalls)
                    ratio = max(ratios)
                    metrics = {"evaluator_version": "v2", "valid": True, "combined_score": speed_score(recall, ratio),
                               "recall": recall, "speed_ratio": ratio, "validation_failures": []}

    append_candidate({"task": TASK, "evaluator_version": "v2", "program_path": program_path, "code": code,
                      "metrics": metrics, "sandbox_ok": metrics["valid"]})
    return metrics
