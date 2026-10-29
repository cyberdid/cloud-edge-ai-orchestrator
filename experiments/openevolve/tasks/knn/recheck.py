"""Independent re-check for approximate kNN (never visible to the mutating model).

Validity: RIE's ``check_neighbors`` on fresh randomised challenges (seeded by the code
hash), the exact-copy rule (an indexed query returns itself first at distance 0) and
the static check. Score: recall@k against exact ids computed on the host, on a
holdout from a different distribution, with the speed penalty of the working
evaluator; candidate time is the sum over 3 disjoint query batches, so caching
results across calls does not help.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from antigravity_hybrid_orchestrator.integrity.oracle import check_neighbors
from experiments.openevolve.candidate_log import code_sha256
from experiments.openevolve.sandbox_exec import run_candidate
from experiments.openevolve.static_check import forbidden_primitives
from experiments.openevolve.tasks.knn.data import brute_knn, fresh_specs, make_set
from experiments.openevolve.tasks.knn.driver import DRIVER
from experiments.openevolve.tasks.knn.working_evaluator import speed_score

HOLDOUT_SETS = [
    dict(seed=9001, n=180_000, d=24, k=10, centres=200, sigma=0.08),
    dict(seed=9002, n=120_000, d=48, k=8, centres=80, sigma=0.05),
    dict(seed=9003, n=250_000, d=12, k=10, centres=150, sigma=0.04),
]
SANDBOX_TIMEOUT_S = 300


@lru_cache(maxsize=None)
def _holdout_reference(index: int) -> np.ndarray:
    x, qs, k, _ = make_set(**HOLDOUT_SETS[index])
    return brute_knn(x, qs, k)[0]


def _fresh_failures(specs: list[dict], outputs: list) -> list[str]:
    failures = []
    for n, (spec, out) in enumerate(zip(specs, outputs)):
        if isinstance(out, dict):
            failures.append(f"fresh {n}: {out['error']}")
            continue
        x, qs, k, copied = make_set(**spec)
        ids, dists = out
        found = check_neighbors(x.tolist(), qs.tolist(), ids, dists, k)
        failures += [f"fresh {n}: {f}" for f in found]
        if not found:
            for row, index in enumerate(copied.tolist()):
                if ids[row][0] != index or dists[row][0] != 0:
                    failures.append(f"fresh {n} query {row}: indexed copy not returned first at distance 0")
    return failures


def recheck_knn(program_code: str) -> dict:
    reasons = list(forbidden_primitives(program_code))
    specs = fresh_specs(int(code_sha256(program_code)[:8], 16))
    run = run_candidate(program_code, DRIVER, {"mode": "recheck", "fresh": specs, "sets": HOLDOUT_SETS},
                        timeout_s=SANDBOX_TIMEOUT_S)
    ref = run_candidate("", DRIVER, {"mode": "reference", "sets": HOLDOUT_SETS}, timeout_s=SANDBOX_TIMEOUT_S)
    invalid = {"valid": False, "score": 0.0, "holdout_recall": 0.0, "holdout_speed_ratio": 0.0}
    if not run.ok or not ref.ok:
        return dict(invalid, reasons=[f"run failed: {(run.error or ref.error)[-300:]}"] + reasons)
    reasons = _fresh_failures(specs, run.output["fresh"]) + reasons
    recalls, ratios = [], []
    for index, (out, ref_time) in enumerate(zip(run.output["sets"], ref.output["sets"])):
        if "error" in out:
            reasons.append(f"holdout {index}: {out['error']}")
            continue
        k = HOLDOUT_SETS[index]["k"]
        truth = _holdout_reference(index)
        ids = out["ids"]
        if len(ids) != len(truth) or any(len(row) != k for row in ids):
            reasons.append(f"holdout {index}: wrong result shape")
            continue
        recalls.append(sum(len(set(a) & set(b.tolist())) for a, b in zip(ids, truth)) / truth.size)
        ratios.append(out["time"] / ref_time)
    if len(recalls) != len(HOLDOUT_SETS):
        return dict(invalid, reasons=reasons)
    recall, ratio = sum(recalls) / len(recalls), max(ratios)
    valid = not reasons
    return {"valid": valid, "score": speed_score(recall, ratio) if valid else 0.0, "reasons": reasons,
            "holdout_recall": recall, "holdout_speed_ratio": ratio}
