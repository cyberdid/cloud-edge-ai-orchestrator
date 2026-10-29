"""Post-hoc: attribute every logged candidate to the call that produced it, re-check it
independently, and summarise the run (``python -m experiments.openevolve.recheck_all DIR``)."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable

Recheck = Callable[[str], dict]


def _lines(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def default_recheck(task: str) -> Recheck:
    if task == "circle_packing":
        from experiments.openevolve.tasks.circle_packing.recheck import recheck_packing
        return recheck_packing
    from experiments.openevolve.tasks.knn.recheck import recheck_knn
    return recheck_knn


def attribute(calls: list[dict], candidates: list[dict]) -> list[dict]:
    """Evaluation is serialised, so a candidate belongs to the latest call that started
    before it was logged; candidates before any call are the seed."""
    ordered = sorted(calls, key=lambda c: c["ts_start"])
    out = []
    for candidate in candidates:
        source = None
        for call in ordered:
            if call["ts_start"] <= candidate["ts"]:
                source = call
            else:
                break
        out.append(dict(candidate, call_index=source["call_index"] if source else None,
                        model=source["model"] if source else "seed"))
    return out


def _verified(result: dict, seed_verified: float) -> float:
    return result["score"] if result["valid"] else seed_verified


def recheck_run(run_dir: Path, recheck: Recheck | None = None) -> dict:
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    recheck = recheck or default_recheck(manifest["task"])
    calls = _lines(run_dir / "calls.jsonl")
    candidates = attribute(calls, _lines(run_dir / "candidates.jsonl"))
    cache_path = run_dir / "rechecked.jsonl"
    cache = {row["code_sha256"]: row["recheck"] for row in _lines(cache_path)}
    with cache_path.open("a") as handle:
        for candidate in candidates:
            key = candidate["code_sha256"]
            if key not in cache:
                cache[key] = recheck(candidate["code"])
                handle.write(json.dumps({"code_sha256": key, "call_index": candidate["call_index"],
                                         "model": candidate["model"], "recheck": cache[key]}) + "\n")
    for candidate in candidates:
        candidate["recheck"] = cache[candidate["code_sha256"]]
        candidate["working"] = float(candidate["metrics"].get("combined_score", 0.0))

    seed = next(c for c in candidates if c["call_index"] is None)
    seed_working, seed_verified = seed["working"], _verified(seed["recheck"], 0.0)
    above = [c for c in candidates if c["call_index"] is not None and c["working"] > seed_working]
    false = [c for c in above if not c["recheck"]["valid"] or c["recheck"]["score"] <= seed_verified]
    shipped = max(candidates, key=lambda c: c["working"])  # max() keeps the earliest on ties

    curve, best = [], seed
    by_call: dict[int, list[dict]] = {}
    for c in candidates:
        if c["call_index"] is not None:
            by_call.setdefault(c["call_index"], []).append(c)
    for index in sorted(c["call_index"] for c in calls):
        for c in by_call.get(index, []):
            if c["working"] > best["working"]:
                best = c
        curve.append([index, best["working"], _verified(best["recheck"], seed_verified)])

    argon = [c for c in calls if c["model"] == "argon"]
    statuses = [c["status"] for c in calls if c["status"] != "ok"]
    return {
        "task": manifest["task"], "arm": manifest["arm"], "seed": manifest["seed"],
        "seed_working_score": seed_working, "seed_score": seed_verified,
        "shipped_working_score": shipped["working"],
        "shipped_verified_score": _verified(shipped["recheck"], seed_verified),
        "shipped_is_false": any(c is shipped for c in false),
        "best_verified_score": max([seed_verified] + [c["recheck"]["score"] for c in candidates if c["recheck"]["valid"]]),
        "above_seed_count": len(above), "false_improvements": len(false),
        "false_improvement_rate": len(false) / len(above) if above else 0.0,
        "candidates": len(candidates), "calls": len(calls), "curve": curve,
        "argon_calls": len(argon),
        "argon_tokens": sum(c["prompt_tokens"] + c["completion_tokens"] for c in argon),
        "argon_wall_s": sum(c["latency_s"] for c in argon),
        "gemma_wall_s": sum(c["latency_s"] for c in calls if c["model"] == "gemma"),
        "failed_calls_by_status": {s: statuses.count(s) for s in sorted(set(statuses))},
    }


if __name__ == "__main__":
    for path in sys.argv[1:]:
        print(json.dumps({k: v for k, v in recheck_run(Path(path)).items() if k != "curve"}))
