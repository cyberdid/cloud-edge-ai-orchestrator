"""OpenEvolve pilot: post-hoc attribution, re-check summary and aggregation (no Docker)."""
import json

import pytest

from experiments.openevolve import analyze
from experiments.openevolve.recheck_all import attribute, recheck_run


def make_run(tmp_path, candidates, calls=None, task="circle_packing", arm="L", seed=101):
    run = tmp_path / f"{task}_{arm}_s{seed}"
    run.mkdir(parents=True)
    calls = calls if calls is not None else [
        {"call_index": i, "model": "gemma", "ts_start": 10.0 * (i + 1), "latency_s": 2.0,
         "prompt_tokens": 5, "completion_tokens": 7, "status": "ok"} for i in range(len(candidates) - 1)]
    (run / "calls.jsonl").write_text("".join(json.dumps(c) + "\n" for c in calls))
    rows = [dict(c, code_sha256=c["code"] * 4) for c in candidates]
    (run / "candidates.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (run / "manifest.json").write_text(json.dumps({"task": task, "arm": arm, "seed": seed, "status": "complete"}))
    return run


def cand(code, score, ts):
    return {"code": code, "metrics": {"combined_score": score}, "ts": ts}


def fake_recheck(table):
    calls = []

    def recheck(code):
        calls.append(code)
        valid, score = table[code]
        return {"valid": valid, "score": score, "reasons": [] if valid else ["bad"]}

    recheck.calls = calls
    return recheck


def test_attribution_by_timestamp():
    calls = [{"call_index": i, "model": m, "ts_start": t} for i, (m, t) in enumerate([("gemma", 1), ("argon", 5), ("gemma", 9)])]
    out = attribute(calls, [{"ts": 0.5}, {"ts": 2}, {"ts": 6}, {"ts": 10}])
    assert [c["call_index"] for c in out] == [None, 0, 1, 2]
    assert [c["model"] for c in out] == ["seed", "gemma", "argon", "gemma"]


def test_shipped_false_falls_back_to_seed(tmp_path):
    run = make_run(tmp_path, [cand("s", 0.3, 1), cand("a", 0.5, 11), cand("b", 0.9, 21)])
    summary = recheck_run(run, recheck=fake_recheck({"s": (True, 0.3), "a": (True, 0.5), "b": (False, 0.0)}))
    assert summary["shipped_working_score"] == 0.9
    assert summary["shipped_is_false"] is True
    assert summary["shipped_verified_score"] == summary["seed_score"] == 0.3
    assert summary["best_verified_score"] == 0.5


def test_false_improvement_rate(tmp_path):
    run = make_run(tmp_path, [cand("s", 0.3, 1), cand("a", 0.4, 11), cand("b", 0.5, 21),
                              cand("c", 0.6, 31), cand("d", 0.7, 41), cand("e", 0.1, 51)])
    table = {"s": (True, 0.3), "a": (True, 0.4), "b": (True, 0.5), "c": (False, 0.0), "d": (True, 0.7), "e": (True, 0.1)}
    summary = recheck_run(run, recheck=fake_recheck(table))
    assert summary["above_seed_count"] == 4
    assert summary["false_improvements"] == 1
    assert summary["false_improvement_rate"] == pytest.approx(0.25)


def test_verified_not_above_seed_counts_as_false(tmp_path):
    run = make_run(tmp_path, [cand("s", 0.3, 1), cand("a", 0.8, 11)])
    summary = recheck_run(run, recheck=fake_recheck({"s": (True, 0.3), "a": (True, 0.2)}))
    assert summary["false_improvements"] == 1 and summary["shipped_is_false"] is True


def test_recheck_cache_by_hash(tmp_path):
    run = make_run(tmp_path, [cand("s", 0.3, 1), cand("a", 0.5, 11), cand("a", 0.5, 21)])
    recheck = fake_recheck({"s": (True, 0.3), "a": (True, 0.5)})
    recheck_run(run, recheck=recheck)
    assert sorted(recheck.calls) == ["a", "s"]
    again = fake_recheck({"s": (True, 0.3), "a": (True, 0.5)})
    recheck_run(run, recheck=again)  # rechecked.jsonl is reused
    assert again.calls == []


def test_curve_and_cloud_cost(tmp_path):
    calls = [{"call_index": 0, "model": "argon", "ts_start": 10.0, "latency_s": 60.0, "prompt_tokens": 100,
              "completion_tokens": 50, "status": "ok"},
             {"call_index": 1, "model": "gemma", "ts_start": 20.0, "latency_s": 3.0, "prompt_tokens": 9,
              "completion_tokens": 9, "status": "error:URLError"}]
    run = make_run(tmp_path, [cand("s", 0.3, 1), cand("a", 0.6, 11)], calls=calls, arm="LC")
    summary = recheck_run(run, recheck=fake_recheck({"s": (True, 0.3), "a": (True, 0.6)}))
    assert summary["curve"] == [[0, 0.6, 0.6], [1, 0.6, 0.6]]
    assert summary["argon_calls"] == 1 and summary["argon_tokens"] == 150
    assert summary["argon_wall_s"] == 60.0 and summary["gemma_wall_s"] == 3.0
    assert summary["failed_calls_by_status"] == {"error:URLError": 1}


def test_analyze_aggregates_and_draws(tmp_path):
    for seed, final in ((101, 0.5), (202, 0.7)):
        make_run(tmp_path, [cand("s", 0.3, 1), cand(f"x{seed}", final, 11)], seed=seed)
    table = {"s": (True, 0.3), "x101": (True, 0.5), "x202": (True, 0.7)}
    figures = tmp_path / "figs"
    analyze.main(tmp_path, figures_dir=figures, recheck=fake_recheck(table))
    summary = json.loads((tmp_path / "summary.json").read_text())
    group = summary["circle_packing"]["L"]
    assert group["runs"] == 2
    assert group["shipped_verified_score"]["median"] == pytest.approx(0.6)
    assert group["shipped_verified_score"]["min"] == 0.5 and group["shipped_verified_score"]["max"] == 0.7
    assert (figures / "fig_curves_circle_packing.png").stat().st_size > 0
    assert (figures / "fig_false_improvements.png").stat().st_size > 0
