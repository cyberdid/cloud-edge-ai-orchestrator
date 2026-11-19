"""Exploratory (not pre-registered) significance tests for the OpenEvolve pilot and the
Colab scale study. Both pre-registrations specify descriptive reporting only; these tests
are added afterwards and are labelled exploratory wherever they are reported.

Inputs: the run folders with their re-check caches, plus the serial kNN re-timing files.
Endpoints are the registered ones: shipped verified score (seed's verified score when the
working-best program fails the re-check). Circle packing uses the cached re-check; kNN uses
the serial re-timing (median per program), because parallel timings were unusable.

usage: python scripts/analysis/oe_exploratory_tests.py COLAB_RUNS PILOT_RUNS SERIAL_DIR OUT.json
(run with the orchestrator repo on PYTHONPATH)
"""
from __future__ import annotations

import json
import re
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.openevolve.recheck_all import _lines, attribute

BOOT = 10_000
RNG = np.random.default_rng(20261119)
PILOT_FIGURE_PATH = Path(__file__).resolve().parents[2] / "docs/openevolve_pilot/fig_false_improvements_serial_2026-11-19.png"


@dataclass(frozen=True)
class SerialScores:
    by_hash: dict[str, list[dict]]
    seed_hash: str
    seed_score: float
    seed_calibration_score: float

    def value(self, code_hash: str, seed_score: float | None = None) -> tuple[bool, float]:
        # The seed hash has five designated calibration rows and three further rows
        # from the same code when it was selected as a shipped program. Pool all
        # measurements for the main per-program median; expose the five-row median
        # separately for sensitivity analysis.
        if code_hash == self.seed_hash:
            return True, self.seed_score if seed_score is None else seed_score
        rows = self.by_hash.get(code_hash)
        if not rows:
            raise KeyError(f"no serial re-timing found for candidate {code_hash}")
        scores = [row["recheck"]["score"] for row in rows]
        if not all(np.isfinite(score) for score in scores):
            raise ValueError(f"non-finite serial score for candidate {code_hash}")
        valid = all(row["recheck"]["valid"] for row in rows)
        return (True, statistics.median(scores)) if valid else (False, 0.0)


def serial_scores(serial_dir: Path) -> SerialScores:
    out: dict[str, list[dict]] = defaultdict(list)
    seed_rows: list[dict] = []
    primary_name = next((name for name in (
        "knn_serial_retiming_2026-11-19.jsonl", "knn_serial.jsonl"
    ) if (serial_dir / name).is_file()), None)
    if primary_name is None:
        raise FileNotFoundError(f"no serial seed timing file in {serial_dir}")
    secondary_candidate = (
        "knn_serial_above_seed_2026-11-19.jsonl"
        if primary_name == "knn_serial_retiming_2026-11-19.jsonl"
        else "knn_serial_above.jsonl"
    )
    secondary_name = secondary_candidate if (serial_dir / secondary_candidate).is_file() else None
    for name in (primary_name, secondary_name):
        if name is None:
            continue
        for row in _lines(serial_dir / name):
            out[row["code_sha256"]].append(row)
            if row.get("kind") == "seed":
                seed_rows.append(row)
    seed_hashes = {row["code_sha256"] for row in seed_rows}
    seed_reps = {row["rep"] for row in seed_rows}
    if len(seed_hashes) != 1 or len(seed_rows) != 5 or seed_reps != set(range(5)):
        raise ValueError(f"expected five designated seed measurements for one hash; got {len(seed_rows)} rows and {len(seed_hashes)} hashes")
    if not all(row["recheck"]["valid"] for row in seed_rows):
        raise ValueError("a designated seed calibration repetition failed validation")
    seed_hash = next(iter(seed_hashes))
    all_seed_rows = out[seed_hash]
    if not all(row["recheck"]["valid"] for row in all_seed_rows):
        raise ValueError("a repeated measurement of the seed code failed validation")
    if not all(np.isfinite(row["recheck"]["score"]) for row in all_seed_rows):
        raise ValueError("a seed measurement has a non-finite score")
    seed_score = statistics.median(row["recheck"]["score"] for row in all_seed_rows)
    seed_calibration_score = statistics.median(row["recheck"]["score"] for row in seed_rows)
    return SerialScores(dict(out), seed_hash, seed_score, seed_calibration_score)


def run_endpoint(run: Path, serial: SerialScores, seed_score_override: float | None = None) -> dict:
    manifest = json.loads((run / "manifest.json").read_text())
    cands = attribute(_lines(run / "calls.jsonl"), _lines(run / "candidates.jsonl"))
    cache = {r["code_sha256"]: r["recheck"] for r in _lines(run / "rechecked.jsonl")}
    for c in cands:
        c["w"] = float(c["metrics"].get("combined_score", 0.0))
    seed = next(c for c in cands if c["call_index"] is None)
    shipped = max(cands, key=lambda c: c["w"])
    above = [c for c in cands if c["call_index"] is not None and c["w"] > seed["w"]]
    if manifest["task"] == "knn":
        _, seed_v = serial.value(seed["code_sha256"], seed_score_override)
        if shipped["call_index"] is None or not cache[shipped["code_sha256"]]["valid"]:
            ship_v = seed_v  # seed shipped, or shipped program invalid (validity is timing-independent)
        else:
            ok, v = serial.value(shipped["code_sha256"], seed_v)
            ship_v = v if ok else seed_v
        false = 0
        for c in above:
            if not cache[c["code_sha256"]]["valid"]:
                false += 1
                continue
            ok, v = serial.value(c["code_sha256"], seed_v)
            false += (not ok) or v <= seed_v
    else:
        s = cache[seed["code_sha256"]]
        seed_v = s["score"] if s["valid"] else 0.0
        r = cache[shipped["code_sha256"]]
        ship_v = r["score"] if r["valid"] else seed_v
        false = sum(1 for c in above if not cache[c["code_sha256"]]["valid"]
                    or cache[c["code_sha256"]]["score"] <= seed_v)
    m = re.search(r"_(L|LC|B)_s(\d+)", run.name)
    return {"model": run.parent.name, "task": manifest["task"], "arm": m.group(1) + ("-long" if "long" in run.name else ""),
            "seed": int(m.group(2)), "shipped": ship_v, "seed_verified": seed_v,
            "above": len(above), "false": false}


def boot_median_ci(diffs: list[float]) -> list[float]:
    arr = np.asarray(diffs, float)
    meds = np.median(RNG.choice(arr, size=(BOOT, arr.size), replace=True), axis=1)
    return [float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))]


def paired(a: dict[int, float], b: dict[int, float]) -> dict:
    seeds = sorted(set(a) & set(b))
    d = [b[s] - a[s] for s in seeds]
    pos, neg = sum(x > 0 for x in d), sum(x < 0 for x in d)
    sign_p = stats.binomtest(pos, pos + neg, 0.5).pvalue if pos + neg else 1.0
    nz = [x for x in d if x != 0]
    wil_p = stats.wilcoxon(nz).pvalue if nz else 1.0  # zero differences dropped; exact when no ties
    return {"pairs": len(seeds), "better/equal/worse": [pos, len(d) - pos - neg, neg],
            "median_diff": statistics.median(d), "median_diff_ci95_boot": boot_median_ci(d),
            "sign_test_p": sign_p, "wilcoxon_p": float(wil_p)}


def unpaired(a: list[float], b: list[float]) -> dict:
    if not a or not b:
        raise ValueError("Mann–Whitney comparison requires observations in both groups")
    if not all(np.isfinite(v) for v in [*a, *b]):
        raise ValueError("Mann–Whitney comparison received a non-finite observation")
    p_value = 1.0 if len(set([*a, *b])) == 1 else float(
        stats.mannwhitneyu(b, a, alternative="two-sided", method="auto").pvalue
    )
    return {"n": [len(a), len(b)], "median": [statistics.median(a), statistics.median(b)],
            "mann_whitney_p": p_value}


def holm(ps: dict[str, float]) -> dict[str, float]:
    order = sorted(ps, key=ps.get)
    adj, running = {}, 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (len(order) - i) * ps[k]))
        adj[k] = running
    return adj


def plot_pilot_false_improvements(pooled: dict[str, dict], path: Path) -> None:
    tasks = ("circle_packing", "knn")
    arms = ("L", "B")
    colors = {"L": "#2a78d6", "B": "#1baf7a"}
    x = np.arange(len(tasks), dtype=float)
    width = 0.32
    fig, ax = plt.subplots(figsize=(7.6, 4.2), dpi=180)
    for offset, arm in enumerate(arms):
        rates, labels = [], []
        for task in tasks:
            item = pooled[f"pilot-12b/{task}/{arm}"]
            rates.append(item["rate"])
            labels.append(f"{item['false']}/{item['above_seed']} ({item['rate']:.0%})")
        bars = ax.bar(x + (offset - 0.5) * width, rates, width, color=colors[arm], label=arm)
        ax.bar_label(bars, labels=labels, padding=3, fontsize=8)
    ax.set_xticks(x, ["Circle packing", "kNN"])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("False-improvement rate")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("False-improvement rates after serial kNN re-timing", loc="left", fontsize=12)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e4e3de", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.text(0.01, 0.01, "Circle-packing results unchanged. kNN compares against the pooled seed-code median (8 serial measurements). LC was not run.", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main(colab_runs: Path, pilot_runs: Path, serial_dir: Path, out: Path) -> dict:
    serial = serial_scores(serial_dir)
    runs = []
    seed_only_runs = []
    for manifest in sorted(colab_runs.glob("*/*/manifest.json")) + sorted(pilot_runs.glob("*/manifest.json")):
        if json.loads(manifest.read_text()).get("status") == "complete":
            r = run_endpoint(manifest.parent, serial)
            seed_only = run_endpoint(manifest.parent, serial, serial.seed_calibration_score)
            if manifest.parent.parent == pilot_runs:
                r["model"] = "pilot-12b"
                seed_only["model"] = "pilot-12b"
            runs.append(r)
            seed_only_runs.append(seed_only)

    def sel(model, task, arm, lo=0, hi=1010):
        return {r["seed"]: r["shipped"] for r in runs
                if r["model"] == model and r["task"] == task and r["arm"] == arm and lo <= r["seed"] <= hi}

    res: dict = {"note": "EXPLORATORY: not pre-registered; both studies registered descriptive reporting only."}
    # RQ-cloud: MoE LC vs L, paired by seed 101-909
    res["lc_vs_l"] = {t: paired(sel("gemma4-26b", t, "L", hi=909), sel("gemma4-26b", t, "LC", hi=909))
                      for t in ("circle_packing", "knn")}
    # L vs B per model (unpaired)
    lb = {}
    for model in ("gemma4-e4b", "gemma4-12b", "gemma4-26b", "gemma4-31b", "pilot-12b"):
        for t in ("circle_packing", "knn"):
            a, b = list(sel(model, t, "B").values()), list(sel(model, t, "L").values())
            if a and b:
                lb[f"{model}/{t}"] = unpaired(a, b)
    res["l_vs_b"] = lb
    res["l_vs_b_holm"] = holm({k: v["mann_whitney_p"] for k, v in lb.items()})
    # size ladder, circle packing, arm L
    ladder = {m: list(sel(m, "circle_packing", "L").values()) for m in ("gemma4-e4b", "gemma4-12b", "gemma4-26b", "gemma4-31b")}
    ladder_pairwise = {
        "moe_vs_31b": unpaired(ladder["gemma4-31b"], ladder["gemma4-26b"]),
        "moe_vs_12b": unpaired(ladder["gemma4-12b"], ladder["gemma4-26b"]),
        "moe_vs_e4b": unpaired(ladder["gemma4-e4b"], ladder["gemma4-26b"]),
    }
    res["size_ladder_circle_L"] = {
        "kruskal_p": float(stats.kruskal(*ladder.values()).pvalue),
        **ladder_pairwise,
        "pairwise_holm": holm({k: v["mann_whitney_p"] for k, v in ladder_pairwise.items()}),
    }
    # false-improvement rate per run, MoE L vs LC paired (circle)
    def fi(model, task, arm):
        return {r["seed"]: (r["false"] / r["above"] if r["above"] else 0.0) for r in runs
                if r["model"] == model and r["task"] == task and r["arm"] == arm and r["seed"] <= 909}
    fi_pair = paired(fi("gemma4-26b", "circle_packing", "L"), fi("gemma4-26b", "circle_packing", "LC"))
    pos, equal, neg = fi_pair.pop("better/equal/worse")
    fi_pair["lower/equal/higher"] = [neg, equal, pos]
    res["fi_rate_lc_vs_l_circle"] = fi_pair
    # pooled false-improvement counts with the serial kNN timing
    pooled = defaultdict(lambda: [0, 0])
    for r in runs:
        key = f"{r['model']}/{r['task']}/{r['arm']}" + ("/ext" if r["model"] == "gemma4-26b" and r["seed"] > 1010 else "")
        pooled[key][0] += r["false"]; pooled[key][1] += r["above"]
    res["false_improvements_pooled"] = {k: {"false": v[0], "above_seed": v[1], "rate": v[0] / v[1] if v[1] else None}
                                        for k, v in sorted(pooled.items())}
    seed_only_pilot_l = [r for r in seed_only_runs if r["model"] == "pilot-12b" and r["task"] == "knn" and r["arm"] == "L"]
    seed_only_lc = [r for r in seed_only_runs if r["model"] == "gemma4-26b" and r["task"] == "knn" and r["arm"] == "LC"]
    seed_only_l = {r["seed"]: r["shipped"] for r in seed_only_runs
                   if r["model"] == "gemma4-26b" and r["task"] == "knn" and r["arm"] == "L" and r["seed"] <= 909}
    seed_only_lc_map = {r["seed"]: r["shipped"] for r in seed_only_lc}
    seed_only_false = sum(r["false"] for r in seed_only_pilot_l)
    seed_only_above = sum(r["above"] for r in seed_only_pilot_l)
    res["seed_calibration_sensitivity"] = {
        "alternate_seed_score": serial.seed_calibration_score,
        "pilot_knn_L_false_above": [seed_only_false, seed_only_above],
        "colab_moe_LC_knn_false_above": [sum(r["false"] for r in seed_only_lc), sum(r["above"] for r in seed_only_lc)],
        "colab_moe_LC_vs_L_knn_pairs": paired(seed_only_l, seed_only_lc_map),
    }
    res["runs"] = runs
    res["knn_seed_calibration"] = {
        "code_sha256": serial.seed_hash,
        "all_same_hash_repetitions": len(serial.by_hash[serial.seed_hash]),
        "median_score": serial.seed_score,
        "range_all_repetitions": [min(row["recheck"]["score"] for row in serial.by_hash[serial.seed_hash]),
                                  max(row["recheck"]["score"] for row in serial.by_hash[serial.seed_hash])],
        "designated_calibration_repetitions": 5,
        "designated_calibration_median": serial.seed_calibration_score,
        "designated_calibration_range": [min(row["recheck"]["score"] for row in serial.by_hash[serial.seed_hash] if row.get("kind") == "seed"),
                                          max(row["recheck"]["score"] for row in serial.by_hash[serial.seed_hash] if row.get("kind") == "seed")],
        "policy": "main endpoint uses the median of every serial measurement with the seed code hash; five designated calibration and three shipped-program repeats are both included",
    }
    repeated_ranges = {
        code_hash: max(row["recheck"]["score"] for row in rows) - min(row["recheck"]["score"] for row in rows)
        for code_hash, rows in serial.by_hash.items() if len(rows) > 1
    }
    largest_range_hash = max(repeated_ranges, key=repeated_ranges.get)
    largest_range_scores = sorted(row["recheck"]["score"] for row in serial.by_hash[largest_range_hash])
    res["serial_repeat_variation"] = {
        "definition": "within-code-hash range of serial score repetitions",
        "repeated_hashes": len(repeated_ranges),
        "median_range": statistics.median(repeated_ranges.values()),
        "hashes_with_range_over_0_05": sum(value > 0.05 for value in repeated_ranges.values()),
        "largest_range": repeated_ranges[largest_range_hash],
        "largest_range_scores": largest_range_scores,
    }
    out.write_text(json.dumps(res, indent=1, allow_nan=False) + "\n")
    plot_pilot_false_improvements(res["false_improvements_pooled"], PILOT_FIGURE_PATH)
    return res


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
