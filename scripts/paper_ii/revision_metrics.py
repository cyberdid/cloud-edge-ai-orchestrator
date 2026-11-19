#!/usr/bin/env python3
"""Rebuild Paper II revision metrics from committed run summaries and rechecks.

No model, benchmark, sandbox or network calls are made. Long-budget best-valid
scores are transcribed at the precision published in COLAB_SCALE_RESULTS.md;
selected endpoints and kNN measurements are reconstructed from machine-readable
committed artifacts.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "docs/paper_ii"
RESULTS = PAPER / "results"
FIGURES = PAPER / "figures"
ANALYSIS = ROOT / "docs/colab_scale/results/exploratory_tests_2026-11-19.json"
SERIAL = ROOT / "docs/colab_scale/results/knn_serial_retiming_2026-11-19.jsonl"
PAIRS = ROOT / "docs/colab_scale/results/lc_paired_outcomes_2026-11-08.csv"
OUTCOMES = ROOT / "docs/colab_scale/results/lc_run_outcomes_2026-11-08.csv"
ABOVE = ROOT / "docs/colab_scale/results/lc_above_seed_candidates_2026-11-08.csv"
FINAL_INVENTORY = ROOT / "docs/colab_scale/FINAL_RUNS_2026-11-11.json"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def median(xs: list[float]) -> float:
    if not xs:
        raise ValueError("median of empty observations")
    return float(statistics.median(xs))


def build_metrics() -> dict:
    inventory = json.loads(FINAL_INVENTORY.read_text())
    statuses: dict[str, int] = defaultdict(int)
    for run in inventory["runs"]:
        statuses[run["status"]] += 1
    if inventory["planned"] != 179 or len(inventory["runs"]) != 179 or dict(statuses) != {
        "complete": 167, "missing": 10, "paused_quota": 2
    }:
        raise ValueError(f"unexpected amended Colab inventory: {inventory['planned']=}, {len(inventory['runs'])=}, {dict(statuses)=}")
    analysis = json.loads(ANALYSIS.read_text())
    long_runs = [
        r for r in analysis["runs"]
        if r.get("model") == "gemma4-26b"
        and r.get("task") == "circle_packing"
        and r.get("arm") in {"L-long", "B-long"}
    ]
    expected = {("L-long", 101), ("L-long", 202), ("L-long", 303),
                ("B-long", 101), ("B-long", 202), ("B-long", 303)}
    if {(r["arm"], r["seed"]) for r in long_runs} != expected:
        raise ValueError("long-budget run inventory does not match the six expected runs")

    # The published report gives best-valid scores to 3 decimals. Preserve that
    # precision explicitly; do not imply access to omitted raw run directories.
    reported_best = {
        ("L-long", 101): 2.432,
        ("L-long", 202): 2.204,
        ("L-long", 303): 2.258,
        ("B-long", 101): 1.936,
        ("B-long", 202): 2.072,
        ("B-long", 303): 1.924,
    }
    long_selection = []
    for run in sorted(long_runs, key=lambda r: (r["arm"], r["seed"])):
        best = reported_best[(run["arm"], run["seed"])]
        selected = float(run["shipped"])
        long_selection.append({
            "run_id": f"{run['arm']}_s{run['seed']}",
            "task": "circle_packing",
            "arm": run["arm"],
            "seed": run["seed"],
            "budget_calls": 1000,
            "best_valid_reported": best,
            "selected_verified": selected,
            "selected_status": "valid" if selected > 0 else "invalid_selection_seed_fallback",
            "selection_regret_approx": round(best - selected, 6),
            "precision": "best-valid value reported to 3 decimals; selected endpoint from JSON at full stored precision",
            "sources": [
                "docs/COLAB_SCALE_RESULTS.md: long-budget circle-packing table",
                "docs/colab_scale/results/exploratory_tests_2026-11-19.json:runs",
            ],
        })

    serial_rows = read_jsonl(SERIAL)
    serial_by_hash: dict[str, list[dict]] = defaultdict(list)
    for row in serial_rows:
        serial_by_hash[row["code_sha256"]].append(row)
    seed_rows = [r for r in serial_rows if r["kind"] == "seed"]
    seed_hashes = {r["code_sha256"] for r in seed_rows}
    if len(seed_hashes) != 1 or len(seed_rows) != 5:
        raise ValueError("unexpected designated kNN seed calibration inventory")
    seed_hash = next(iter(seed_hashes))
    seed_all = [float(r["recheck"]["score"]) for r in serial_by_hash[seed_hash]]
    seed_score = median(seed_all)
    if len(seed_all) != 8:
        raise ValueError(f"expected 8 pooled seed observations, found {len(seed_all)}")

    outcomes = read_csv(OUTCOMES)
    outcome_by_id = {r["run_id"]: r for r in outcomes}
    pair_rows = [r for r in read_csv(PAIRS) if r["task"] == "knn"]
    if len(pair_rows) != 9:
        raise ValueError(f"expected 9 completed kNN pairs, found {len(pair_rows)}")
    knn_pairs = []
    selected_hashes: set[str] = {seed_hash}
    lc_selected: list[dict] = []
    for pair in sorted(pair_rows, key=lambda r: int(r["seed"])):
        pair_out = {}
        for arm in ("L", "LC"):
            run_id = f"knn_{arm}_s{pair['seed']}"
            row = outcome_by_id[run_id]
            code_hash = row["best_code_sha256"]
            selected_hashes.add(code_hash)
            checks = [x["recheck"] for x in serial_by_hash.get(code_hash, [])]
            if not checks:
                raise ValueError(f"no serial measurement for selected program {run_id} ({code_hash})")
            valid = all(bool(x["valid"]) for x in checks)
            selected_score = median([float(x["score"]) for x in checks]) if valid else seed_score
            speed_ratios = [float(x["holdout_speed_ratio"]) for x in checks if x.get("holdout_speed_ratio") is not None]
            recalls = [float(x["holdout_recall"]) for x in checks if x.get("holdout_recall") is not None]
            pair_out[arm] = {
                "run_id": run_id,
                "seed": int(pair["seed"]),
                "code_sha256": code_hash,
                "valid_in_serial_recheck": valid,
                "distance_correctness": "passed independent check_neighbors and exact-copy rules" if valid else "failed independent validation",
                "holdout_recall_median": median(recalls) if recalls else None,
                "holdout_speed_ratio_median": median(speed_ratios) if speed_ratios else None,
                "verified_score_median": selected_score,
                "repetitions": len(checks),
                "source": [
                    "docs/colab_scale/results/lc_run_outcomes_2026-11-08.csv",
                    "docs/colab_scale/results/knn_serial_retiming_2026-11-19.jsonl",
                    "experiments/openevolve/tasks/knn/recheck.py",
                ],
            }
            if arm == "LC":
                lc_selected.append(pair_out[arm])
        knn_pairs.append({
            "seed": int(pair["seed"]),
            "L": pair_out["L"],
            "LC": pair_out["LC"],
            "LC_minus_L_verified_score": pair_out["LC"]["verified_score_median"] - pair_out["L"]["verified_score_median"],
        })

    # Rank stability is limited to the ten unique programs actually selected in
    # the nine LC/L pairs (one shared seed program plus nine LC programs).
    if len(selected_hashes) != 10:
        raise ValueError(f"expected 10 unique selected programs, found {len(selected_hashes)}")
    by_rep: dict[int, dict[str, float]] = {i: {} for i in (0, 1, 2)}
    for code_hash in selected_hashes:
        reps = {int(r["rep"]): float(r["recheck"]["score"])
                for r in serial_by_hash[code_hash]
                if r["kind"] == "shipped" and int(r["rep"]) in (0, 1, 2)}
        if set(reps) != {0, 1, 2}:
            raise ValueError(f"incomplete three-repeat measurements for selected hash {code_hash}")
        for rep, score in reps.items():
            by_rep[rep][code_hash] = score
    hash_order = sorted(selected_hashes)
    pairwise_rank = []
    for left, right in ((0, 1), (0, 2), (1, 2)):
        tau = float(kendalltau([by_rep[left][h] for h in hash_order],
                               [by_rep[right][h] for h in hash_order], variant="b").statistic)
        left_top = set(sorted(hash_order, key=lambda h: (-by_rep[left][h], h))[:5])
        right_top = set(sorted(hash_order, key=lambda h: (-by_rep[right][h], h))[:5])
        pairwise_rank.append({"rep_a": left, "rep_b": right, "kendall_tau_b": tau,
                              "top5_overlap": len(left_top & right_top), "top_k": 5})
    crossing = [r for r in lc_selected
                if min(float(x["recheck"]["score"]) for x in serial_by_hash[r["code_sha256"]]) < seed_score
                <= max(float(x["recheck"]["score"]) for x in serial_by_hash[r["code_sha256"]])]

    above_rows = [r for r in read_csv(ABOVE) if r["task"] == "knn" and r["arm"] == "LC"]
    lc_false = analysis["false_improvements_pooled"]["gemma4-26b/knn/LC"]
    invalid_evaluations = sum(r["recheck_valid"].lower() == "false" for r in above_rows)
    if len(above_rows) != 213 or int(lc_false["false"]) != 60 or invalid_evaluations != 4:
        raise ValueError("LC kNN false-improvement denominators do not match committed artifacts")

    return {
        "schema_version": 1,
        "generated_by": "scripts/paper_ii/revision_metrics.py",
        "reproducible_without_model_or_sandbox_calls": True,
        "colab_final_inventory": {
            "planned": inventory["planned"],
            "completed": statuses["complete"],
            "not_started": statuses["missing"],
            "partial_quota_paused": statuses["paused_quota"],
            "source": "docs/colab_scale/FINAL_RUNS_2026-11-11.json",
        },
        "selection_regret": {
            "definition": "max(seed_verified_score, verified score among valid candidates in the reported complete pool) - selected endpoint; negative values are forbidden",
            "coverage_note": "Complete candidate-level archives are not in this Git snapshot. Long-budget circle best-valid scores are transcribed at 3-decimal precision from the published report. No all-candidate kNN regret is estimated.",
            "runs": long_selection,
            "L_long_invalid_selected_runs": sum(r["selected_status"] != "valid" for r in long_selection if r["arm"] == "L-long"),
            "L_long_runs": 3,
            "L_long_median_approx_regret": median([r["selection_regret_approx"] for r in long_selection if r["arm"] == "L-long"]),
            "B_long_median_approx_regret": median([r["selection_regret_approx"] for r in long_selection if r["arm"] == "B-long"]),
        },
        "knn": {
            "seed_score_pooled_median": seed_score,
            "seed_score_designated_calibration_median": median([float(r["recheck"]["score"]) for r in seed_rows]),
            "lc_l_pairs": knn_pairs,
            "lc_selected_valid_count": sum(r["valid_in_serial_recheck"] for r in lc_selected),
            "lc_selected_n": len(lc_selected),
            "lc_selected_holdout_recall_median_range": [
                min(r["holdout_recall_median"] for r in lc_selected),
                max(r["holdout_recall_median"] for r in lc_selected),
            ],
            "lc_selected_holdout_speed_ratio_medians": [
                {"seed": r["seed"], "ratio": r["holdout_speed_ratio_median"]} for r in lc_selected
            ],
            "speed_ratio_definition": "maximum candidate/exact elapsed-time ratio across three holdout sets; lower is faster; not absolute latency",
            "distance_check": "Validity includes the independent check_neighbors contract, true squared distances, exact-copy rule and static restrictions.",
            "rank_stability_selected_programs": {
                "n_unique_code_hashes": len(selected_hashes),
                "repetition_ids_are_ordinal_not_synchronized": True,
                "comparisons": pairwise_rank,
                "selected_lc_hashes_cross_seed_median_threshold": len(crossing),
                "selected_lc_hashes_n": len(lc_selected),
                "top5_tie_break": "code_sha256 ascending",
            },
            "false_improvements": {
                "candidate_evaluations_above_working_seed": len(above_rows),
                "false_candidate_evaluations": int(lc_false["false"]),
                "false_rate": int(lc_false["false"]) / len(above_rows),
                "invalid": invalid_evaluations,
                "valid_but_verified_score_not_above_seed": int(lc_false["false"]) - invalid_evaluations,
                "evaluations_not_unique_programs": True,
            },
        },
        "sources": {
            "final_inventory": "docs/colab_scale/FINAL_RUNS_2026-11-11.json",
            "run_level_summary": "docs/colab_scale/results/exploratory_tests_2026-11-19.json",
            "published_long_best_valid": "docs/COLAB_SCALE_RESULTS.md",
            "lc_run_inventory": "docs/colab_scale/results/lc_run_outcomes_2026-11-08.csv",
            "lc_pairs": "docs/colab_scale/results/lc_paired_outcomes_2026-11-08.csv",
            "lc_above_seed_evaluations": "docs/colab_scale/results/lc_above_seed_candidates_2026-11-08.csv",
            "serial_rechecks": "docs/colab_scale/results/knn_serial_retiming_2026-11-19.jsonl",
            "registered_recheck_implementation": "experiments/openevolve/tasks/knn/recheck.py",
            "metric_definitions": "experiments/openevolve/tasks/knn/working_evaluator.py",
        },
    }


def make_figure(data: dict, language: str) -> Path:
    import matplotlib.pyplot as plt

    FIGURES.mkdir(parents=True, exist_ok=True)
    is_uk = language == "uk"
    colors = {"L-long": "#0072B2", "B-long": "#D55E00", "L": "#0072B2", "LC": "#CC79A7"}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)

    # Left: long-run best-valid versus the endpoint actually selected.
    ax = axes[0]
    title = "Довгий пошук кіл: знайдено та обрано" if is_uk else "Long circle search: found vs selected"
    ax.set_title(title, fontsize=11, weight="bold")
    for arm in ("L-long", "B-long"):
        rows = [r for r in data["selection_regret"]["runs"] if r["arm"] == arm]
        x = [r["seed"] for r in rows]
        best = [r["best_valid_reported"] for r in rows]
        chosen = [r["selected_verified"] for r in rows]
        c = colors[arm]
        ax.plot(x, best, marker="o", linestyle="-", color=c,
                label=("L: найкращий валідний" if is_uk else "L: best valid") if arm == "L-long" else ("B: найкращий валідний" if is_uk else "B: best valid"))
        ax.plot(x, chosen, marker="x", linestyle="--", color=c, alpha=0.8,
                label=("L: обраний endpoint" if is_uk else "L: selected endpoint") if arm == "L-long" else ("B: обраний endpoint" if is_uk else "B: selected endpoint"))
    ax.set_xlabel("Seed")
    ax.set_ylabel("Сума радіусів / radius sum")
    ax.set_xticks([101, 202, 303], ["101", "202", "303"])
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=7.6, loc="lower left", frameon=False)
    ax.text(0.98, 0.96, "L: 2/3 invalid\nB: 0/3 invalid" if not is_uk else "L: 2/3 невалідні\nB: 0/3 невалідні",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox={"boxstyle": "round,pad=.35", "fc": "white", "ec": "#bbbbbb", "alpha": .9})

    # Right: paired kNN selected endpoint, reconstructed from serial re-checks.
    ax = axes[1]
    ax.set_title("kNN: L та LC, 9 matched pairs" if not is_uk else "kNN: L та LC, 9 пар",
                 fontsize=11, weight="bold")
    for row in data["knn"]["lc_l_pairs"]:
        x = row["seed"]
        y_l = row["L"]["verified_score_median"]
        y_lc = row["LC"]["verified_score_median"]
        ax.plot([0, 1], [y_l, y_lc], color="#999999", linewidth=0.9, alpha=0.7, zorder=1)
        ax.scatter(0, y_l, color=colors["L"], s=22, zorder=2)
        ax.scatter(1, y_lc, color=colors["LC"], s=22, zorder=2)
        ax.text(1.04, y_lc, str(x), fontsize=6.5, va="center", color="#555555", clip_on=False)
    ax.set_xticks([0, 1], ["L", "LC"])
    ax.set_xlim(-0.25, 1.22)
    ax.set_ylim(-0.04, 1.08)
    ax.set_ylabel("Перевірений бал / verified score")
    ax.grid(axis="y", alpha=0.25)
    summary = "Медіана Δ=0.776; 8/9 вище" if is_uk else "Median Δ=0.776; 8/9 higher"
    ax.text(0.98, 0.96, summary, transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox={"boxstyle": "round,pad=.35", "fc": "white", "ec": "#bbbbbb", "alpha": .9})
    fig.suptitle("Аудит selection / Selection audit", fontsize=13, weight="bold")
    path = FIGURES / f"fig4_selection_audit_{language}.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=RESULTS / "revision_metrics.json")
    parser.add_argument("--figures", action="store_true", help="also regenerate EN/UK PDF figures")
    args = parser.parse_args()
    data = build_metrics()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {output.relative_to(ROOT) if output.is_relative_to(ROOT) else output}")
    if args.figures:
        for language in ("en", "uk"):
            print(f"Wrote {make_figure(data, language).relative_to(ROOT)}")
    print(json.dumps({
        "long_L_invalid_selected": f"{data['selection_regret']['L_long_invalid_selected_runs']}/3",
        "long_L_median_regret_approx": data["selection_regret"]["L_long_median_approx_regret"],
        "kNN_LC_vs_L_higher": sum(p["LC_minus_L_verified_score"] > 0 for p in data["knn"]["lc_l_pairs"]),
        "kNN_LC_median_paired_difference": median([p["LC_minus_L_verified_score"] for p in data["knn"]["lc_l_pairs"]]),
        "kNN_LC_false_above_seed": f"{data['knn']['false_improvements']['false_candidate_evaluations']}/{data['knn']['false_improvements']['candidate_evaluations_above_working_seed']}",
        "kNN_selected_rank_tau_b": [round(x["kendall_tau_b"], 3) for x in data["knn"]["rank_stability_selected_programs"]["comparisons"]],
    }, indent=2))


if __name__ == "__main__":
    main()
