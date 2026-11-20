"""Descriptive-only summary of the fixed B2 selected-program re-check."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


ARMS = ("v1_matched", "v2")


def summarize(report: dict[str, Any], *, expected_pairs: int = 6) -> dict[str, Any]:
    rows = report.get("runs")
    if not isinstance(rows, list) or len(rows) != expected_pairs * 2:
        raise ValueError("report must contain exactly two assigned-arm rows per pair")
    if report.get("assigned_runs") != expected_pairs * 2:
        raise ValueError("assigned_runs does not match the fixed analysis denominator")
    seen = set()
    per_pair: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        pair_id, arm = row.get("pair_id"), row.get("evaluator_version")
        if not isinstance(pair_id, str) or arm not in ARMS:
            raise ValueError("each row requires a pair_id and known evaluator_version")
        run_id = row.get("run_id")
        if not isinstance(run_id, str) or run_id in seen:
            raise ValueError("run IDs must be present and unique")
        seen.add(run_id)
        if arm in per_pair.setdefault(pair_id, {}):
            raise ValueError(f"duplicate arm row for pair {pair_id}")
        if not isinstance(row.get("valid"), bool):
            raise ValueError(f"{run_id} validity must be a boolean")
        score = row.get("score", 0.0)
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError(f"{run_id} score must be finite numeric")
        per_pair[pair_id][arm] = row
    if len(per_pair) != expected_pairs:
        raise ValueError("pair count does not match the fixed analysis denominator")
    if any(set(arms) != set(ARMS) for arms in per_pair.values()):
        raise ValueError("every assigned pair must contain both arms")

    table = {"both_valid": 0, "v1_valid_v2_invalid": 0,
             "v1_invalid_v2_valid": 0, "both_invalid": 0}
    for arms in per_pair.values():
        v1, v2 = arms["v1_matched"]["valid"], arms["v2"]["valid"]
        if v1 and v2:
            table["both_valid"] += 1
        elif v1:
            table["v1_valid_v2_invalid"] += 1
        elif v2:
            table["v1_invalid_v2_valid"] += 1
        else:
            table["both_invalid"] += 1

    arm_summaries = {}
    for arm in ARMS:
        values = [pair[arm] for pair in per_pair.values()]
        valid_count = sum(row["valid"] for row in values)
        arm_summaries[arm] = {
            "assigned": len(values), "valid": valid_count,
            "invalid_or_missing": len(values) - valid_count,
            "valid_rate": valid_count / len(values),
            "mean_selected_score_including_failures_as_zero":
                sum(float(row.get("score", 0.0)) if row["valid"] else 0.0 for row in values) / len(values),
        }
    risk_difference = arm_summaries["v2"]["invalid_or_missing"] / expected_pairs - \
                      arm_summaries["v1_matched"]["invalid_or_missing"] / expected_pairs
    return {
        "analysis_type": "descriptive_estimation_only",
        "assigned_pairs": expected_pairs,
        "complete_pairs": sum(
            all(not (isinstance(arms[arm].get("reason"), str) and
                     arms[arm]["reason"] in {"missing_manifest", "missing_official_selection",
                                              "missing_selected_program", "selected_program_hash_mismatch",
                                              "run_not_complete"})
                for arm in ARMS) for arms in per_pair.values()),
        "v1_matched": arm_summaries["v1_matched"], "v2": arm_summaries["v2"],
        "paired_table": table,
        "risk_difference_invalid_rate_v2_minus_v1": risk_difference,
        "confirmatory": False,
        "interpretation": "descriptive only; no efficacy, equivalence, or generalization claim",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, default=Path("b2-estimation-summary.json"))
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    result = summarize(report)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
