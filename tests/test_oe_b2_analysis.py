import pytest


def report():
    rows = []
    outcomes = [(True, True), (False, True), (True, False), (True, True), (False, False), (True, True)]
    for i, (v1_valid, v2_valid) in enumerate(outcomes, start=1):
        for version, valid in (("v1_matched", v1_valid), ("v2", v2_valid)):
            rows.append({"run_id": f"P{i}:{version}", "pair_id": f"P{i}", "seed": 20_261_120 + i,
                         "evaluator_version": version, "valid": valid,
                         "score": 0.5 if valid else 0.0})
    return {"assigned_runs": 12, "runs": rows}


def test_summary_reports_descriptive_rates_and_paired_table_without_p_value():
    from experiments.openevolve.analyze_b2_estimation import summarize

    result = summarize(report(), expected_pairs=6)
    assert result["v1_matched"]["valid"] == 4
    assert result["v1_matched"]["invalid_or_missing"] == 2
    assert result["v2"]["valid"] == 4
    assert result["paired_table"] == {
        "both_valid": 3, "v1_valid_v2_invalid": 1,
        "v1_invalid_v2_valid": 1, "both_invalid": 1,
    }
    assert result["assigned_pairs"] == 6
    assert "p_value" not in result


def test_summary_fails_closed_on_duplicate_or_missing_arm_rows():
    from experiments.openevolve.analyze_b2_estimation import summarize

    bad = report()
    bad["runs"].append(bad["runs"][0])
    with pytest.raises(ValueError):
        summarize(bad, expected_pairs=6)


def test_summary_does_not_call_a_partial_run_pair_complete():
    from experiments.openevolve.analyze_b2_estimation import summarize

    partial = report()
    partial["runs"][-1]["reason"] = "run_not_complete"
    result = summarize(partial, expected_pairs=6)

    assert result["complete_pairs"] == 5

    bad = report()
    bad["runs"].pop()
    with pytest.raises(ValueError):
        summarize(bad, expected_pairs=6)
