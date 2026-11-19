import json
from pathlib import Path

from scripts.analysis.oe_exploratory_tests import plot_pilot_false_improvements, serial_scores, unpaired


def _row(kind, code_hash, score, rep):
    return {
        "kind": kind,
        "code_sha256": code_hash,
        "runs": ["seed" if kind == "seed" else "example"],
        "rep": rep,
        "wall_s": 1.0,
        "recheck": {"valid": True, "score": score, "reasons": []},
    }


def test_seed_score_uses_all_repeats_for_same_code_hash(tmp_path: Path):
    seed_hash = "seed-hash"
    rows = [
        *[_row("seed", seed_hash, score, rep) for rep, score in enumerate([0.19, 0.20, 0.21, 0.24, 0.25])],
        *[_row("shipped", seed_hash, score, rep) for rep, score in enumerate([0.22, 0.23, 0.24])],
    ]
    path = tmp_path / "knn_serial.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    scores = serial_scores(tmp_path)

    assert scores.seed_hash == seed_hash
    assert scores.seed_calibration_score == 0.21
    assert scores.seed_score == 0.225
    assert scores.value(seed_hash) == (True, 0.225)


def test_serial_scores_reads_published_artifact_filenames(tmp_path: Path):
    seed_hash = "seed-hash"
    rows = [_row("seed", seed_hash, score, rep) for rep, score in enumerate([0.19, 0.20, 0.21, 0.24, 0.25])]
    (tmp_path / "knn_serial_retiming_2026-11-19.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    (tmp_path / "knn_serial_above_seed_2026-11-19.jsonl").write_text("")

    scores = serial_scores(tmp_path)

    assert scores.seed_hash == seed_hash
    assert scores.seed_score == 0.21


def test_unpaired_all_ties_returns_valid_p_value_one():
    result = unpaired([0.223] * 10, [0.223] * 10)

    assert result["mann_whitney_p"] == 1.0


def test_serial_false_improvement_figure_is_written(tmp_path: Path):
    pooled = {
        "pilot-12b/circle_packing/L": {"false": 21, "above_seed": 140, "rate": 0.15},
        "pilot-12b/circle_packing/B": {"false": 22, "above_seed": 86, "rate": 22 / 86},
        "pilot-12b/knn/L": {"false": 33, "above_seed": 42, "rate": 33 / 42},
        "pilot-12b/knn/B": {"false": 5, "above_seed": 5, "rate": 1.0},
    }
    output = tmp_path / "serial.png"

    plot_pilot_false_improvements(pooled, output)

    assert output.is_file()
    assert output.stat().st_size > 1_000
