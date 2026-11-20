"""Independently re-check only the official selected program for each B2 run."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable

from experiments.openevolve.colab.b2_protocol import load_protocol

CHECKER_COMPONENTS = (
    "antigravity_hybrid_orchestrator/integrity/oracle.py",
    "antigravity_hybrid_orchestrator/runtime/sandbox.py",
    "experiments/openevolve/candidate_log.py",
    "experiments/openevolve/tasks/knn/recheck.py",
    "experiments/openevolve/tasks/knn/data.py",
    "experiments/openevolve/tasks/knn/driver.py",
    "experiments/openevolve/sandbox_exec.py",
    "experiments/openevolve/static_check.py",
    "experiments/openevolve/tasks/knn/working_evaluator.py",
)


def checker_fingerprint(repo_root: Path) -> dict[str, Any]:
    components = {}
    for relative in CHECKER_COMPONENTS:
        path = repo_root / relative
        components[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    joined = "".join(f"{path}\0{digest}\n" for path, digest in sorted(components.items()))
    return {"components": components, "sha256": hashlib.sha256(joined.encode()).hexdigest()}


def recheck_selected(root: str | Path, protocol: dict[str, Any], *, protocol_sha256: str,
                     source_commit: str, checker: Callable[[str], dict[str, Any]] | None = None,
                     repo_root: str | Path | None = None) -> dict[str, Any]:
    """Re-check each official selected artifact; missing/invalid selections count as failures."""
    root = Path(root).resolve()
    if checker is None:
        from experiments.openevolve.tasks.knn.recheck import recheck_knn
        checker = recheck_knn
    repo_root = Path(repo_root or Path(__file__).resolve().parents[2]).resolve()
    fingerprint = checker_fingerprint(repo_root)
    rows = []
    for pair in protocol["pairs"]:
        for version in ("v1_matched", "v2"):
            run_id = f"{pair['pair_id']}:{version}"
            run_dir = root / "runs" / pair["pair_id"] / version
            manifest_path = run_dir / "manifest.json"
            if not manifest_path.is_file():
                rows.append({"run_id": run_id, "pair_id": pair["pair_id"], "seed": pair["seed"],
                             "evaluator_version": version, "valid": False, "score": 0.0,
                             "reason": "missing_manifest", "selection_sha256": None})
                continue
            try:
                manifest = json.loads(manifest_path.read_text())
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid manifest JSON for {run_id}") from exc
            expected = {"task": "knn", "arm": "L", "seed": pair["seed"],
                        "git_sha": source_commit, "protocol_id": protocol["protocol_id"],
                        "protocol_sha256": protocol_sha256, "model_digest": protocol["model_digest"],
                        "evaluator_version": version}
            for key, value in expected.items():
                if manifest.get(key) != value:
                    raise ValueError(f"{run_id} manifest mismatch for {key}")
            selection = manifest.get("selection", {})
            selection_sha = selection.get("code_sha256")
            base = {"run_id": run_id, "pair_id": pair["pair_id"], "seed": pair["seed"],
                    "evaluator_version": version, "selection_sha256": selection_sha}
            if manifest.get("status") != "complete":
                rows.append({**base, "valid": False, "score": 0.0,
                             "reason": "run_not_complete"})
                continue
            if selection.get("status") != "recorded" or not isinstance(selection_sha, str):
                rows.append({**base, "valid": False, "score": 0.0,
                             "reason": "missing_official_selection"})
                continue
            relative = Path(selection.get("path", ""))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"{run_id} selected-program path escapes its run directory")
            program_path = (run_dir / relative).resolve()
            if not program_path.is_relative_to(run_dir.resolve()):
                raise ValueError(f"{run_id} selected-program path escapes its run directory")
            if not program_path.is_file():
                rows.append({**base, "valid": False, "score": 0.0,
                             "reason": "missing_selected_program"})
                continue
            code = program_path.read_text()
            actual_sha = hashlib.sha256(code.encode()).hexdigest()
            if actual_sha != selection_sha:
                rows.append({**base, "valid": False, "score": 0.0,
                             "reason": "selected_program_hash_mismatch"})
                continue
            try:
                result = checker(code)
            except Exception as exc:
                result = {"valid": False, "score": 0.0,
                          "reasons": [f"checker_exception:{type(exc).__name__}"]}
            valid = result.get("valid") is True
            rows.append({**base, "valid": valid,
                         "score": float(result.get("score", 0.0)) if valid else 0.0,
                         "holdout_recall": result.get("holdout_recall"),
                         "holdout_speed_ratio": result.get("holdout_speed_ratio"),
                         "reason": None if valid else result.get("reasons", ["checker_invalid"]),
                         "checker_result": result})
    invalid = sum(row["valid"] is not True for row in rows)
    return {
        "protocol_id": protocol["protocol_id"], "protocol_sha256": protocol_sha256,
        "source_commit": source_commit, "task": "knn", "assigned_runs": len(rows),
        "checked_runs": sum("checker_result" in row for row in rows),
        "invalid_selected_runs": invalid, "checker_fingerprint": fingerprint,
        "holdout_definition": "experiments/openevolve/tasks/knn/recheck.py::HOLDOUT_SETS",
        "holdout_sets": [9001, 9002, 9003], "created_unix": time.time(), "runs": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--protocol", type=Path, default=Path(
        "experiments/openevolve/configs/b2_estimation_2026-11-20.json"))
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, default=Path("b2-selected-rechecks.json"))
    args = parser.parse_args()
    protocol = load_protocol(args.protocol)
    protocol_sha256 = hashlib.sha256(args.protocol.read_bytes()).hexdigest()
    report = recheck_selected(args.root, protocol, protocol_sha256=protocol_sha256,
                              source_commit=args.source_commit)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in
                      ("protocol_id", "assigned_runs", "checked_runs", "invalid_selected_runs")},
                     sort_keys=True))


if __name__ == "__main__":
    main()
