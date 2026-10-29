"""Pilot tasks: seed program, working evaluator (OpenEvolve entry point) and config."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TASKS = {
    "circle_packing": {
        "initial": ROOT / "upstream/circle_packing/initial_program.py",
        "evaluator": ROOT / "tasks/circle_packing/working_evaluator.py",
        "config": ROOT / "configs/circle_packing.yaml",
    },
    "knn": {
        "initial": ROOT / "tasks/knn/initial_program.py",
        "evaluator": ROOT / "tasks/knn/working_evaluator.py",
        "config": ROOT / "configs/knn.yaml",
    },
}
