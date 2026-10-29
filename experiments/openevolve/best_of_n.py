"""Arm B: best-of-N without evolution. Every attempt mutates the seed program with the
same diff-evolution prompt OpenEvolve would build, but with no history, top programs
or inspirations; the best attempt by working score is what B ships."""
from __future__ import annotations

import importlib.util
import json
import random
from pathlib import Path

from openevolve.config import load_config
from openevolve.prompt.sampler import PromptSampler
from openevolve.utils.code_utils import apply_diff_blocks, enforce_evolve_blocks, extract_diffs

from experiments.openevolve.router import Router
from experiments.openevolve.tasks import TASKS


def _load_evaluator(path: Path):
    spec = importlib.util.spec_from_file_location(f"oe_pilot_eval_{path.parent.name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def child_program(seed_code: str, reply: str, diff_pattern: str) -> str | None:
    """Apply a SEARCH/REPLACE reply to the seed as OpenEvolve's worker does; None = failed."""
    try:
        blocks = extract_diffs(reply, diff_pattern)
        if not blocks:
            return None
        child, applied = apply_diff_blocks(seed_code, blocks)
        if applied == 0 or child == seed_code:
            return None
        child = enforce_evolve_blocks(seed_code, child)
    except ValueError:
        return None
    return None if child == seed_code else child


def run_best_of_n(task: str, router: Router, n: int, out_dir: Path, seed: int, resume: bool) -> dict:
    spec = TASKS[task]
    config = load_config(spec["config"])
    sampler = PromptSampler(config.prompt)
    evaluator = _load_evaluator(spec["evaluator"])
    seed_code = spec["initial"].read_text()
    work = Path(out_dir) / "best_of_n"
    work.mkdir(parents=True, exist_ok=True)
    state_path = work / "state.json"
    state = json.loads(state_path.read_text()) if resume and state_path.exists() else {}
    if "seed_metrics" not in state:
        (work / "seed.py").write_text(seed_code)
        state = {"seed_metrics": evaluator.evaluate(str(work / "seed.py")), "evaluated": 0, "failed_attempts": 0}
        state_path.write_text(json.dumps(state))
    for attempt in range(router.call_index, n):
        random.seed(f"{seed}:{attempt}")  # template stochasticity in the prompt sampler
        prompt = sampler.build_prompt(
            current_program=seed_code, parent_program=seed_code, program_metrics=state["seed_metrics"],
            previous_programs=[], top_programs=[], inspirations=[], language="python",
            evolution_round=attempt, diff_based_evolution=True,
        )
        reply = router.complete([{"role": "system", "content": prompt["system"]},
                                 {"role": "user", "content": prompt["user"]}])
        child = child_program(seed_code, reply, config.diff_pattern)
        if child is None:
            state["failed_attempts"] += 1
        else:
            path = work / f"attempt_{attempt:03d}.py"
            path.write_text(child)
            evaluator.evaluate(str(path))
            state["evaluated"] += 1
        state_path.write_text(json.dumps(state))
    return {"attempts": n, "evaluated": state["evaluated"], "failed_attempts": state["failed_attempts"]}
