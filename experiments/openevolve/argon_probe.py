"""Measure what one Argon mutation costs before the campaign
(``python -m experiments.openevolve.argon_probe --calls 3``).

Sends the circle-packing seed prompt that arms B/L+C use through the same ``agy``
backend and prints latency, tokens, reply length and whether the reply applies as a
diff. The owner reads the agy weekly quota before and after; probe data is never
analysed as campaign data."""
from __future__ import annotations

import argparse
import json
import time

from openevolve.config import load_config
from openevolve.prompt.sampler import PromptSampler

from experiments.openevolve.best_of_n import child_program
from experiments.openevolve.router import agy_argon
from experiments.openevolve.tasks import TASKS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calls", type=int, default=3)
    parser.add_argument("--task", choices=sorted(TASKS), default="circle_packing")
    args = parser.parse_args()
    spec = TASKS[args.task]
    config = load_config(spec["config"])
    seed_code = spec["initial"].read_text()
    prompt = PromptSampler(config.prompt).build_prompt(
        current_program=seed_code, parent_program=seed_code, program_metrics={"combined_score": 0.0},
        language="python", diff_based_evolution=True)
    messages = [{"role": "system", "content": prompt["system"]}, {"role": "user", "content": prompt["user"]}]
    argon = agy_argon()
    for index in range(args.calls):
        started = time.time()
        try:
            text, tokens_in, tokens_out = argon(messages)[:3]
            error = ""
        except Exception as exc:
            text, tokens_in, tokens_out, error = "", 0, 0, str(exc)[:300]
        print(json.dumps({
            "call": index, "latency_s": round(time.time() - started, 1), "prompt_tokens": tokens_in,
            "completion_tokens": tokens_out, "reply_chars": len(text),
            "applies_as_diff": child_program(seed_code, text, config.diff_pattern) is not None, "error": error,
        }), flush=True)


if __name__ == "__main__":
    main()
