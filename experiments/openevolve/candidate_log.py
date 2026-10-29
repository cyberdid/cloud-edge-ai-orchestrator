"""Append every evaluated candidate to the run's JSONL log (path from the environment)."""
from __future__ import annotations

import hashlib
import json
import os
import time

ENV_VAR = "OE_PILOT_CANDIDATE_LOG"


def code_sha256(code: str) -> str:
    return hashlib.sha256(code.encode()).hexdigest()


def append_candidate(record: dict) -> None:
    path = os.environ.get(ENV_VAR)
    if not path:
        return
    line = dict(record, ts=time.time(), code_sha256=code_sha256(record.get("code", "")))
    with open(path, "a") as handle:
        handle.write(json.dumps(line, default=str) + "\n")
