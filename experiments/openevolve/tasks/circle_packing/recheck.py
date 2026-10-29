"""Independent re-check for circle packing (never visible to the mutating model).

Recomputes validity and the sum of radii from the returned geometry with a strict
tolerance, ignores the reported sum, requires two identical sandbox runs and a clean
static check. Hard-coded coordinates are a valid construction, not a hack.
"""
from __future__ import annotations

import math

from experiments.openevolve.sandbox_exec import run_candidate
from experiments.openevolve.static_check import forbidden_primitives
from experiments.openevolve.tasks.circle_packing.driver import DRIVER

N_CIRCLES = 26
TOL = 1e-12
SANDBOX_TIMEOUT_S = 60


def _geometry_failures(centers, radii) -> list[str]:
    if len(centers) != N_CIRCLES or len(radii) != N_CIRCLES:
        return [f"expected {N_CIRCLES} circles, got {len(centers)} centres and {len(radii)} radii"]
    if any(len(c) != 2 for c in centers):
        return ["centres must be (x, y) pairs"]
    values = [v for c in centers for v in c] + list(radii)
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
        return ["non-finite coordinate or radius"]
    failures = [f"circle {i}: radius {r} <= 0" for i, r in enumerate(radii) if r <= 0]
    for i, ((x, y), r) in enumerate(zip(centers, radii)):
        if x - r < -TOL or y - r < -TOL or x + r > 1 + TOL or y + r > 1 + TOL:
            failures.append(f"circle {i}: outside the unit square")
    for i in range(N_CIRCLES):
        for j in range(i + 1, N_CIRCLES):
            gap = math.dist(centers[i], centers[j]) - (radii[i] + radii[j])
            if gap < -TOL:
                failures.append(f"circles {i},{j}: overlap {-gap:.3e}")
    return failures


def recheck_packing(program_code: str) -> dict:
    reasons = list(forbidden_primitives(program_code))
    first = run_candidate(program_code, DRIVER, {}, timeout_s=SANDBOX_TIMEOUT_S)
    if not first.ok:
        return {"valid": False, "score": 0.0, "reasons": [f"run failed: {first.error[-300:]}"] + reasons}
    centers, radii = first.output["centers"], first.output["radii"]
    reasons = _geometry_failures(centers, radii) + reasons
    second = run_candidate(program_code, DRIVER, {}, timeout_s=SANDBOX_TIMEOUT_S)
    if not second.ok or second.output != first.output:
        reasons.append("non-deterministic output across two runs")
    valid = not reasons
    return {"valid": valid, "score": float(sum(radii)) if valid else 0.0, "reasons": reasons}
