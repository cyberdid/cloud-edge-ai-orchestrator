"""Aggregate pilot runs: per-run re-check summaries, per task x arm statistics and figures
(``python -m experiments.openevolve.analyze benchmarks/openevolve_pilot``)."""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from experiments.openevolve.recheck_all import Recheck, recheck_run  # noqa: E402

ARMS = ("L", "LC", "B")
ARM_LABELS = {"L": "L: Gemma only", "LC": "L+C: Gemma + 10% Argon", "B": "B: best-of-N"}
# Categorical slots 1-3 of the dataviz reference palette (validated, light mode);
# slot 3 is below 3:1 contrast, so every line is also direct-labelled.
ARM_COLORS = {"L": "#2a78d6", "LC": "#eb6834", "B": "#1baf7a"}
INK, MUTED, GRID = "#2b2b29", "#6b6a63", "#e4e3de"
FIGURES_DIR = Path(__file__).resolve().parents[2] / "docs/openevolve_pilot"


def _stats(values: list[float]) -> dict:
    return {"median": statistics.median(values), "min": min(values), "max": max(values)} if values else {}


def _style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _curves_figure(task: str, runs: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    _style(ax)
    seeds = [r["seed_score"] for r in runs]
    if seeds:
        ax.axhline(statistics.median(seeds), color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    for arm in ARMS:
        curves = [r["curve"] for r in runs if r["arm"] == arm and r["curve"]]
        if not curves:
            continue
        length = min(len(c) for c in curves)
        xs = [curves[0][i][0] + 1 for i in range(length)]
        ys = [statistics.median(c[i][2] for c in curves) for i in range(length)]
        ax.plot(xs, ys, color=ARM_COLORS[arm], linewidth=2, label=ARM_LABELS[arm])
        ax.annotate(arm if arm != "LC" else "L+C", (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK)
    ax.set_xlabel("LLM calls", color=INK, fontsize=10)
    ax.set_ylabel("verified score of shipped program\n(median over seeds)", color=INK, fontsize=10)
    ax.set_title(f"{task}: verified progress by budget (dashed: seed)", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _false_figure(summary: dict, path: Path) -> None:
    tasks = sorted(summary)
    fig, ax = plt.subplots(figsize=(7, 3.6), dpi=150)
    _style(ax)
    width = 0.26
    for offset, arm in enumerate(ARMS):
        xs, ys = [], []
        for t, task in enumerate(tasks):
            group = summary[task].get(arm)
            if group:
                xs.append(t + (offset - 1) * (width + 0.02))
                ys.append(group["pooled_false_improvement_rate"])
        bars = ax.bar(xs, ys, width=width, color=ARM_COLORS[arm], label=ARM_LABELS[arm])
        ax.bar_label(bars, labels=[f"{y:.0%}" for y in ys], fontsize=8, color=INK, padding=2)
    ax.set_xticks(range(len(tasks)), tasks, color=INK)
    ax.set_ylabel("false improvements\n(share of above-seed candidates)", color=INK, fontsize=10)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.02))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(root: Path, figures_dir: Path = FIGURES_DIR, recheck: Recheck | None = None) -> dict:
    root = Path(root)
    runs = []
    for manifest in sorted(root.glob("*/manifest.json")):
        result = recheck_run(manifest.parent, recheck=recheck)
        (manifest.parent / "summary.json").write_text(json.dumps(result, indent=2))
        runs.append(result)
    summary: dict = {}
    for task in sorted({r["task"] for r in runs}):
        for arm in ARMS:
            group = [r for r in runs if r["task"] == task and r["arm"] == arm]
            if not group:
                continue
            above = sum(r["above_seed_count"] for r in group)
            summary.setdefault(task, {})[arm] = {
                "runs": len(group),
                "seeds": [r["seed"] for r in group],
                "seed_score": _stats([r["seed_score"] for r in group]),
                "shipped_verified_score": _stats([r["shipped_verified_score"] for r in group]),
                "shipped_working_score": _stats([r["shipped_working_score"] for r in group]),
                "best_verified_score": _stats([r["best_verified_score"] for r in group]),
                "false_improvement_rate": _stats([r["false_improvement_rate"] for r in group]),
                "pooled_false_improvement_rate": sum(r["false_improvements"] for r in group) / above if above else 0.0,
                "shipped_false_runs": sum(r["shipped_is_false"] for r in group),
                "argon_calls": sum(r["argon_calls"] for r in group),
                "argon_tokens": sum(r["argon_tokens"] for r in group),
                "argon_wall_s": sum(r["argon_wall_s"] for r in group),
                "gemma_wall_s": sum(r["gemma_wall_s"] for r in group),
            }
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    for task in summary:
        _curves_figure(task, [r for r in runs if r["task"] == task], figures_dir / f"fig_curves_{task}.png")
    _false_figure(summary, figures_dir / "fig_false_improvements.png")
    return summary


if __name__ == "__main__":
    main(Path(sys.argv[1]))
