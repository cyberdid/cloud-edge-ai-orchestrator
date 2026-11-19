"""Regenerate Paper II's publication figures from the committed run-level analysis."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "DejaVu Sans"


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "docs/paper_ii"
DATA_PATH = ROOT / "docs/colab_scale/results/exploratory_tests_2026-11-19.json"
COLORS = {"L": "#2B6E9F", "B": "#D17A28", "LC": "#168A78"}
MODEL_ORDER = ["gemma4-e4b", "gemma4-12b", "gemma4-26b", "gemma4-31b"]
MODEL_LABELS = {"gemma4-e4b": "E4B", "gemma4-12b": "12B", "gemma4-26b": "26B\nMoE", "gemma4-31b": "31B"}


def load_results() -> dict:
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    if not data.get("note", "").startswith("EXPLORATORY"):
        raise ValueError("Expected the dated exploratory analysis artifact")
    return data


def figure_pipeline(lang: str) -> None:
    labels = {
        "en": ["Task + seed", "Search\nL / LC / B", "Working fitness\nscore + selection", "Independent\nre-check", "Verified\nendpoint"],
        "uk": ["Задача +\nпочаткова програма", "Пошук\nL / LC / B", "Робочий fitness\nі відбір", "Незалежна\nперевірка", "Перевірений\nрезультат"],
    }
    notes = {
        "en": "The holdout is never shown to the generator or working evaluator",
        "uk": "Відкладена перевірка недоступна генератору й робочому оцінювачу",
    }
    fig, ax = plt.subplots(figsize=(10.0, 2.55))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2.55)
    ax.axis("off")
    xs = [0.25, 2.18, 4.12, 6.06, 8.0]
    width, height, y = 1.62, 0.83, 1.02
    fills = ["#E7F0F7", "#E8F3EF", "#F8F0E5", "#E5F2F0", "#EAF1E7"]
    edges = ["#7FA6C0", "#79A998", "#D2A875", "#5D9D91", "#8AA778"]
    for i, x in enumerate(xs):
        box = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.06,rounding_size=0.12",
                             linewidth=1.2, edgecolor=edges[i], facecolor=fills[i])
        ax.add_patch(box)
        ax.text(x + width / 2, y + height / 2, labels[lang][i], ha="center", va="center",
                fontsize=9.5, weight="bold", color="#253644")
        if i < len(xs) - 1:
            ax.add_patch(FancyArrowPatch((x + width + 0.03, y + height / 2),
                                         (xs[i + 1] - 0.06, y + height / 2),
                                         arrowstyle="-|>", mutation_scale=12, linewidth=1.15,
                                         color="#75828A"))
    ax.text(5.0, 0.46, notes[lang], ha="center", va="center", fontsize=8.5, color="#52616A")
    ax.text(2.99, 2.15, "LC: ~10% Argon calls" if lang == "en" else "LC: близько 10% викликів Argon",
            ha="center", va="center", fontsize=8, color="#126F62")
    fig.tight_layout(pad=0.25)
    fig.savefig(PAPER / "figures" / f"fig1_pipeline_{lang}.pdf", bbox_inches="tight")
    fig.savefig(PAPER / "figures" / f"fig1_pipeline_{lang}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def figure_circle_scale(data: dict, lang: str) -> None:
    rows = [r for r in data["runs"] if r["task"] == "circle_packing" and r["model"] in MODEL_ORDER
            and r["seed"] <= 1010 and r["arm"] in ("L", "B")]
    if not rows:
        raise ValueError("No scale-ladder circle-packing runs found")
    fig, ax = plt.subplots(figsize=(8.0, 4.35))
    rng = np.random.default_rng(20261119)
    center = np.arange(len(MODEL_ORDER), dtype=float)
    offsets = {"L": -0.15, "B": 0.15}
    markers = {"L": "o", "B": "s"}
    for arm in ("L", "B"):
        for i, model in enumerate(MODEL_ORDER):
            values = np.asarray([r["shipped"] for r in rows if r["model"] == model and r["arm"] == arm], float)
            if values.size == 0:
                continue
            x = center[i] + offsets[arm] + rng.normal(0, 0.025, values.size)
            ax.scatter(x, values, s=30, marker=markers[arm], color=COLORS[arm], alpha=0.78,
                       edgecolor="white", linewidth=0.45, zorder=3,
                       label=(("Evolution (L)" if arm == "L" else "Best-of-N (B)") if lang == "en"
                              else ("Еволюція (L)" if arm == "L" else "Best-of-N (B)")) if i == 0 else None)
            median = float(np.median(values))
            ax.plot([center[i] + offsets[arm] - 0.09, center[i] + offsets[arm] + 0.09],
                    [median, median], color="#263746", linewidth=2.3, zorder=4)
    ax.set_xticks(center, [MODEL_LABELS[m] for m in MODEL_ORDER])
    ax.set_ylim(-0.08, 2.78)
    ax.set_yticks(np.arange(0, 2.76, 0.5))
    ax.set_ylabel("Independent score: sum of radii" if lang == "en" else "Незалежна оцінка: сума радіусів")
    ax.set_xlabel("Gemma 4 model family" if lang == "en" else "Сімейство моделей Gemma 4")
    ax.grid(axis="y", color="#E2E6E8", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    ax.text(0.99, 0.025, "n=10 per arm; 31B n=5" if lang == "en" else "n=10 на режим; 31B n=5",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="#5E6B72")
    fig.tight_layout()
    fig.savefig(PAPER / "figures" / f"fig2_circle_scale_{lang}.pdf", bbox_inches="tight")
    fig.savefig(PAPER / "figures" / f"fig2_circle_scale_{lang}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def figure_lc_pairs(data: dict, lang: str) -> None:
    rows = [r for r in data["runs"] if r["model"] == "gemma4-26b" and r["arm"] in ("L", "LC")
            and r["seed"] <= 909 and r["task"] in ("circle_packing", "knn")]
    task_names = {"en": {"circle_packing": "Circle packing", "knn": "Approximate kNN"},
                  "uk": {"circle_packing": "Пакування кіл", "knn": "Наближений kNN"}}
    ylabels = {"en": {"circle_packing": "Verified sum of radii", "knn": "Serially re-timed score"},
               "uk": {"circle_packing": "Перевірена сума радіусів", "knn": "Оцінка після серійного перерахунку"}}
    stats_text = {"en": {"circle_packing": "7 / 9 higher; Wilcoxon p = 0.148",
                         "knn": "8 / 9 higher; Wilcoxon p = 0.0078"},
                  "uk": {"circle_packing": "Вища у 7 / 9 пар; Wilcoxon p = 0,148",
                         "knn": "Вища у 8 / 9 пар; Wilcoxon p = 0,0078"}}
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.15))
    for ax, task in zip(axes, ("circle_packing", "knn")):
        values = defaultdict(dict)
        for r in rows:
            if r["task"] == task:
                values[r["seed"]][r["arm"]] = r["shipped"]
        for seed in sorted(values):
            if set(values[seed]) != {"L", "LC"}:
                continue
            lv, lcv = values[seed]["L"], values[seed]["LC"]
            ax.plot([0, 1], [lv, lcv], color="#AAB4BA", alpha=0.72, linewidth=1.0, zorder=1)
            ax.scatter([0], [lv], s=34, color=COLORS["L"], edgecolor="white", linewidth=0.45, zorder=3)
            ax.scatter([1], [lcv], s=38, color=COLORS["LC"], edgecolor="white", linewidth=0.45, zorder=3)
        ax.set_title(task_names[lang][task], loc="left", fontsize=11, weight="bold")
        ax.set_xticks([0, 1], ["L", "LC"])
        ax.set_xlim(-0.35, 1.35)
        if task == "circle_packing":
            ax.set_ylim(-0.1, 2.82)
            ax.set_yticks(np.arange(0, 2.76, 0.5))
        else:
            ax.set_ylim(0, 1.1)
            ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.set_ylabel(ylabels[lang][task])
        ax.grid(axis="y", color="#E2E6E8", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(0.5, 0.035, stats_text[lang][task], transform=ax.transAxes,
                ha="center", va="bottom", fontsize=7.8, color="#52616A")
    legend = [plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["L"],
                         markeredgecolor="white", markersize=7, label="L: Gemma-only" if lang == "en" else "L: лише Gemma"),
              plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["LC"],
                         markeredgecolor="white", markersize=7, label="LC: ~10% Argon" if lang == "en" else "LC: близько 10% Argon")]
    fig.legend(handles=legend, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(PAPER / "figures" / f"fig3_lc_pairs_{lang}.pdf", bbox_inches="tight")
    fig.savefig(PAPER / "figures" / f"fig3_lc_pairs_{lang}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    data = load_results()
    for lang in ("en", "uk"):
        figure_pipeline(lang)
        figure_circle_scale(data, lang)
        figure_lc_pairs(data, lang)


if __name__ == "__main__":
    main()
