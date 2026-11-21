"""Shared publication styling for Paper II figures."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLORS = {
    "L": "#2369A1",       # blue
    "B": "#C4511C",       # vermilion
    "LC": "#00856A",      # green
    "ink": "#253746",
    "muted": "#586975",
    "pair": "#AAB5BC",
    "grid": "#E4E9EC",
    "panel": "#F5F7F8",
}


def apply_style() -> None:
    """Use legible, consistent typography at the manuscript's print size."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9.5,
        "axes.titleweight": "bold",
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "axes.labelcolor": COLORS["ink"],
        "text.color": COLORS["ink"],
        "xtick.color": COLORS["muted"],
        "ytick.color": COLORS["muted"],
        "axes.edgecolor": "#AEB9C0",
        "axes.linewidth": 0.65,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })


def polish_axis(ax) -> None:
    """Apply restrained axes and grid treatment to a quantitative panel."""
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.65)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#AEB9C0")
    ax.spines["bottom"].set_color("#AEB9C0")
    ax.tick_params(length=3, width=0.65, color="#87959E", pad=3)
