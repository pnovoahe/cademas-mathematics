"""Publication plotting style (Okabe–Ito; CADEMAS operator series)."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

OKABE_ITO = [
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
    "#000000",
]

# Operator → Okabe–Ito mapping (colorblind-safe, distinct on print).
OPERATOR_COLORS = {
    "linear": OKABE_ITO[4],  # #0072B2
    "geometric": OKABE_ITO[5],  # #D55E00
    "min": OKABE_ITO[2],  # #009E73
    "max": OKABE_ITO[0],  # #E69F00
}

PANEL_LETTER_FP = FontProperties(family="Helvetica", weight="bold", size=12)


def apply_paper_style(font_size: int = 11) -> None:
    """Publication defaults: Helvetica/Arial, white faces, no top/right spines."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": font_size,
            "axes.labelsize": font_size,
            "axes.titlesize": font_size + 1,
            "legend.fontsize": font_size - 1,
            "xtick.labelsize": font_size - 1,
            "ytick.labelsize": font_size - 1,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "pdf.fonttype": 42,
            "savefig.format": "pdf",
            "savefig.bbox": "tight",
            "savefig.dpi": 300,
            "mathtext.fontset": "custom",
            "mathtext.rm": "Helvetica",
            "mathtext.it": "Helvetica Oblique",
            "mathtext.bf": "Helvetica Bold",
            "mathtext.sf": "Helvetica",
            "mathtext.tt": "Helvetica",
        }
    )


def apply_helvetica_style(font_scale: float = 1.1, **overrides) -> None:
    """Compatibility wrapper used by existing plotting modules."""
    base = 11
    apply_paper_style(font_size=max(9, int(round(base * font_scale / 1.1))))
    if overrides:
        plt.rcParams.update(overrides)


def add_panel_letter(ax, letter: str, *, x: float = 0.02, y: float = 1.02) -> None:
    """Panel tag, e.g. (a), above the axes (Helvetica bold)."""
    ax.text(
        x,
        y,
        f"({letter})",
        transform=ax.transAxes,
        fontproperties=PANEL_LETTER_FP,
        va="bottom",
        ha="left",
        clip_on=False,
    )


def add_panel_header(ax, letter: str, label: str) -> None:
    """Bold panel letter above axes; centered title."""
    add_panel_letter(ax, letter)
    ax.set_title(label, fontsize=9.5, pad=4)


def style_axes_frame(ax) -> None:
    """White panel, visible left/bottom spines, hidden top/right."""
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine_name in ("left", "bottom"):
        spine = ax.spines[spine_name]
        spine.set_visible(True)
        spine.set_edgecolor("black")
        spine.set_linewidth(1.0)


OPERATOR_STYLE = {
    "linear": {
        "color": OPERATOR_COLORS["linear"],
        "marker": "v",
        "markersize": 13.0,
        "linestyle": "-",
        "linewidth": 2.5,
        "zorder": 2,
    },
    "geometric": {
        "color": OPERATOR_COLORS["geometric"],
        "marker": "s",
        "markersize": 9.0,
        "linestyle": "-",
        "linewidth": 2.0,
        "zorder": 3,
    },
    "min": {
        "color": OPERATOR_COLORS["min"],
        "marker": "o",
        "markersize": 6.5,
        "linestyle": "-",
        "linewidth": 2.0,
        "zorder": 4,
    },
    "max": {
        "color": OPERATOR_COLORS["max"],
        "marker": "^",
        "markersize": 6,
        "linestyle": "-",
        "linewidth": 1.8,
        "zorder": 3,
    },
}


def plot_kwargs(operator: str, **overrides) -> dict:
    """Return matplotlib plot kwargs for a given operator."""
    return {**OPERATOR_STYLE[operator], **overrides}


def plot_operator(ax, x, y, operator: str, label: str | None = None, **overrides):
    """Plot a series with the canonical operator styling."""
    style = plot_kwargs(operator, **overrides)
    marker = style.pop("marker")
    linestyle = style.pop("linestyle")
    return ax.plot(x, y, f"{marker}{linestyle}", label=label, **style)


def _series_color(operator: str) -> str:
    """Line and CI band color for an operator."""
    return OPERATOR_STYLE[operator]["color"]


def plot_operator_mean_ci(
    ax,
    x,
    y_mean,
    y_lo,
    y_hi,
    operator: str,
    label: str | None = None,
    ci_alpha: float = 0.22,
    **overrides,
):
    """Plot mean line with shaded 95% CI band."""
    color = _series_color(operator)
    ax.fill_between(x, y_lo, y_hi, color=color, alpha=ci_alpha, linewidth=0)
    return plot_operator(ax, x, y_mean, operator, label=label, **overrides)
