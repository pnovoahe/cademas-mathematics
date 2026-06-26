"""Shared operator colors, markers, and line styles for all CADEMAS figures."""

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import seaborn as sns

HELVETICA = "Helvetica"

# Panel background: lighter than seaborn paper (#EAEAF2), with white grid stripes on top.
PANEL_FACECOLOR = "#F3F3F6"
FIGURE_FACECOLOR = "white"

# Magma sequential samples, spaced and away from palette extremes (t ≈ 0.1 / 0.9)
MAGMA_OPERATOR_STOPS = {
    "linear": 0.28,
    "min": 0.52,
    "geometric": 0.76,
    "max": 0.40,
}


def _magma_hex(t: float) -> str:
    rgba = plt.get_cmap("magma")(t)
    return mcolors.to_hex(rgba[:3])


OPERATOR_COLORS = {op: _magma_hex(t) for op, t in MAGMA_OPERATOR_STOPS.items()}


def apply_helvetica_style(font_scale: float = 1.1, **overrides) -> None:
    """Configure matplotlib/seaborn to use Helvetica for all text, including mathtext."""
    sns.set_theme(context="paper", font_scale=font_scale)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [HELVETICA, "Arial", "DejaVu Sans", "sans-serif"],
            "mathtext.fontset": "custom",
            "mathtext.rm": HELVETICA,
            "mathtext.it": "Helvetica Oblique",
            "mathtext.bf": "Helvetica Bold",
            "mathtext.sf": HELVETICA,
            "mathtext.tt": HELVETICA,
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "savefig.format": "pdf",
            "savefig.bbox": "tight",
            "savefig.dpi": 300,
            "axes.edgecolor": "black",
            "axes.linewidth": 1.0,
            "axes.facecolor": PANEL_FACECOLOR,
            "figure.facecolor": FIGURE_FACECOLOR,
            "savefig.facecolor": FIGURE_FACECOLOR,
            "axes.grid": True,
            "grid.color": "white",
            **overrides,
        }
    )


def style_axes_frame(ax) -> None:
    """Ensure a visible black border on all four spines and a light panel background."""
    ax.set_facecolor(PANEL_FACECOLOR)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("black")
        spine.set_linewidth(1.0)


OPERATOR_STYLE = {
    "linear": {
        "color": "#3D82C5",
        "marker": "v",
        "markersize": 8,
        #"markerfacecolor": "white",
        #"markeredgecolor": "#C31F5C",
        #"markeredgewidth": 2,
        "linestyle": "-",
        "linewidth": 3,
    },
    "min": {
        "color": "#FBB216",
        "marker": "s",
        "markersize": 6,
        "linestyle": "-",
        "linewidth": 2,
    },
    "geometric": {
        "color": "#C31F5C",
        "marker": "o",
        "markersize": 4,
        "linestyle": "-",
        "linewidth": 1,
    },
    "max": {
        "color": OPERATOR_COLORS["max"],
        "marker": "^",
        "markersize": 6,
        "linestyle": "-",
        "linewidth": 1.8,
    },
}


def plot_kwargs(operator: str, **overrides) -> dict:
    """Return matplotlib plot kwargs for a given operator."""
    style = {**OPERATOR_STYLE[operator], **overrides}
    return style


def plot_operator(ax, x, y, operator: str, label: str | None = None, **overrides):
    """Plot a series with the canonical operator styling."""
    style = plot_kwargs(operator, **overrides)
    marker = style.pop("marker")
    linestyle = style.pop("linestyle")
    return ax.plot(x, y, f"{marker}{linestyle}", label=label, **style)


def _ci_bounds(series) -> tuple[float, float, float]:
    import numpy as np

    values = np.asarray(series, dtype=float)
    return (
        float(np.mean(values)),
        float(np.percentile(values, 2.5)),
        float(np.percentile(values, 97.5)),
    )


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
