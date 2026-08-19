"""Reusable publication plotting helpers (Okabe–Ito, Helvetica/Arial).

Style settings are ported from ``experiments/plot_style.py`` so that new
simulation figures remain consistent with the existing CADEMAS-ML article.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties

from common.config import (
    FIG_DPI,
    FONT_SIZE,
    GROUPED_BAR_FIGSIZE,
    OPERATOR_COLORS,
    OPERATOR_FULL_LABELS,
    OPERATOR_LABELS,
    OPERATOR_MARKERS,
    OPERATOR_MARKERSIZES,
    OPERATORS,
    PLOT_FONT_SCALE,
    SAVE_PDF,
    SAVE_PNG,
    SAVE_SVG,
    SINGLE_PANEL_FIGSIZE,
    TWO_PANEL_FIGSIZE,
)

PANEL_LETTER_FP = FontProperties(family="Helvetica", weight="bold", size=12)

OPERATOR_STYLE: dict[str, dict] = {
    "linear": {
        "color": OPERATOR_COLORS["linear"],
        "marker": OPERATOR_MARKERS["linear"],
        "markersize": OPERATOR_MARKERSIZES["linear"],
        "linestyle": "-",
        "linewidth": 2.5,
        "zorder": 2,
    },
    "geometric": {
        "color": OPERATOR_COLORS["geometric"],
        "marker": OPERATOR_MARKERS["geometric"],
        "markersize": OPERATOR_MARKERSIZES["geometric"],
        "linestyle": "-",
        "linewidth": 2.0,
        "zorder": 3,
    },
    "min": {
        "color": OPERATOR_COLORS["min"],
        "marker": OPERATOR_MARKERS["min"],
        "markersize": OPERATOR_MARKERSIZES["min"],
        "linestyle": "-",
        "linewidth": 2.0,
        "zorder": 4,
    },
}

LEGEND_KW: dict = {
    "loc": "upper left",
    "frameon": True,
    "facecolor": "white",
    "edgecolor": "0.75",
    "framealpha": 1.0,
}


def apply_paper_style(font_size: int | None = None) -> None:
    """Publication defaults: Helvetica/Arial, white faces, no top/right spines."""
    if font_size is None:
        font_size = max(9, int(round(FONT_SIZE * PLOT_FONT_SCALE / 1.1)))
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
            "savefig.dpi": FIG_DPI,
            "mathtext.fontset": "custom",
            "mathtext.rm": "Helvetica",
            "mathtext.it": "Helvetica Oblique",
            "mathtext.bf": "Helvetica Bold",
            "mathtext.sf": "Helvetica",
            "mathtext.tt": "Helvetica",
        }
    )


def style_axes_frame(ax: Axes) -> None:
    """White panel, visible left/bottom spines, hidden top/right."""
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine_name in ("left", "bottom"):
        spine = ax.spines[spine_name]
        spine.set_visible(True)
        spine.set_edgecolor("black")
        spine.set_linewidth(1.0)


def add_panel_letter(ax: Axes, letter: str, *, x: float = 0.02, y: float = 1.02) -> None:
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


def save_figure(
    fig: Figure,
    path_stem: Path,
    *,
    bbox_inches: str | None = "tight",
) -> list[Path]:
    """Export a figure according to SAVE_PDF / SAVE_PNG / SAVE_SVG flags."""
    path_stem = Path(path_stem)
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    save_kw: dict = {"dpi": FIG_DPI, "bbox_inches": bbox_inches}
    if SAVE_PDF:
        pdf_path = path_stem.with_suffix(".pdf")
        fig.savefig(pdf_path, **save_kw)
        written.append(pdf_path)
    if SAVE_PNG:
        png_path = path_stem.with_suffix(".png")
        fig.savefig(png_path, **save_kw)
        written.append(png_path)
    if SAVE_SVG:
        svg_path = path_stem.with_suffix(".svg")
        fig.savefig(svg_path, **save_kw)
        written.append(svg_path)
    plt.close(fig)
    return written


def style_square_box(ax: Axes) -> None:
    """Closed square axes box, independent of the data aspect ratio."""
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("black")
        spine.set_linewidth(1.0)
    ax.set_box_aspect(1)


def plot_grouped_bars(
    ax: Axes,
    df: pd.DataFrame,
    *,
    x_col: str,
    group_col: str,
    y_col: str,
    ci_low_col: str,
    ci_high_col: str,
    x_values: tuple[float, ...] | list[float] | None = None,
    groups: tuple[str, ...] = OPERATORS,
    group_labels: dict[str, str] | None = None,
    ylabel: str = "",
    xlabel: str = "",
    ylim: tuple[float, float] | None = (0.0, 1.0),
) -> Axes:
    """Grouped barplot with asymmetric 95% CI error bars."""
    apply_paper_style()
    if group_labels is None:
        group_labels = OPERATOR_LABELS
    if x_values is None:
        x_values = tuple(sorted(df[x_col].unique()))

    x = np.arange(len(x_values), dtype=float)
    n_groups = len(groups)
    width = 0.8 / max(n_groups, 1)
    offsets = (np.arange(n_groups) - (n_groups - 1) / 2.0) * width

    for offset, group in zip(offsets, groups):
        means, yerr_lo, yerr_hi = [], [], []
        for xv in x_values:
            match = df[df[group_col] == group]
            match = match[np.isclose(match[x_col].to_numpy(), float(xv))]
            if match.empty:
                means.append(np.nan)
                yerr_lo.append(0.0)
                yerr_hi.append(0.0)
                continue
            mean = float(match[y_col].iloc[0])
            lo = float(match[ci_low_col].iloc[0])
            hi = float(match[ci_high_col].iloc[0])
            means.append(mean)
            yerr_lo.append(max(0.0, mean - lo))
            yerr_hi.append(max(0.0, hi - mean))

        ax.bar(
            x + offset,
            means,
            width=width * 0.92,
            color=OPERATOR_COLORS[group],
            edgecolor="black",
            linewidth=0.6,
            label=group_labels.get(group, group),
            zorder=3,
        )
        ax.errorbar(
            x + offset,
            means,
            yerr=np.vstack([yerr_lo, yerr_hi]),
            fmt="none",
            ecolor="black",
            elinewidth=0.9,
            capsize=3,
            zorder=4,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{xv:.2f}" for xv in x_values])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    style_axes_frame(ax)
    ax.legend(**LEGEND_KW)
    return ax


def plot_line_mean_ci(
    ax: Axes,
    df: pd.DataFrame,
    *,
    x_col: str,
    group_col: str,
    y_col: str,
    ci_low_col: str,
    ci_high_col: str,
    groups: tuple[str, ...] = OPERATORS,
    group_labels: dict[str, str] | None = None,
    xlabel: str = "",
    ylabel: str = "",
    ylim: tuple[float, float] | None = (0.0, 1.0),
    ci_alpha: float = 0.22,
) -> Axes:
    """Line plot of Monte Carlo means with shaded 95% CI bands."""
    apply_paper_style()
    if group_labels is None:
        group_labels = OPERATOR_FULL_LABELS
    for group in groups:
        sub = df[df[group_col] == group].sort_values(x_col)
        style = OPERATOR_STYLE[group]
        color = style["color"]
        ax.fill_between(
            sub[x_col],
            sub[ci_low_col],
            sub[ci_high_col],
            color=color,
            alpha=ci_alpha,
            linewidth=0,
            zorder=max(style.get("zorder", 2) - 1, 1),
        )
        ax.plot(
            sub[x_col],
            sub[y_col],
            color=color,
            marker=style["marker"],
            markersize=style["markersize"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            label=group_labels.get(group, group),
            zorder=style.get("zorder", 2),
            markeredgecolor="white",
            markeredgewidth=0.5,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    style_axes_frame(ax)
    ax.legend(**LEGEND_KW)
    return ax


def plot_boxplot(
    ax: Axes,
    data: dict[str, np.ndarray],
    *,
    ylabel: str = "",
    xlabel: str = "",
    groups: tuple[str, ...] | None = None,
    group_labels: dict[str, str] | None = None,
) -> Axes:
    """Boxplot helper with the article operator colors."""
    apply_paper_style()
    if group_labels is None:
        group_labels = OPERATOR_LABELS
    if groups is None:
        groups = tuple(data.keys())
    positions = np.arange(1, len(groups) + 1)
    boxes = ax.boxplot(
        [np.asarray(data[g], dtype=float) for g in groups],
        positions=positions,
        patch_artist=True,
        widths=0.55,
        medianprops={"color": "black", "linewidth": 1.2},
        whiskerprops={"color": "black"},
        capprops={"color": "black"},
        flierprops={"marker": "o", "markersize": 3, "alpha": 0.4},
    )
    for patch, group in zip(boxes["boxes"], groups):
        patch.set_facecolor(OPERATOR_COLORS.get(group, "#888888"))
        patch.set_edgecolor("black")
        patch.set_linewidth(0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels([group_labels.get(g, g) for g in groups])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    style_axes_frame(ax)
    return ax


def new_single_axes(figsize: tuple[float, float] | None = None) -> tuple[Figure, Axes]:
    apply_paper_style()
    fig, ax = plt.subplots(figsize=figsize or SINGLE_PANEL_FIGSIZE)
    return fig, ax


def new_square_two_panel() -> tuple[Figure, Axes, Axes]:
    """Two side-by-side square axes boxes at the Experiment 01 figure height."""
    apply_paper_style()
    height = TWO_PANEL_FIGSIZE[1]
    left_in, right_in = 0.72, 0.16
    bottom_in, top_in = 0.54, 0.34
    gap_in = 0.92
    side = height - bottom_in - top_in
    width = left_in + side + gap_in + side + right_in
    fig = plt.figure(figsize=(width, height))
    ax_a = fig.add_axes(
        [left_in / width, bottom_in / height, side / width, side / height]
    )
    ax_b = fig.add_axes(
        [
            (left_in + side + gap_in) / width,
            bottom_in / height,
            side / width,
            side / height,
        ]
    )
    return fig, ax_a, ax_b


def new_square_single_panel() -> tuple[Figure, Axes]:
    """One square axes box at the same height as the two-panel figures."""
    apply_paper_style()
    height = TWO_PANEL_FIGSIZE[1]
    left_in, right_in = 0.72, 0.28
    bottom_in, top_in = 0.54, 0.34
    side = height - bottom_in - top_in
    width = left_in + side + right_in
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes(
        [left_in / width, bottom_in / height, side / width, side / height]
    )
    return fig, ax


def square_two_panel_line_ci(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    *,
    path_stem: Path,
    x_col: str,
    xlabel: str,
    y_col_a: str,
    ci_low_a: str,
    ci_high_a: str,
    ylabel_a: str,
    ylim_a: tuple[float, float] | None,
    y_col_b: str,
    ci_low_b: str,
    ci_high_b: str,
    ylabel_b: str,
    ylim_b: tuple[float, float] | None,
    xticks: tuple[float, ...] | list[float] | None = None,
    yticks_a: list[float] | None = None,
    yticks_b: list[float] | None = None,
) -> list[Path]:
    """Two square line+CI panels sharing the same x-variable (e.g. δ)."""
    fig, ax_a, ax_b = new_square_two_panel()
    plot_line_mean_ci(
        ax_a,
        df_a,
        x_col=x_col,
        group_col="operator",
        y_col=y_col_a,
        ci_low_col=ci_low_a,
        ci_high_col=ci_high_a,
        group_labels=OPERATOR_LABELS,
        xlabel=xlabel,
        ylabel=ylabel_a,
        ylim=ylim_a,
    )
    plot_line_mean_ci(
        ax_b,
        df_b,
        x_col=x_col,
        group_col="operator",
        y_col=y_col_b,
        ci_low_col=ci_low_b,
        ci_high_col=ci_high_b,
        group_labels=OPERATOR_LABELS,
        xlabel=xlabel,
        ylabel=ylabel_b,
        ylim=ylim_b,
    )
    if xticks is not None:
        ax_a.set_xticks(list(xticks))
        ax_b.set_xticks(list(xticks))
    if yticks_a is not None:
        ax_a.set_yticks(yticks_a)
    if yticks_b is not None:
        ax_b.set_yticks(yticks_b)
    style_square_box(ax_a)
    style_square_box(ax_b)
    add_panel_letter(ax_a, "a")
    add_panel_letter(ax_b, "b")
    return save_figure(fig, path_stem, bbox_inches=None)


def square_single_panel_line_ci(
    df: pd.DataFrame,
    *,
    path_stem: Path,
    x_col: str,
    y_col: str,
    ci_low_col: str,
    ci_high_col: str,
    xlabel: str,
    ylabel: str,
    ylim: tuple[float, float] | None,
    xticks: tuple[float, ...] | list[float] | None = None,
    yticks: list[float] | None = None,
) -> list[Path]:
    """One square line+CI panel."""
    fig, ax = new_square_single_panel()
    plot_line_mean_ci(
        ax,
        df,
        x_col=x_col,
        group_col="operator",
        y_col=y_col,
        ci_low_col=ci_low_col,
        ci_high_col=ci_high_col,
        group_labels=OPERATOR_LABELS,
        xlabel=xlabel,
        ylabel=ylabel,
        ylim=ylim,
    )
    if xticks is not None:
        ax.set_xticks(list(xticks))
    if yticks is not None:
        ax.set_yticks(yticks)
    style_square_box(ax)
    return save_figure(fig, path_stem, bbox_inches=None)


def grouped_bar_figure(
    df: pd.DataFrame,
    *,
    y_col: str,
    ci_low_col: str,
    ci_high_col: str,
    ylabel: str,
    xlabel: str,
    x_values: tuple[float, ...] | list[float],
    path_stem: Path,
    ylim: tuple[float, float] | None = (0.0, 1.0),
) -> list[Path]:
    fig, ax = new_single_axes(GROUPED_BAR_FIGSIZE)
    plot_grouped_bars(
        ax,
        df,
        x_col="lambda",
        group_col="operator",
        y_col=y_col,
        ci_low_col=ci_low_col,
        ci_high_col=ci_high_col,
        x_values=x_values,
        ylabel=ylabel,
        xlabel=xlabel,
        ylim=ylim,
    )
    fig.tight_layout()
    return save_figure(fig, path_stem)


def policy_violation_two_panel(
    bar_df: pd.DataFrame,
    dense_df: pd.DataFrame,
    *,
    path_stem: Path,
    x_values: tuple[float, ...] | list[float],
) -> list[Path]:
    """Side-by-side (a) grouped bars and (b) dense $V(\\lambda)$ sweep.

    Both plot areas are closed square boxes. Figure height is kept; width is
    set so the two squares fit with room for labels, without shrinking height.
    """
    fig, ax_a, ax_b = new_square_two_panel()
    plot_grouped_bars(
        ax_a,
        bar_df,
        x_col="lambda",
        group_col="operator",
        y_col="policy_violation_rate_mean",
        ci_low_col="policy_violation_rate_ci_low",
        ci_high_col="policy_violation_rate_ci_high",
        x_values=x_values,
        group_labels=OPERATOR_LABELS,
        ylabel=r"Policy violation rate $V$",
        xlabel=r"Trade-off parameter $\lambda$",
        ylim=(0.0, 1.0),
    )
    plot_line_mean_ci(
        ax_b,
        dense_df,
        x_col="lambda",
        group_col="operator",
        y_col="policy_violation_rate_mean",
        ci_low_col="policy_violation_rate_ci_low",
        ci_high_col="policy_violation_rate_ci_high",
        group_labels=OPERATOR_LABELS,
        xlabel=r"Trade-off parameter $\lambda$",
        ylabel=r"Policy violation rate $V$",
        ylim=(-0.08, 1.0),
    )
    ax_b.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    style_square_box(ax_a)
    style_square_box(ax_b)
    add_panel_letter(ax_a, "a")
    add_panel_letter(ax_b, "b")
    return save_figure(fig, path_stem, bbox_inches=None)
