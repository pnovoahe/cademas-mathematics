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
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from common.config import (
    AGREEMENT_LAMBDA_ALPHAS,
    FIG_DPI,
    FONT_SIZE,
    GROUPED_BAR_FIGSIZE,
    OKABE_ITO,
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
    TOP_K,
    TWO_PANEL_FIGSIZE,
    agreement_operator_specs,
)
from common.metrics import top_k_indices

PANEL_LETTER_FP = FontProperties(family="Helvetica", weight="bold", size=12)

# Panel (a)/(b–d) $(R,Q)$ quadrant strata.
COLOR_WEAK = "#7F7F7F"  # lower-left
COLOR_STRONG = "#009E73"  # upper-right (green)
COLOR_RWEAK = "#0072B2"  # upper-left (blue)
COLOR_QWEAK = "#F0E442"  # lower-right (yellow)
COLOR_VETO = "#D55E00"  # Q=0 (strong orange)
QUADRANT_THRESHOLD = 0.5

# Draw order: background first, then highlights. All strata use circles;
# color + alpha carry the visual encoding (overlap should remain readable).
STRATA_STYLE: tuple[dict, ...] = (
    {
        "key": "weak",
        "label": "Weak",
        "color": COLOR_WEAK,
        "marker": "o",
        "alpha": 0.28,
        "size": 18.0,
        "zorder": 2,
    },
    {
        "key": "strong",
        "label": "Strong",
        "color": COLOR_STRONG,
        "marker": "o",
        "alpha": 0.32,
        "size": 18.0,
        "zorder": 3,
    },
    {
        "key": "rweak",
        "label": "Rweak",
        "color": COLOR_RWEAK,
        "marker": "o",
        "alpha": 0.40,
        "size": 18.0,
        "zorder": 4,
    },
    {
        "key": "qweak",
        "label": "Qweak",
        "color": COLOR_QWEAK,
        "marker": "o",
        "alpha": 0.42,
        "size": 18.0,
        "zorder": 4,
    },
    {
        "key": "veto",
        "label": r"Veto ($Q=0$)",
        "color": COLOR_VETO,
        "marker": "o",
        "alpha": 0.50,
        "size": 20.0,
        "zorder": 5,
    },
)

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


def add_panel_letter(
    ax: Axes,
    letter: str,
    *,
    x: float = 0.02,
    y: float = 1.02,
    font_size: int | None = None,
) -> None:
    fp = (
        PANEL_LETTER_FP
        if font_size is None
        else FontProperties(family="Helvetica", weight="bold", size=font_size)
    )
    ax.text(
        x,
        y,
        f"({letter})",
        transform=ax.transAxes,
        fontproperties=fp,
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


def style_square_axes_frame(ax: Axes) -> None:
    """Square plot area with only left/bottom spines visible."""
    style_axes_frame(ax)
    ax.set_box_aspect(1)


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
    clip_error_nonneg: bool = True,
    apply_style: bool = True,
    show_legend: bool = True,
    xtick_format: str = "{:.2f}",
) -> Axes:
    """Grouped barplot with asymmetric 95% CI error bars."""
    if apply_style:
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
            err_lo = mean - lo
            err_hi = hi - mean
            if clip_error_nonneg:
                err_lo = max(0.0, err_lo)
                err_hi = max(0.0, err_hi)
            yerr_lo.append(err_lo)
            yerr_hi.append(err_hi)

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
    ax.set_xticklabels([xtick_format.format(xv) for xv in x_values])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    style_axes_frame(ax)
    if show_legend:
        ax.legend(**LEGEND_KW)
    return ax


def _agreement_config_bar_style(
    operator: str, lam: float | None
) -> tuple[str, float]:
    """Return ``(facecolor, alpha)`` for one agreement-configuration bar."""
    color = OPERATOR_COLORS[operator]
    if lam is None:
        return color, 1.0
    alpha = AGREEMENT_LAMBDA_ALPHAS.get(float(lam), 1.0)
    return color, alpha


def _agreement_config_bar_offsets(
    specs: tuple[tuple[str, str, float | None, str], ...],
    *,
    total_width: float = 0.82,
    family_gap: float = 0.055,
) -> tuple[np.ndarray, float]:
    """Bar centres and width with small gaps between linear / geometric / min."""
    family_sizes: list[int] = []
    prev_op: str | None = None
    for _cid, operator, _lam, _label in specs:
        if operator != prev_op:
            family_sizes.append(1)
            prev_op = operator
        else:
            family_sizes[-1] += 1

    n_bars = sum(family_sizes)
    n_gaps = max(len(family_sizes) - 1, 0)
    bar_w = (total_width - n_gaps * family_gap) / max(n_bars, 1)
    offsets: list[float] = []
    cursor = -total_width / 2.0 + bar_w / 2.0
    for fi, count in enumerate(family_sizes):
        if fi > 0:
            cursor += family_gap
        for _ in range(count):
            offsets.append(cursor)
            cursor += bar_w
    off = np.asarray(offsets, dtype=float)
    off -= float(off.mean())
    return off, bar_w


def _sensitivity_bar_panel_ylim(
    agg_metrics: pd.DataFrame,
    metric: str,
    sigma_values: tuple[float, ...] | list[float],
) -> tuple[float, float]:
    """Data-driven y-limits from 95\\% CI bounds (figure $\\sigma_R$ grid only)."""
    if metric == "policy_violation_rate":
        return (0.0, 1.0)

    sub = agg_metrics[
        np.isin(agg_metrics["sigma_r"].to_numpy(), [float(s) for s in sigma_values])
    ]
    low_col, high_col = f"{metric}_ci_low", f"{metric}_ci_high"
    if sub.empty:
        return (0.0, 1.05)

    lo = float(sub[low_col].min())
    hi = float(sub[high_col].max())
    span = max(hi - lo, 0.08)
    pad = 0.06 * span
    y0, y1 = lo - pad, hi + pad

    if metric == "jaccard_top_k":
        y0 = max(0.0, y0)
        y1 = min(1.05, max(1.0, y1))
        return (y0, y1)

    # Kendall τ: allow modest negative values; do not force the axis to −1.
    if metric == "kendall_tau":
        y0 = min(y0, -0.12)
        y0 = max(y0, -0.45)
        y1 = min(1.05, max(1.0, y1))
        return (y0, y1)

    return (y0, y1)


def plot_agreement_config_bars(
    ax: Axes,
    df: pd.DataFrame,
    *,
    x_col: str,
    group_col: str,
    y_col: str,
    ci_low_col: str,
    ci_high_col: str,
    x_values: tuple[float, ...] | list[float],
    specs: tuple[tuple[str, str, float | None, str], ...] | None = None,
    ylabel: str = "",
    xlabel: str = "",
    ylim: tuple[float, float] | None = (0.0, 1.0),
    clip_error_nonneg: bool = True,
    show_legend: bool = False,
    xtick_format: str = "{:.2f}",
    reference_y: float | None = None,
    family_gaps: bool = True,
    title: str = "",
    legend_anchor_y: float = 0.96,
) -> Axes:
    """Grouped bars for seven agreement configs (λ variants share hue, vary alpha)."""
    if specs is None:
        specs = agreement_operator_specs()

    groups = tuple(s[0] for s in specs)
    x = np.arange(len(x_values), dtype=float)
    if family_gaps:
        offsets, bar_w = _agreement_config_bar_offsets(specs)
    else:
        n_groups = len(groups)
        bar_w = 0.8 / max(n_groups, 1)
        offsets = (np.arange(n_groups) - (n_groups - 1) / 2.0) * bar_w

    legend_handles: list[Patch] = []
    for offset, (cid, operator, lam, label) in zip(offsets, specs):
        color, alpha = _agreement_config_bar_style(operator, lam)
        means, yerr_lo, yerr_hi = [], [], []
        for xv in x_values:
            match = df[df[group_col] == cid]
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
            err_lo = mean - lo
            err_hi = hi - mean
            if clip_error_nonneg:
                err_lo = max(0.0, err_lo)
                err_hi = max(0.0, err_hi)
            yerr_lo.append(err_lo)
            yerr_hi.append(err_hi)

        ax.bar(
            x + offset,
            means,
            width=bar_w,
            color=color,
            alpha=alpha,
            edgecolor=color,
            linewidth=0.0,
            zorder=3,
        )
        ax.errorbar(
            x + offset,
            means,
            yerr=np.vstack([yerr_lo, yerr_hi]),
            fmt="none",
            ecolor="black",
            elinewidth=0.9,
            capsize=1.8,
            zorder=4,
        )
        legend_handles.append(
            Patch(
                facecolor=color,
                edgecolor=color,
                linewidth=0.0,
                alpha=alpha,
                label=label,
            )
        )

    ax.set_xticks(x)
    ax.set_xticklabels([xtick_format.format(xv) for xv in x_values])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, pad=4)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if reference_y is not None:
        ax.axhline(
            float(reference_y),
            color="black",
            linestyle="--",
            linewidth=0.95,
            dashes=(3.0, 2.0),
            zorder=2,
        )
    style_axes_frame(ax)
    if show_legend:
        legend_kw = {k: v for k, v in LEGEND_KW.items() if k != "loc"}
        ax.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, legend_anchor_y),
            ncol=4,
            columnspacing=0.55,
            handlelength=1.0,
            handletextpad=0.35,
            **legend_kw,
        )
    return ax


def plot_agreement_config_boxplots(
    ax: Axes,
    df: pd.DataFrame,
    *,
    x_col: str,
    group_col: str,
    y_col: str,
    x_values: tuple[float, ...] | list[float],
    specs: tuple[tuple[str, str, float | None, str], ...] | None = None,
    ylabel: str = "",
    xlabel: str = "",
    ylim: tuple[float, float] | None = (0.0, 1.0),
    show_legend: bool = False,
    xtick_format: str = "{:.2f}",
) -> Axes:
    """Grouped notched boxplots for seven agreement configs."""
    if specs is None:
        specs = agreement_operator_specs()

    x = np.arange(len(x_values), dtype=float)
    n_groups = len(specs)
    width = 0.8 / max(n_groups, 1)
    offsets = (np.arange(n_groups) - (n_groups - 1) / 2.0) * width
    legend_handles: list[Patch] = []

    for offset, (cid, operator, lam, label) in zip(offsets, specs):
        color, alpha = _agreement_config_bar_style(operator, lam)
        box_data: list[np.ndarray] = []
        positions: list[float] = []
        for i, xv in enumerate(x_values):
            sub = df[df[group_col] == cid]
            sub = sub[np.isclose(sub[x_col].to_numpy(), float(xv))]
            values = sub[y_col].dropna().to_numpy(dtype=float)
            box_data.append(values)
            positions.append(float(i + offset))

        bp = ax.boxplot(
            box_data,
            positions=positions,
            widths=width * 0.92,
            notch=True,
            patch_artist=True,
            showfliers=False,
            zorder=3,
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(alpha)
            patch.set_edgecolor("black")
            patch.set_linewidth(0.6)
        for key in ("whiskers", "caps", "medians"):
            for artist in bp.get(key, []):
                artist.set_color("black")
                artist.set_linewidth(0.9 if key == "medians" else 0.8)
        legend_handles.append(
            Patch(
                facecolor=color,
                edgecolor="black",
                linewidth=0.6,
                alpha=alpha,
                label=label,
            )
        )

    ax.set_xticks(x)
    ax.set_xticklabels([xtick_format.format(xv) for xv in x_values])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    style_axes_frame(ax)
    if show_legend:
        legend_kw = {k: v for k, v in LEGEND_KW.items() if k != "loc"}
        ax.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=4,
            columnspacing=0.55,
            handlelength=1.0,
            handletextpad=0.35,
            **legend_kw,
        )
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
    apply_style: bool = True,
    show_legend: bool = True,
    series_scale: float = 1.0,
) -> Axes:
    """Line plot of Monte Carlo means with shaded 95% CI bands."""
    if apply_style:
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
            markersize=style["markersize"] * series_scale,
            linestyle=style["linestyle"],
            linewidth=style["linewidth"] * series_scale,
            label=group_labels.get(group, group),
            zorder=style.get("zorder", 2),
            markeredgecolor="white",
            markeredgewidth=0.5 * series_scale,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    style_axes_frame(ax)
    if show_legend:
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


def _context_strata_masks(
    R: np.ndarray,
    Q: np.ndarray,
    *,
    threshold: float = QUADRANT_THRESHOLD,
) -> dict[str, np.ndarray]:
    """Return quadrant masks keyed by stratum name.

    Dashed guides at ``threshold`` split the plane into four quadrants:
    - weak: lower-left (excluding vetoes);
    - strong: upper-right;
    - qweak: lower-right with $Q>0$;
    - rweak: upper-left;
    - veto: $Q=0$.
    """
    r = np.asarray(R, dtype=float)
    q = np.asarray(Q, dtype=float)
    thr = float(threshold)
    veto = q <= 0.0
    non_veto = ~veto
    return {
        "veto": veto,
        "qweak": non_veto & (r >= thr) & (q < thr),
        "rweak": non_veto & (r < thr) & (q >= thr),
        "weak": non_veto & (r < thr) & (q < thr),
        "strong": non_veto & (r >= thr) & (q >= thr),
    }


def _scatter_strata(
    ax: Axes,
    x: np.ndarray,
    y: np.ndarray,
    masks: dict[str, np.ndarray],
    *,
    size_scale: float = 1.0,
    alpha_scale: float = 1.0,
    rasterized: bool = True,
) -> None:
    """Draw stratum markers in background-to-foreground order."""
    for style in STRATA_STYLE:
        mask = masks[style["key"]]
        if not np.any(mask):
            continue
        kw: dict = {
            "s": style["size"] * size_scale,
            "c": style["color"],
            "marker": style["marker"],
            "alpha": min(1.0, style["alpha"] * alpha_scale),
            "label": style["label"],
            "rasterized": rasterized,
            "zorder": style["zorder"],
        }
        if style["marker"] in {"+", "x"}:
            kw["linewidths"] = style.get("linewidths", 1.0)
        else:
            kw["linewidths"] = 0
        ax.scatter(x[mask], y[mask], **kw)


def _draw_rq_scatter(
    ax: Axes,
    R: np.ndarray,
    Q: np.ndarray,
    is_veto: np.ndarray,
    *,
    threshold: float = QUADRANT_THRESHOLD,
) -> None:
    """Scatter $R$ vs $Q$ with quadrant strata and dashed $R=Q=0.5$ guides."""
    del is_veto  # strata derived from $(R,Q)$ for visual consistency
    R = np.asarray(R, dtype=float)
    Q = np.asarray(Q, dtype=float)
    masks = _context_strata_masks(R, Q, threshold=threshold)

    ax.axhline(
        float(threshold),
        color="black",
        linestyle="--",
        linewidth=0.95,
        dashes=(3.0, 2.0),
        zorder=1,
    )
    ax.axvline(
        float(threshold),
        color="black",
        linestyle="--",
        linewidth=0.95,
        dashes=(3.0, 2.0),
        zorder=1,
    )
    _scatter_strata(ax, R, Q, masks)
    ax.set_xlabel(r"Predictive score $R$")
    ax.set_ylabel(r"Contextual score $Q$")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    style_axes_frame(ax)


def _add_top_horizontal_legend(
    ax: Axes,
    handles,
    *,
    x_center: float,
    y_bottom: float,
    y_top: float,
    legend_y: float,
    ncol: int | None = None,
    columnspacing: float = 0.85,
    handlelength: float = 1.0,
    handletextpad: float = 0.35,
    markerscale: float | None = None,
) -> None:
    """Centered horizontal legend in the upper margin created by extending $y$ limits."""
    ax.set_ylim(y_bottom, y_top)
    if ncol is None:
        ncol = len(handles)
    legend_base = {k: v for k, v in LEGEND_KW.items() if k != "loc"}
    legend_kw = {
        "columnspacing": columnspacing,
        "handlelength": handlelength,
        "handletextpad": handletextpad,
    }
    if markerscale is not None:
        legend_kw["markerscale"] = markerscale
    ax.legend(
        handles=handles,
        ncol=ncol,
        loc="lower center",
        bbox_to_anchor=(x_center, legend_y),
        bbox_transform=ax.transData,
        borderaxespad=0.0,
        **legend_kw,
        **legend_base,
    )


def _add_two_row_top_legend(
    ax: Axes,
    *,
    top_handles,
    bottom_handles,
    x_center: float,
    y_bottom: float,
    y_top: float,
    top_y: float,
    bottom_y: float,
    top_ncol: int,
    bottom_ncol: int,
    top_in_axes_coords: bool = False,
    columnspacing: float = 0.85,
    handlelength: float = 1.0,
    handletextpad: float = 0.35,
) -> None:
    """Two-row centered legend in the upper margin (row-major layout)."""
    ax.set_ylim(y_bottom, y_top)
    legend_base = {k: v for k, v in LEGEND_KW.items() if k != "loc"}
    legend_kw = {
        "borderaxespad": 0.0,
        "columnspacing": columnspacing,
        "handlelength": handlelength,
        "handletextpad": handletextpad,
        **legend_base,
    }
    if top_in_axes_coords:
        top_bbox = (0.5, top_y)
        top_transform = ax.transAxes
        top_loc = "lower center"
    else:
        top_bbox = (x_center, top_y)
        top_transform = ax.transData
        top_loc = "lower center"
    top_legend = ax.legend(
        handles=top_handles,
        ncol=top_ncol,
        loc=top_loc,
        bbox_to_anchor=top_bbox,
        bbox_transform=top_transform,
        **legend_kw,
    )
    ax.add_artist(top_legend)
    ax.legend(
        handles=bottom_handles,
        ncol=bottom_ncol,
        loc="lower center",
        bbox_to_anchor=(x_center, bottom_y),
        bbox_transform=ax.transData,
        **legend_kw,
    )


def _finish_panel_a_legend(
    ax: Axes,
    *,
    y_top: float = 1.18,
    legend_y: float = 1.03,
) -> None:
    handles, _ = ax.get_legend_handles_labels()
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    _add_top_horizontal_legend(
        ax,
        handles,
        x_center=0.5,
        y_bottom=-0.02,
        y_top=y_top,
        legend_y=legend_y,
        ncol=3,
        columnspacing=0.55,
        handletextpad=0.25,
    )


def _top_k_p_cutoff(
    P: np.ndarray,
    *,
    R: np.ndarray,
    case_id: np.ndarray | None,
    K: int = TOP_K,
) -> float:
    """Minimum $P$ among the Top-$K$ set (manuscript tie-break protocol)."""
    P = np.asarray(P, dtype=float)
    R = np.asarray(R, dtype=float)
    idx = top_k_indices(P, K, R=R, case_id=case_id)
    return float(np.min(P[idx]))


def _stratum_density_profile(
    values: np.ndarray,
    y_grid: np.ndarray,
) -> np.ndarray | None:
    """Return count-weighted KDE density on ``y_grid``, or ``None`` if too few points."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return None
    if float(np.ptp(values)) < 1e-12:
        dens = np.zeros_like(y_grid, dtype=float)
        idx = int(np.argmin(np.abs(y_grid - float(values[0]))))
        dens[idx] = float(values.size)
        return dens
    from scipy.stats import gaussian_kde

    kde = gaussian_kde(values)
    return kde(y_grid) * float(values.size)


def _violin_halfwidth_interpolator(
    values: np.ndarray,
    *,
    half_w: float,
    y_grid: np.ndarray | None = None,
):
    """Return ``half_width(y)`` following the full-sample violin density envelope."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if y_grid is None:
        y_grid = np.linspace(0.0, 1.0, 220)
    dens = _stratum_density_profile(values, y_grid)
    if dens is None:
        # Fallback: constant width for tiny samples.
        return lambda y: np.full(np.shape(y), half_w * 0.35, dtype=float)
    dens = dens / max(float(np.max(dens)), 1e-12)
    # Soft floor so sparse tails still get a thin band of points.
    dens = np.maximum(dens, 0.08)
    half = half_w * dens

    def half_at(y: np.ndarray | float) -> np.ndarray:
        return np.interp(np.asarray(y, dtype=float), y_grid, half)

    return half_at


def _plot_single_lambda_operator_violins(
    ax: Axes,
    p_data: dict[tuple[str, float], np.ndarray],
    *,
    lam: float,
    R: np.ndarray,
    Q: np.ndarray,
    case_id: np.ndarray | None = None,
    operators: tuple[str, ...] = OPERATORS,
    operator_labels: dict[str, str] | None = None,
    top_k: int = TOP_K,
    threshold: float = QUADRANT_THRESHOLD,
    show_ylabel: bool = True,
) -> None:
    """One $\\lambda$ panel: black-outline violins with in-contour stratum points."""
    if operator_labels is None:
        operator_labels = OPERATOR_LABELS

    n_ops = len(operators)
    positions = np.arange(1, n_ops + 1, dtype=float)
    violin_width = 0.62
    half_w = violin_width / 2.0
    masks = _context_strata_masks(R, Q, threshold=threshold)
    y_grid = np.linspace(0.0, 1.0, 220)
    jitter_rng = np.random.default_rng(42 + int(round(float(lam) * 1000)))

    for i_op, operator in enumerate(operators):
        pos = float(positions[i_op])
        values = np.asarray(p_data[(operator, lam)], dtype=float)
        half_at = _violin_halfwidth_interpolator(values, half_w=half_w, y_grid=y_grid)

        # Points first (behind), then outline / median / Top-K on top.
        for style in STRATA_STYLE:
            overlay = values[masks[style["key"]]]
            if overlay.size == 0:
                continue
            # Stay inside the outline: local half-width of the full violin at each P.
            local_half = half_at(overlay) * 0.92
            jitter = jitter_rng.uniform(-1.0, 1.0, size=overlay.size) * local_half
            ax.scatter(
                pos + jitter,
                overlay,
                s=style["size"] * 0.70,
                c=style["color"],
                marker="o",
                alpha=style["alpha"],
                linewidths=0,
                zorder=1 + 0.1 * style["zorder"],
            )

        parts = ax.violinplot(
            [values],
            positions=[pos],
            widths=violin_width,
            showmeans=False,
            showmedians=True,
            showextrema=False,
        )
        for body in parts["bodies"]:
            body.set_facecolor("none")
            body.set_alpha(1.0)
            body.set_edgecolor("black")
            body.set_linewidth(1.2)
            body.set_zorder(6)
        if parts.get("cmedians") is not None:
            parts["cmedians"].set_color("black")
            parts["cmedians"].set_linewidth(1.15)
            parts["cmedians"].set_zorder(8)

        p_cutoff = _top_k_p_cutoff(values, R=R, case_id=case_id, K=top_k)
        ax.plot(
            [pos - half_w, pos + half_w],
            [p_cutoff, p_cutoff],
            color="black",
            linestyle="--",
            linewidth=1.05,
            dashes=(3.0, 2.0),
            zorder=9,
            clip_on=True,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels([operator_labels.get(op, op) for op in operators])
    ax.set_xlabel("")
    if show_ylabel:
        ax.set_ylabel(r"Aggregated score $P$")
    else:
        ax.set_ylabel("")
    ax.set_ylim(-0.02, 1.02)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_title(rf"$\lambda={lam:.2f}$", pad=4)
    style_axes_frame(ax)


def _finish_panel_c_legend(
    ax: Axes,
    *,
    y_top: float = 1.16,
    legend_y: float = 1.03,
) -> None:
    handles, _ = ax.get_legend_handles_labels()
    y_bottom = float(ax.get_ylim()[0])
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    _add_top_horizontal_legend(
        ax,
        handles,
        x_center=0.5,
        y_bottom=y_bottom,
        y_top=y_top,
        legend_y=legend_y,
        ncol=len(OPERATORS),
        columnspacing=2.0,
        handlelength=1.6,
        handletextpad=0.55,
        markerscale=0.9,
    )


def _policy_compliance_overview_layout(
    fig: Figure,
    side_in: float,
    *,
    left_in: float = 0.68,
    right_in: float = 0.42,
    top_in: float = 0.38,
    bottom_in: float = 0.48,
    col_gap_in: float = 0.54,
    row_gap_in: float = 0.48,
    row_gap_top_in: float | None = None,
) -> dict[str, tuple[float, float, float, float]]:
    """Layout: (a)|(b); (c)|(d); (e) full-width dense $V(\\lambda)$."""
    fig_w, fig_h = fig.get_size_inches()
    inv_w, inv_h = 1.0 / fig_w, 1.0 / fig_h
    full_w_in = 2.0 * side_in + col_gap_in
    gap_mid = row_gap_in
    gap_top = row_gap_in if row_gap_top_in is None else row_gap_top_in
    e_bottom_in = bottom_in
    mid_bottom_in = e_bottom_in + side_in + gap_mid
    top_bottom_in = mid_bottom_in + side_in + gap_top
    right_left_in = left_in + side_in + col_gap_in
    return {
        "a": (left_in * inv_w, top_bottom_in * inv_h, side_in * inv_w, side_in * inv_h),
        "b": (right_left_in * inv_w, top_bottom_in * inv_h, side_in * inv_w, side_in * inv_h),
        "c": (left_in * inv_w, mid_bottom_in * inv_h, side_in * inv_w, side_in * inv_h),
        "d": (right_left_in * inv_w, mid_bottom_in * inv_h, side_in * inv_w, side_in * inv_h),
        "e": (left_in * inv_w, e_bottom_in * inv_h, full_w_in * inv_w, side_in * inv_h),
    }


def _operator_agreement_two_panel_layout(
    fig: Figure,
    *,
    side_in: float = 2.45,
    left_in: float = 0.68,
    right_in: float = 0.42,
    top_in: float = 0.38,
    bottom_in: float = 0.55,
    col_gap_in: float = 0.78,
) -> dict[str, tuple[float, float, float, float]]:
    """Side-by-side panels for Kendall / Jaccard agreement heatmaps."""
    fig_w, fig_h = fig.get_size_inches()
    inv_w, inv_h = 1.0 / fig_w, 1.0 / fig_h
    right_left_in = left_in + side_in + col_gap_in
    y_in = bottom_in
    return {
        "a": (left_in * inv_w, y_in * inv_h, side_in * inv_w, side_in * inv_h),
        "b": (right_left_in * inv_w, y_in * inv_h, side_in * inv_w, side_in * inv_h),
    }


# Symmetric vertical inset (data units) for in-panel agreement colorbars:
# same gap above $A_M$ and below the top of the $y$-axis.
AGREEMENT_COLORBAR_Y_GAP: float = 0.18


def _agreement_heatmap_crop_slices(
    specs: tuple[tuple[str, str, float | None, str], ...] | None = None,
) -> tuple[tuple[int, ...], tuple[int, ...], float, float, float]:
    """Row/col slices for cropped heatmaps and in-panel colorbar anchors."""
    if specs is None:
        specs = agreement_operator_specs()
    n = len(specs)
    row_idx = tuple(range(1, n))  # omit $A_L(0.1)$ on $y$
    col_idx = tuple(range(n - 1))  # omit $A_M$ on $x$
    x_cb = float(len(col_idx) - 1)  # $A_G(0.9)$
    n_rows = len(row_idx)
    gap = float(AGREEMENT_COLORBAR_Y_GAP)
    y_plot_top = -0.5  # imshow half-cell offset (origin upper)
    y_am_row_top = float(n_rows) - 1.5
    y_cb_bottom = y_am_row_top - gap
    y_cb_top = y_plot_top + gap
    return row_idx, col_idx, x_cb, y_cb_top, y_cb_bottom


def _crop_agreement_matrix(
    mat: np.ndarray,
    row_idx: tuple[int, ...],
    col_idx: tuple[int, ...],
) -> np.ndarray:
    """Submatrix keeping only pairs with original row index $>$ column index."""
    out = np.full((len(row_idx), len(col_idx)), np.nan, dtype=float)
    for i_sub, i_orig in enumerate(row_idx):
        for j_sub, j_orig in enumerate(col_idx):
            if i_orig > j_orig:
                out[i_sub, j_sub] = mat[i_orig, j_orig]
    return out


def _plot_cropped_operator_agreement_heatmap(
    ax: Axes,
    agreement_pairs: pd.DataFrame,
    *,
    metric: str,
    title: str = "",
    sigma_r: float | None = None,
    vmin: float = 0.0,
    vmax: float = 1.0,
    cmap: LinearSegmentedColormap | str | None = None,
    annotate: bool = True,
    fontsize: float = 7.0,
    cell_fontsize: float | None = None,
    face_alpha: float = 0.55,
):
    """Cropped lower-triangle heatmap (no $A_L(0.1)$ on $y$, no $A_M$ on $x$)."""
    specs = agreement_operator_specs()
    config_ids = tuple(s[0] for s in specs)
    row_idx, col_idx, _, _, _ = _agreement_heatmap_crop_slices(specs)
    tick_x = tuple(specs[j][3] for j in col_idx)
    tick_y = tuple(specs[i][3] for i in row_idx)
    mat = _pair_matrix(
        agreement_pairs,
        metric=metric,
        config_ids=config_ids,
        sigma_r=sigma_r,
    )
    cropped = _crop_agreement_matrix(mat, row_idx, col_idx)
    return _plot_triangular_agreement_heatmap(
        ax,
        cropped,
        tick_labels=tick_x,
        ytick_labels=tick_y,
        title=title,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        annotate=annotate,
        fontsize=fontsize,
        cell_fontsize=cell_fontsize,
        face_alpha=face_alpha,
        mask_mode="finite",
    )


def _draw_operator_agreement_heatmap_panel(
    ax: Axes,
    agreement_pairs: pd.DataFrame,
    *,
    metric: str,
    title: str,
    vmin: float,
    vmax: float,
    cmap: LinearSegmentedColormap | str | None = None,
    cell_fontsize: float = 9.0,
    tick_fontsize: float = 6.5,
    colorbar: bool = True,
) -> None:
    im = _plot_cropped_operator_agreement_heatmap(
        ax,
        agreement_pairs,
        metric=metric,
        title=title,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        fontsize=tick_fontsize,
        cell_fontsize=cell_fontsize,
    )
    if colorbar:
        _, _, x_cb, y_cb_top, y_cb_bottom = _agreement_heatmap_crop_slices()
        _add_inpanel_agreement_colorbar(
            ax,
            im,
            vmin=vmin,
            vmax=vmax,
            n_ticks=5,
            tick_fontsize=tick_fontsize - 0.5,
            x_data=x_cb,
            y_top_data=y_cb_top,
            y_bottom_data=y_cb_bottom,
        )


def policy_compliance_three_panel(
    *,
    R: np.ndarray,
    Q: np.ndarray,
    is_veto: np.ndarray,
    case_id: np.ndarray | None,
    p_data: dict[tuple[str, float], np.ndarray],
    dense_df: pd.DataFrame,
    path_stem: Path,
    lambda_values: tuple[float, ...] | list[float],
) -> list[Path]:
    """Overview: (a) $R$–$Q$; (b–d) violins; (e) $V(\\lambda)$."""
    panel_fs = max(8, int(round(FONT_SIZE * PLOT_FONT_SCALE / 1.45)))
    apply_paper_style(font_size=panel_fs)

    lambda_values = tuple(float(x) for x in lambda_values)
    if len(lambda_values) != 3:
        raise ValueError("policy_compliance_three_panel expects exactly three lambda values.")

    side_in = 2.45
    left_in, right_in = 0.68, 0.42
    top_in, bottom_in = 0.38, 0.55
    # Wider column gap so panel (b) can carry its own y-axis title.
    col_gap_in, row_gap_in = 0.78, 0.50
    # Slightly more space between row 1 (a–b) and row 2 (c–d) so λ titles
    # clear the x tick labels above.
    row_gap_top_in = 0.62
    fig_w = left_in + 2.0 * side_in + col_gap_in + right_in
    fig_h = bottom_in + 3.0 * side_in + 2.0 * row_gap_in + row_gap_top_in + top_in
    fig = plt.figure(figsize=(fig_w, fig_h))

    rects = _policy_compliance_overview_layout(
        fig,
        side_in,
        left_in=left_in,
        right_in=right_in,
        top_in=top_in,
        bottom_in=bottom_in,
        col_gap_in=col_gap_in,
        row_gap_in=row_gap_in,
        row_gap_top_in=row_gap_top_in,
    )

    letter_fs = panel_fs + 1
    ax_a = fig.add_axes(rects["a"])
    _draw_rq_scatter(ax_a, R, Q, is_veto)
    style_square_axes_frame(ax_a)
    _finish_panel_a_legend(ax_a)
    add_panel_letter(ax_a, "a", font_size=letter_fs)

    for letter, key, lam, show_ylabel in (
        ("b", "b", lambda_values[0], True),
        ("c", "c", lambda_values[1], True),
        ("d", "d", lambda_values[2], False),
    ):
        ax = fig.add_axes(rects[key])
        _plot_single_lambda_operator_violins(
            ax,
            p_data,
            lam=lam,
            R=R,
            Q=Q,
            case_id=case_id,
            show_ylabel=show_ylabel,
        )
        style_square_axes_frame(ax)
        add_panel_letter(ax, letter, font_size=letter_fs)

    ax_e = fig.add_axes(rects["e"])
    plot_line_mean_ci(
        ax_e,
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
        apply_style=False,
        show_legend=False,
        series_scale=0.9,
    )
    _finish_panel_c_legend(ax_e)
    add_panel_letter(ax_e, "e", font_size=letter_fs)

    return save_figure(fig, path_stem, bbox_inches=None)


def policy_compliance_operator_agreement(
    *,
    agreement_pairs: pd.DataFrame | None = None,
    path_stem: Path,
) -> list[Path]:
    """Two-panel figure: Kendall $\\tau$ and Jaccard Top-$K$ agreement heatmaps."""
    panel_fs = max(8, int(round(FONT_SIZE * PLOT_FONT_SCALE / 1.45)))
    apply_paper_style(font_size=panel_fs)

    side_in = 2.45
    left_in, right_in = 0.68, 0.42
    top_in, bottom_in = 0.38, 0.55
    col_gap_in = 0.78
    fig_w = left_in + 2.0 * side_in + col_gap_in + right_in
    fig_h = bottom_in + side_in + top_in
    fig = plt.figure(figsize=(fig_w, fig_h))
    rects = _operator_agreement_two_panel_layout(
        fig,
        side_in=side_in,
        left_in=left_in,
        right_in=right_in,
        top_in=top_in,
        bottom_in=bottom_in,
        col_gap_in=col_gap_in,
    )
    letter_fs = panel_fs + 1

    if agreement_pairs is None:
        agreement_pairs = pd.DataFrame(
            columns=["operator_a", "operator_b", "kendall_tau_mean", "jaccard_top_k_mean"]
        )

    ax_a = fig.add_axes(rects["a"])
    _draw_operator_agreement_heatmap_panel(
        ax_a,
        agreement_pairs,
        metric="kendall_tau",
        title=r"Kendall $\tau$",
        vmin=-1.0,
        vmax=1.0,
    )
    style_square_axes_frame(ax_a)
    add_panel_letter(ax_a, "a", font_size=letter_fs)

    ax_b = fig.add_axes(rects["b"])
    _draw_operator_agreement_heatmap_panel(
        ax_b,
        agreement_pairs,
        metric="jaccard_top_k",
        title=r"Jaccard Top-$K$",
        vmin=0.0,
        vmax=1.0,
        cmap=_agreement_jaccard_cmap(),
    )
    style_square_axes_frame(ax_b)
    add_panel_letter(ax_b, "b", font_size=letter_fs)

    return save_figure(fig, path_stem, bbox_inches=None)


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


def _sensitivity_2x3_layout(
    fig: Figure,
    *,
    left_in: float = 0.72,
    right_in: float = 0.28,
    top_in: float = 0.42,
    bottom_in: float = 0.52,
    col_gap_in: float = 0.42,
    row_gap_in: float = 0.58,
    panel_w_in: float = 2.15,
    panel_h_in: float = 2.15,
) -> dict[str, tuple[float, float, float, float]]:
    """Six equal panels in a 2×3 grid (row-major a–f)."""
    fig_w, fig_h = fig.get_size_inches()
    inv_w, inv_h = 1.0 / fig_w, 1.0 / fig_h
    rects: dict[str, tuple[float, float, float, float]] = {}
    letters = ("a", "b", "c", "d", "e", "f")
    for idx, letter in enumerate(letters):
        row, col = divmod(idx, 3)
        # row 0 is top
        x_in = left_in + col * (panel_w_in + col_gap_in)
        y_in = bottom_in + (1 - row) * (panel_h_in + row_gap_in)
        rects[letter] = (
            x_in * inv_w,
            y_in * inv_h,
            panel_w_in * inv_w,
            panel_h_in * inv_h,
        )
    return rects


def sensitivity_r_noise_overview(
    *,
    noisy_pops: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]],
    agg_metrics: pd.DataFrame,
    path_stem: Path,
    sigma_values: tuple[float, ...] | list[float],
) -> list[Path]:
    """2×3 figure: (a–c) $R'$–$Q$ scatters; (d–f) V / τ / Jaccard bars + 95\\% CI."""
    sigma_values = tuple(float(s) for s in sigma_values)
    if len(sigma_values) != 3:
        raise ValueError("sensitivity_r_noise_overview expects three σ_R values.")

    panel_fs = max(8, int(round(FONT_SIZE * PLOT_FONT_SCALE / 1.45)))
    apply_paper_style(font_size=panel_fs)

    left_in, right_in = 0.72, 0.28
    top_in, bottom_in = 0.42, 0.55
    col_gap_in, row_gap_in = 0.48, 0.72
    panel_w_in = panel_h_in = 2.20
    fig_w = left_in + 3.0 * panel_w_in + 2.0 * col_gap_in + right_in
    fig_h = bottom_in + 2.0 * panel_h_in + row_gap_in + top_in
    fig = plt.figure(figsize=(fig_w, fig_h))
    rects = _sensitivity_2x3_layout(
        fig,
        left_in=left_in,
        right_in=right_in,
        top_in=top_in,
        bottom_in=bottom_in,
        col_gap_in=col_gap_in,
        row_gap_in=row_gap_in,
        panel_w_in=panel_w_in,
        panel_h_in=panel_h_in,
    )
    letter_fs = panel_fs + 1
    scatter_y_bottom, scatter_y_top = -0.02, 1.22

    for letter, sigma_r in zip(("a", "b", "c"), sigma_values):
        ax = fig.add_axes(rects[letter])
        R, Q, is_veto = noisy_pops[float(sigma_r)]
        _draw_rq_scatter(ax, R, Q, is_veto)
        style_square_axes_frame(ax)
        ax.set_title(rf"$\sigma_R={sigma_r:.2f}$", pad=4)
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        if letter == "a":
            _finish_panel_a_legend(
                ax,
                y_top=scatter_y_top,
                legend_y=0.98,
            )
        else:
            ax.set_ylim(scatter_y_bottom, scatter_y_top)
        add_panel_letter(ax, letter, font_size=letter_fs)

    bar_specs = (
        ("d", "policy_violation_rate", r"Policy violation rate $V$", True),
        ("e", "kendall_tau", r"Kendall $\tau$ vs $\sigma_R=0.0$", False),
        ("f", "jaccard_top_k", r"Jaccard Top-$K$ vs $\sigma_R=0.0$", True),
    )
    config_specs = agreement_operator_specs()
    for letter, metric, panel_title, clip_nonneg in bar_specs:
        ax = fig.add_axes(rects[letter])
        ylim = _sensitivity_bar_panel_ylim(agg_metrics, metric, sigma_values)
        plot_agreement_config_bars(
            ax,
            agg_metrics,
            x_col="sigma_r",
            group_col="config_id",
            y_col=f"{metric}_mean",
            ci_low_col=f"{metric}_ci_low",
            ci_high_col=f"{metric}_ci_high",
            x_values=sigma_values,
            specs=config_specs,
            ylabel=panel_title,
            title=panel_title,
            xlabel=r"Predictive noise $\sigma_R$",
            ylim=ylim,
            clip_error_nonneg=clip_nonneg,
            show_legend=(letter == "d"),
            reference_y=0.0 if metric == "kendall_tau" else None,
            family_gaps=True,
            legend_anchor_y=0.94,
        )
        style_square_axes_frame(ax)
        add_panel_letter(ax, letter, font_size=letter_fs)

    return save_figure(fig, path_stem, bbox_inches=None)


def _agreement_diverging_cmap() -> LinearSegmentedColormap:
    """Okabe–Ito map for Kendall $\\tau$: blue (−1) → yellow (0) → orange (+1)."""
    low = OKABE_ITO[4]  # #0072B2 blue  (near -1)
    mid = OKABE_ITO[3]  # #F0E442 yellow (near 0)
    high = OKABE_ITO[5]  # #D55E00 orange (near +1)
    cmap = LinearSegmentedColormap.from_list(
        "agreement_kendall_okabe_ito",
        [(0.0, low), (0.5, mid), (1.0, high)],
        N=256,
    )
    cmap.set_bad(color=(1.0, 1.0, 1.0, 0.0))
    return cmap


def _agreement_jaccard_cmap() -> LinearSegmentedColormap:
    """Two-color Okabe–Ito map for Jaccard: yellow (near 0) → orange (near 1)."""
    low = OKABE_ITO[3]  # #F0E442 yellow
    high = OKABE_ITO[5]  # #D55E00 orange
    cmap = LinearSegmentedColormap.from_list(
        "agreement_jaccard_okabe_ito",
        [(0.0, low), (1.0, high)],
        N=256,
    )
    cmap.set_bad(color=(1.0, 1.0, 1.0, 0.0))
    return cmap


def _pair_matrix(
    agg_pairs: pd.DataFrame,
    *,
    metric: str,
    config_ids: tuple[str, ...],
    sigma_r: float | None = None,
) -> np.ndarray:
    """Build mean-agreement matrix; off-diagonal from ``agg_pairs`` only."""
    n = len(config_ids)
    mat = np.full((n, n), np.nan, dtype=float)
    sub = agg_pairs
    if sigma_r is not None:
        sub = sub[np.isclose(sub["sigma_r"].to_numpy(), float(sigma_r))]
    col = f"{metric}_mean"
    index = {cid: i for i, cid in enumerate(config_ids)}
    for _, row in sub.iterrows():
        a = str(row["operator_a"])
        b = str(row["operator_b"])
        if a not in index or b not in index:
            continue
        i, j = index[a], index[b]
        if i == j:
            continue
        mat[i, j] = float(row[col])
    return mat


def _plot_triangular_agreement_heatmap(
    ax: Axes,
    matrix: np.ndarray,
    *,
    tick_labels: tuple[str, ...] | list[str],
    ytick_labels: tuple[str, ...] | list[str] | None = None,
    title: str = "",
    vmin: float = 0.0,
    vmax: float = 1.0,
    cmap: LinearSegmentedColormap | str | None = None,
    annotate: bool = True,
    fontsize: float = 7.0,
    cell_fontsize: float | None = None,
    face_alpha: float = 0.55,
    mask_mode: str = "lower",
):
    """Lower-triangle heatmap without diagonal (stepped / escalonada)."""
    n_rows, n_cols = matrix.shape
    if ytick_labels is None:
        ytick_labels = tick_labels
    if len(tick_labels) != n_cols or len(ytick_labels) != n_rows:
        raise ValueError("tick label counts must match matrix shape.")
    display = np.array(matrix, dtype=float, copy=True)
    if mask_mode == "lower":
        mask = np.triu(np.ones((n_rows, n_cols), dtype=bool), k=0)
    elif mask_mode == "finite":
        mask = ~np.isfinite(display)
    else:
        raise ValueError(f"Unknown mask_mode: {mask_mode!r}")
    masked = np.ma.array(display, mask=mask)
    if cmap is None:
        cmap_obj = _agreement_diverging_cmap()
    elif isinstance(cmap, str):
        cmap_obj = plt.get_cmap(cmap).copy()
        cmap_obj.set_bad(color=(1.0, 1.0, 1.0, 0.0))
    else:
        cmap_obj = cmap.copy() if hasattr(cmap, "copy") else cmap
        cmap_obj.set_bad(color=(1.0, 1.0, 1.0, 0.0))
    im = ax.imshow(
        masked,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap_obj,
        origin="upper",
        alpha=float(face_alpha),
    )
    ax.set_facecolor("none")
    ax.set_xticks(range(n_cols))
    ax.set_yticks(range(n_rows))
    ax.set_xticklabels(list(tick_labels), rotation=45, ha="right", fontsize=fontsize)
    ax.set_yticklabels(list(ytick_labels), fontsize=fontsize)
    ax.set_title(title, pad=4)
    ann_fs = float(cell_fontsize) if cell_fontsize is not None else float(fontsize) + 1.5
    if annotate:
        for i in range(n_rows):
            for j in range(n_cols):
                if not np.isfinite(display[i, j]):
                    continue
                if mask_mode == "lower" and i <= j:
                    continue
                val = float(display[i, j])
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=ann_fs,
                    fontweight="medium",
                )
    ax.tick_params(length=0)
    # Only principal axes (left / bottom); no enclosing frame.
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")
    return im


def _add_inpanel_agreement_colorbar(
    ax: Axes,
    im,
    *,
    vmin: float,
    vmax: float,
    n_ticks: int = 5,
    tick_fontsize: float = 6.0,
    x_data: float | None = None,
    y_top_data: float | None = None,
    y_bottom_data: float | None = None,
) -> None:
    """Vertical in-panel colorbar anchored at $(x,y)$ data coordinates."""
    if x_data is None or y_top_data is None or y_bottom_data is None:
        _, _, x_default, y_top_default, y_bottom_default = _agreement_heatmap_crop_slices()
        if x_data is None:
            x_data = x_default
        if y_top_data is None:
            y_top_data = y_top_default
        if y_bottom_data is None:
            y_bottom_data = y_bottom_default

    def _data_to_axes(x: float, y: float) -> tuple[float, float]:
        display = ax.transData.transform((x, y))
        return ax.transAxes.inverted().transform(display)

    x_center, y_top = _data_to_axes(float(x_data), float(y_top_data))
    _, y_bottom = _data_to_axes(float(x_data), float(y_bottom_data))

    bar_w = 0.038
    x_left = x_center - bar_w / 2.0
    bar_h = max(y_top - y_bottom, bar_w * 2.0)
    cax = ax.inset_axes([x_left, y_bottom, bar_w, bar_h], transform=ax.transAxes)
    ticks = np.linspace(float(vmin), float(vmax), int(n_ticks))
    cb = ax.figure.colorbar(im, cax=cax, ticks=ticks)
    cb.ax.tick_params(labelsize=tick_fontsize, length=2.0, pad=1.0)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("black")


def sensitivity_operator_agreement(
    *,
    agg_pairs: pd.DataFrame,
    path_stem: Path,
    sigma_values: tuple[float, ...] | list[float],
) -> list[Path]:
    """2×3 figure: triangular Kendall / Jaccard agreement among 7 configs."""
    sigma_values = tuple(float(s) for s in sigma_values)
    if len(sigma_values) != 3:
        raise ValueError("sensitivity_operator_agreement expects three σ_R values.")

    specs = agreement_operator_specs()
    _, _, x_cb, y_cb_top, y_cb_bottom = _agreement_heatmap_crop_slices(specs)

    panel_fs = max(7, int(round(FONT_SIZE * PLOT_FONT_SCALE / 1.55)))
    apply_paper_style(font_size=panel_fs)

    left_in, right_in = 0.85, 0.58
    top_in, bottom_in = 0.42, 0.72
    col_gap_in, row_gap_in = 0.48, 0.72
    panel_w_in = panel_h_in = 2.35
    fig_w = left_in + 3.0 * panel_w_in + 2.0 * col_gap_in + right_in
    fig_h = bottom_in + 2.0 * panel_h_in + row_gap_in + top_in
    fig = plt.figure(figsize=(fig_w, fig_h))
    rects = _sensitivity_2x3_layout(
        fig,
        left_in=left_in,
        right_in=right_in,
        top_in=top_in,
        bottom_in=bottom_in,
        col_gap_in=col_gap_in,
        row_gap_in=row_gap_in,
        panel_w_in=panel_w_in,
        panel_h_in=panel_h_in,
    )
    letter_fs = panel_fs + 1

    for letter, sigma_r in zip(("a", "b", "c"), sigma_values):
        ax = fig.add_axes(rects[letter])
        im_tau = _plot_cropped_operator_agreement_heatmap(
            ax,
            agg_pairs,
            metric="kendall_tau",
            sigma_r=sigma_r,
            title=rf"Kendall $\tau$, $\sigma_R={sigma_r:.2f}$",
            vmin=-1.0,
            vmax=1.0,
            cell_fontsize=8.5,
        )
        if letter == "c":
            _add_inpanel_agreement_colorbar(
                ax,
                im_tau,
                vmin=-1.0,
                vmax=1.0,
                n_ticks=5,
                tick_fontsize=5.5,
                x_data=x_cb,
                y_top_data=y_cb_top,
                y_bottom_data=y_cb_bottom,
            )
        add_panel_letter(ax, letter, font_size=letter_fs)

    for letter, sigma_r in zip(("d", "e", "f"), sigma_values):
        ax = fig.add_axes(rects[letter])
        im_jac = _plot_cropped_operator_agreement_heatmap(
            ax,
            agg_pairs,
            metric="jaccard_top_k",
            sigma_r=sigma_r,
            title=rf"Jaccard, $\sigma_R={sigma_r:.2f}$",
            vmin=0.0,
            vmax=1.0,
            cmap=_agreement_jaccard_cmap(),
            cell_fontsize=8.5,
        )
        if letter == "f":
            _add_inpanel_agreement_colorbar(
                ax,
                im_jac,
                vmin=0.0,
                vmax=1.0,
                n_ticks=5,
                tick_fontsize=5.5,
                x_data=x_cb,
                y_top_data=y_cb_top,
                y_bottom_data=y_cb_bottom,
            )
        add_panel_letter(ax, letter, font_size=letter_fs)

    return save_figure(fig, path_stem, bbox_inches=None)
