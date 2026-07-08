"""Figures for the real attrition example (100-case CADEMAS-ML cohort)."""

from __future__ import annotations

import os
from pathlib import Path

# Writable cache when ~/.matplotlib is not available (CI/sandbox).
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text
from bumplot.bezier import bezier_curve
from matplotlib.path import Path
from matplotlib.ticker import FixedLocator

from plot_style import apply_helvetica_style, style_axes_frame
from real_example import (
    BASELINE_KEY,
    BUMP_LABELS,
    BUMP_TRACKS,
    OPERATORS,
    TOP_K,
    rank_displacement,
    top_k_union,
)
from systemic_config import FIGURES_DIR, PLOT_FONT_SCALE, SINGLE_PANEL_FIGSIZE

PLANE_SIDE = SINGLE_PANEL_FIGSIZE[1] * 1.5
PLANE_FIGSIZE = (PLANE_SIDE, PLANE_SIDE)
BUMP_FIGSIZE = (SINGLE_PANEL_FIGSIZE[0] * 1.5, SINGLE_PANEL_FIGSIZE[1] * 2.5)
BUMP_X_MARGIN_LEFT = 1.0
BUMP_X_MARGIN_RIGHT = 0.25
BUMP_SUBPLOT_LEFT = 0.14
BUMP_SUBPLOT_RIGHT = 0.97
BUMP_Y_TICK_STEP = 1
LABEL_FONT = 9
PLANE_Y_TOP = 1.15
PLANE_Y_TICK_MAX = 1.0
COHORT_SCATTER_SIZE = 36
COHORT_COLOR = "0.62"
HIGHLIGHT_EDGE = "white"
# Non-baseline bump columns where outsiders can enter the Top-K tier.
GHOST_BUMP_KEYS = tuple(k for k in BUMP_TRACKS if k != BASELINE_KEY)
GHOST_MARKER_SHAPE = "*"
GHOST_MARKER_COLOR = "0.50"
HIGHLIGHT_MARKER_SIZE = 100
GHOST_MARKER_SIZE = 120
BUMP_ANNOT_FONT_SCALE = 0.92
GHOST_LABEL_BBOX = {
    "boxstyle": "round,pad=0.28",
    "facecolor": "white",
    "edgecolor": GHOST_MARKER_COLOR,
    "alpha": 0.8,
    "linewidth": 0.0,
}
PLANE_LABEL_Y_OFFSET = 0.05
PLANE_LABEL_SE_OFFSET = (0.038, -0.038)
PLANE_LABEL_NW_OFFSET = (-0.038, 0.038)
PLANE_LABEL_NE_OFFSET = (0.038, 0.038)
PLANE_LABEL_SW_OFFSET = (-0.038, -0.038)
PLANE_LABEL_S_OFFSET = (0.0, -0.088)
PLANE_LABEL_N_OFFSET = (0.0, 0.055)
PLANE_LABEL_W_OFFSET = (-0.048, 0.0)
PLANE_LABEL_PROXIMITY = 0.05
PLANE_LABEL_OVERRIDES: dict[str, str] = {
    "Andrew Wood": "southeast",
    "Eleanor Foster": "northwest",
    "Laura Baker": "southeast",
    "Alice Brown": "southeast",
    "Camila Phillips": "southeast",
    "Caroline Bennett": "northeast",
    "Andrew Ross": "southeast",
    "Julia Smith": "northwest",
    "Caroline Roberts": "northwest",
    "Dominic Collins": "northwest",
    "Lucy Scott": "southeast",
    "Jessica Wilson": "southeast",
    "Madison Foster": "southwest",
    "Christian Bennett": "southwest",
    "Amelia Davis": "northwest",
    "Noah Murphy": "northeast",
    "Eleanor Foster": "northeast",
    "Leah Cooper": "southwest",
    "Caroline Adams": "northwest",
    "Claire Bailey": "northwest",
    "Hunter Adams": "northwest",
}
PLANE_LABEL_OFFSET_OVERRIDES: dict[str, tuple[float, float]] = {
    "Andrew Wood": (-0.028, 0.1),
    "Camila Phillips": (0.022, -0.2),
    "Andrew Ross": (-0.01, -0.06),
    "Julia Smith": (0.03, 0.1),
    "Caroline Roberts": (-0.038, 0.28),
    "Dominic Collins": (-0.02, 0.035),
    "Alice Brown": (0.045, -0.12),
    "Lucy Scott": (-0.03, -0.05),
    "Jessica Wilson": (0.028, -0.24),
    "Madison Foster": (0.01, -0.15),
    "Caroline Bennett": (0.0, 0.1),
    "Christian Bennett": (0.01, -0.1),
    "Amelia Davis": (-0.028, 0.1),
    "Noah Murphy": (0.028, 0.1),
    "Eleanor Foster": (-0.1, 0.08),
    "Laura Baker": (0.02, 0.02),
    "Caroline Adams": (0.13, 0.12),
    "Claire Bailey": (-0.06, 0.08),
    "Hunter Adams": (-0.07, 0.2),
}
PLANE_ADJUST_TEXT = {
    "expand": (1.05, 2.0),
    "force_text": (0.45, 1.05),
    "force_static": (0.4, 0.8),
    "force_pull": (0.02, 0.02),
    "force_explode": (0.18, 1.0),
    "only_move": {"text": "y", "static": "xy", "explode": "y", "pull": "y"},
    "iter_lim": 2000,
    "prevent_crossings": True,
    "min_arrow_len": 10,
}
PLANE_ARROW_PROPS = {
    "arrowstyle": "-",
    "color": GHOST_MARKER_COLOR,
    "lw": 0.9,
    "shrinkA": 4,
    "shrinkB": 4,
    "connectionstyle": "arc3,rad=0",
}


def _ghost_case_ids(df: pd.DataFrame, baseline_ids: list[str]) -> list[str]:
    """Outsiders entering Top-K under any non-baseline bump-chart setting."""
    baseline_set = set(baseline_ids)
    ghosts: list[str] = []
    seen: set[str] = set()
    for key in GHOST_BUMP_KEYS:
        for slot in range(1, TOP_K + 1):
            matches = df[df[f"rank_{key}"] == slot]
            if matches.empty:
                continue
            cid = matches.iloc[0]["case_id"]
            if cid not in baseline_set and cid not in seen:
                seen.add(cid)
                ghosts.append(cid)
    return ghosts


def _plot_ghost_scatter(
    ax: plt.Axes,
    x,
    y,
    *,
    color: str | None = None,
    size: float | None = None,
    zorder: int = 3,
) -> None:
    """Star marker for outsider (ghost) cases."""
    ax.scatter(
        x,
        y,
        s=size or GHOST_MARKER_SIZE,
        marker=GHOST_MARKER_SHAPE,
        c=color or GHOST_MARKER_COLOR,
        edgecolors="white",
        linewidths=0.6,
        zorder=zorder,
        alpha=0.95,
    )


def _plot_baseline_ghost_markers(
    ax: plt.Axes,
    df: pd.DataFrame,
    x_positions: np.ndarray,
    baseline_ids: list[str],
    font_size: float,
) -> None:
    """Gray star markers for outsiders in Top-K on non-baseline bump-chart columns."""
    baseline_set = set(baseline_ids)
    for key in GHOST_BUMP_KEYS:
        col_idx = BUMP_TRACKS.index(key)
        x = x_positions[col_idx]

        for slot in range(1, TOP_K + 1):
            tied = df[df[f"rank_{key}"] == slot]
            if tied.empty:
                continue

            outsiders = tied[~tied["case_id"].isin(baseline_set)]
            baseline_tied = tied[tied["case_id"].isin(baseline_set)]
            if outsiders.empty:
                continue

            outsider = outsiders.iloc[0]
            baseline_rank = int(outsider[f"rank_{BASELINE_KEY}"])
            case_name = outsider["case_id"]

            if not baseline_tied.empty:
                # Tie with a baseline trajectory: star overlays the baseline circle.
                for _ in baseline_tied.itertuples():
                    _plot_ghost_scatter(ax, [x], [slot], zorder=5)
            else:
                _plot_ghost_scatter(ax, [x], [slot], zorder=5)

            ax.annotate(
                f"{baseline_rank} — {case_name}",
                (x, slot),
                textcoords="offset points",
                xytext=(-6, 0),
                fontsize=font_size,
                ha="right",
                va="center",
                color=GHOST_MARKER_COLOR,
                bbox=GHOST_LABEL_BBOX,
                zorder=6,
                clip_on=False,
            )


def _case_colors(case_ids: list[str]) -> dict[str, str]:
    cmap = plt.get_cmap("tab20")
    return {cid: cmap(i % 20) for i, cid in enumerate(case_ids)}


def _union_styles(df: pd.DataFrame) -> tuple[list[str], dict[str, str]]:
    """Shared Top-K union cases and colors (aligned with predictive score vs. context plane)."""
    union_ids = top_k_union(df, k=TOP_K)
    return union_ids, _case_colors(union_ids)


def _baseline_top_k_ids(df: pd.DataFrame, k: int = TOP_K) -> list[str]:
    """Top-K case IDs under baseline $A_L$ ($\\lambda = 0.5$)."""
    top = df.nsmallest(k, f"rank_{BASELINE_KEY}")
    return top["case_id"].tolist()


def _plane_yticks() -> list[float]:
    step = 0.2
    ticks = np.arange(0.0, PLANE_Y_TICK_MAX + step / 2, step)
    return [float(t) for t in ticks if t <= PLANE_Y_TICK_MAX + 1e-9]


def _assign_alternating_label_sides(xs: list[float], ys: list[float]) -> list[int]:
    """Assign +1 (above) / -1 (below), alternating within each nearby cluster."""
    n = len(xs)
    if n == 0:
        return []
    prox2 = PLANE_LABEL_PROXIMITY**2
    adj: list[list[int]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            if dx * dx + dy * dy <= prox2:
                adj[i].append(j)
                adj[j].append(i)

    sides = [1] * n
    seen: set[int] = set()
    for start in range(n):
        if start in seen:
            continue
        component: list[int] = []
        stack = [start]
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            component.append(u)
            stack.extend(v for v in adj[u] if v not in seen)

        component.sort(key=lambda i: (xs[i], ys[i]))
        for k, idx in enumerate(component):
            sides[idx] = 1 if k % 2 == 0 else -1

    return sides


def _annotate_plane_label_fixed(
    ax: plt.Axes,
    name: str,
    x: float,
    y: float,
    color: str,
    weight: str,
    font_size: float,
    placement: str,
) -> None:
    """Fixed label placement outside adjustText (diagonal quadrants)."""
    if placement == "southeast":
        dq, dr = PLANE_LABEL_OFFSET_OVERRIDES.get(name, PLANE_LABEL_SE_OFFSET)
        ax.annotate(
            name,
            (x, y),
            xytext=(x + dq, y + dr),
            textcoords="data",
            fontsize=font_size,
            ha="left",
            va="top",
            color=color,
            weight=weight,
            arrowprops=PLANE_ARROW_PROPS,
            clip_on=False,
        )
        return
    if placement == "northwest":
        dq, dr = PLANE_LABEL_OFFSET_OVERRIDES.get(name, PLANE_LABEL_NW_OFFSET)
        ax.annotate(
            name,
            (x, y),
            xytext=(x + dq, y + dr),
            textcoords="data",
            fontsize=font_size,
            ha="right",
            va="bottom",
            color=color,
            weight=weight,
            arrowprops=PLANE_ARROW_PROPS,
            clip_on=False,
        )
        return
    if placement == "northeast":
        dq, dr = PLANE_LABEL_OFFSET_OVERRIDES.get(name, PLANE_LABEL_NE_OFFSET)
        ax.annotate(
            name,
            (x, y),
            xytext=(x + dq, y + dr),
            textcoords="data",
            fontsize=font_size,
            ha="left",
            va="bottom",
            color=color,
            weight=weight,
            arrowprops=PLANE_ARROW_PROPS,
            clip_on=False,
        )
        return
    if placement == "southwest":
        dq, dr = PLANE_LABEL_OFFSET_OVERRIDES.get(name, PLANE_LABEL_SW_OFFSET)
        ax.annotate(
            name,
            (x, y),
            xytext=(x + dq, y + dr),
            textcoords="data",
            fontsize=font_size,
            ha="right",
            va="top",
            color=color,
            weight=weight,
            arrowprops=PLANE_ARROW_PROPS,
            clip_on=False,
        )
        return
    if placement == "south":
        dq, dr = PLANE_LABEL_S_OFFSET
        ax.annotate(
            name,
            (x, y),
            xytext=(x + dq, y + dr),
            textcoords="data",
            fontsize=font_size,
            ha="center",
            va="top",
            color=color,
            weight=weight,
            arrowprops=PLANE_ARROW_PROPS,
            clip_on=False,
        )
        return
    if placement == "north":
        dq, dr = PLANE_LABEL_N_OFFSET
        ax.annotate(
            name,
            (x, y),
            xytext=(x + dq, y + dr),
            textcoords="data",
            fontsize=font_size,
            ha="center",
            va="bottom",
            color=color,
            weight=weight,
            arrowprops=PLANE_ARROW_PROPS,
            clip_on=False,
        )
        return
    if placement == "west":
        dq, dr = PLANE_LABEL_OFFSET_OVERRIDES.get(name, PLANE_LABEL_W_OFFSET)
        ax.annotate(
            name,
            (x, y),
            xytext=(x + dq, y + dr),
            textcoords="data",
            fontsize=font_size,
            ha="right",
            va="center",
            color=color,
            weight=weight,
            arrowprops=PLANE_ARROW_PROPS,
            clip_on=False,
        )
        return
    raise ValueError(f"Unknown plane label placement: {placement}")


def plot_risk_context_plane(
    df: pd.DataFrame,
    output_path: Path | None = None,
    baseline_ids: list[str] | None = None,
    baseline_colors: dict[str, str] | None = None,
    ghost_ids: list[str] | None = None,
) -> Path:
    """Scatter of predictive score vs context score for the full cohort."""
    apply_helvetica_style(font_scale=PLOT_FONT_SCALE)
    text_size = plt.rcParams["ytick.labelsize"]
    label_size = text_size * BUMP_ANNOT_FONT_SCALE
    fig, ax = plt.subplots(figsize=PLANE_FIGSIZE)
    style_axes_frame(ax)

    if baseline_ids is None:
        baseline_ids = _baseline_top_k_ids(df)
    if baseline_colors is None:
        baseline_colors = _case_colors(baseline_ids)
    if ghost_ids is None:
        ghost_ids = _ghost_case_ids(df, baseline_ids)

    x_max = max(0.78, float(df["Q"].max()) * 1.05)

    ax.scatter(
        df["Q"],
        df["R"],
        s=COHORT_SCATTER_SIZE,
        c=COHORT_COLOR,
        edgecolors="white",
        linewidths=0.4,
        zorder=2,
        alpha=0.5,
    )

    texts = []
    xs: list[float] = []
    ys: list[float] = []
    ghost_set = set(ghost_ids)
    highlight_specs: list[tuple[str, str, str]] = [
        (cid, baseline_colors[cid], "bold") for cid in baseline_ids
    ] + [(cid, GHOST_MARKER_COLOR, "bold") for cid in ghost_ids]

    pending: list[tuple[str, float, float, str, str]] = []
    for name, color, weight in highlight_specs:
        row = df.loc[df["case_id"] == name]
        if row.empty:
            continue
        x, y = float(row["Q"].iloc[0]), float(row["R"].iloc[0])
        if name in ghost_set:
            _plot_ghost_scatter(ax, [x], [y], color=GHOST_MARKER_COLOR, zorder=5)
        else:
            ax.scatter(
                [x],
                [y],
                s=HIGHLIGHT_MARKER_SIZE,
                c=[color],
                edgecolors=HIGHLIGHT_EDGE,
                linewidths=0.8,
                zorder=5,
            )
        pending.append((name, x, y, color, weight))

    xs = [p[1] for p in pending]
    ys = [p[2] for p in pending]
    label_sides = _assign_alternating_label_sides(xs, ys)

    xs = []
    ys = []
    for (name, x, y, color, weight), side in zip(pending, label_sides):
        override = PLANE_LABEL_OVERRIDES.get(name)
        if override is not None:
            _annotate_plane_label_fixed(ax, name, x, y, color, weight, label_size, override)
            continue

        label_y = y + side * PLANE_LABEL_Y_OFFSET
        xs.append(x)
        ys.append(y)
        texts.append(
            ax.text(
                x,
                label_y,
                name,
                fontsize=label_size,
                color=color,
                weight=weight,
                ha="center",
                va="bottom" if side > 0 else "top",
                clip_on=True,
            )
        )

    ax.set_xlabel(r"Context score $Q_i$ (Digital Transformation)")
    ax.set_ylabel(r"Predictive score $R_i$")
    ax.set_title("Predictive score vs. context plane", fontweight="bold")
    ax.set_xlim(-0.02, x_max)
    ax.set_ylim(-0.02, PLANE_Y_TOP)
    ax.yaxis.set_major_locator(FixedLocator(_plane_yticks()))
    ax.set_box_aspect(1)

    if texts:
        adjust_text(
            texts,
            x=xs,
            y=ys,
            target_x=xs,
            target_y=ys,
            ax=ax,
            arrowprops=PLANE_ARROW_PROPS,
            **PLANE_ADJUST_TEXT,
        )

    fig.tight_layout()
    out = output_path or (FIGURES_DIR / "real_example_plane.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_rank_bump_chart(
    df: pd.DataFrame,
    output_path: Path | None = None,
    case_ids: list[str] | None = None,
    colors: dict[str, str] | None = None,
) -> Path:
    """Bump chart of rank trajectories for the baseline Top-K tier ($A_L$, $\\lambda=0.5$)."""
    apply_helvetica_style(font_scale=PLOT_FONT_SCALE)
    text_size = plt.rcParams["ytick.labelsize"]
    label_size = text_size * BUMP_ANNOT_FONT_SCALE
    fig, ax = plt.subplots(figsize=BUMP_FIGSIZE)
    style_axes_frame(ax)
    ax.tick_params(axis="both", labelsize=text_size)

    if case_ids is None:
        case_ids = _baseline_top_k_ids(df)
    if colors is None:
        colors = _case_colors(case_ids)

    x_labels = [BUMP_LABELS[key] for key in BUMP_TRACKS]
    x_positions = np.arange(len(BUMP_TRACKS), dtype=float)

    rank_by_case: dict[str, list[int]] = {}
    max_rank = 0
    for cid in case_ids:
        ranks = [int(df.loc[df["case_id"] == cid, f"rank_{key}"].iloc[0]) for key in BUMP_TRACKS]
        rank_by_case[cid] = ranks
        max_rank = max(max_rank, max(ranks))

    for cid in case_ids:
        ranks = np.array(rank_by_case[cid], dtype=float)
        color = colors[cid]
        disp = rank_displacement(df, cid)
        weight = "bold" if disp >= 3 else "normal"
        lw = 2.6 if disp >= 3 else 1.6

        vertices, codes = bezier_curve(x_positions, ranks, force=0.55)
        path = Path(vertices, codes)
        ax.add_patch(
            patches.PathPatch(
                path,
                facecolor="none",
                edgecolor=color,
                linewidth=lw,
                alpha=0.92,
                zorder=2,
            )
        )
        ax.scatter(
            x_positions,
            ranks,
            s=HIGHLIGHT_MARKER_SIZE,
            c=[color],
            edgecolors="white",
            linewidths=0.8,
            zorder=4,
        )

        ax.annotate(
            cid,
            (x_positions[0], ranks[0]),
            textcoords="offset points",
            xytext=(-5, 0),
            fontsize=label_size,
            ha="right",
            va="center",
            color=color,
            weight=weight,
            clip_on=False,
        )

    _plot_baseline_ghost_markers(ax, df, x_positions, case_ids, label_size)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels)
    y_upper = max(max_rank + 1, TOP_K + 1)
    ax.set_ylim(0, y_upper)
    ax.invert_yaxis()
    ax.yaxis.set_major_locator(
        FixedLocator(list(range(1, y_upper + 1, BUMP_Y_TICK_STEP)))
    )
    for tick in ax.get_yticklabels():
        try:
            rank = int(float(tick.get_text()))
        except ValueError:
            continue
        if 1 <= rank <= TOP_K:
            tick.set_fontweight("bold")
    ax.set_ylabel("Rank (1 = highest priority)", fontsize=text_size)
    ax.set_title(
        rf"Rank trajectories (Top-{TOP_K} under $A_L$, $\lambda=0.5$)",
        fontweight="bold",
        fontsize=text_size,
    )
    ax.set_xlim(
        -BUMP_X_MARGIN_LEFT,
        len(BUMP_TRACKS) - 1 + BUMP_X_MARGIN_RIGHT,
    )

    fig.subplots_adjust(
        left=BUMP_SUBPLOT_LEFT,
        right=BUMP_SUBPLOT_RIGHT,
        bottom=0.24,
    )
    out = output_path or (FIGURES_DIR / "real_example_bump.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_real_example(df: pd.DataFrame) -> tuple[Path, Path]:
    """Generate both real-example figures."""
    baseline_ids = _baseline_top_k_ids(df)
    baseline_colors = _case_colors(baseline_ids)
    ghost_ids = _ghost_case_ids(df, baseline_ids)
    plane = plot_risk_context_plane(
        df,
        baseline_ids=baseline_ids,
        baseline_colors=baseline_colors,
        ghost_ids=ghost_ids,
    )
    bump = plot_rank_bump_chart(df, case_ids=baseline_ids, colors=baseline_colors)
    return plane, bump


def results_table_latex(df: pd.DataFrame, k: int = TOP_K) -> str:
    """Generate LaTeX tabular rows for baseline Top-K plus ghost entrants."""
    baseline_ids = _baseline_top_k_ids(df)
    ghost_ids = set(_ghost_case_ids(df, baseline_ids))
    union_ids = top_k_union(df, k=k)
    score_cols = [f"P_{key}" for key in OPERATORS]
    rank_cols = [f"rank_{key}" for key in OPERATORS]
    cols = ["case_id", "R", "Q", "attrition"] + score_cols + rank_cols
    lines = []
    for cid in union_ids:
        row = df.loc[df["case_id"] == cid, cols].iloc[0]
        name = row["case_id"]
        attr = "Yes" if str(row["attrition"]).lower() in ("yes", "true", "1") else "No"
        ghost = "Yes" if name in ghost_ids else "No"
        scores = " & ".join(f"{row[f'P_{key}']:.4f}" for key in OPERATORS)
        ranks = " & ".join(str(int(row[f"rank_{key}"])) for key in OPERATORS)
        lines.append(
            f"{name} & {row['R']:.4f} & {row['Q']:.4f} & {scores} & {ranks} & {ghost} & {attr} \\\\"
        )
    return "\n".join(lines)
