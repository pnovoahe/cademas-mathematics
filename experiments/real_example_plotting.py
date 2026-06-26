"""Figure for the real attrition example (CADEMAS-ML cohort)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_style import OPERATOR_STYLE, apply_helvetica_style, style_axes_frame
from real_example import LAMBDA, OPERATOR_LABELS, OPERATORS, TOP_K, summarize_example
from systemic_config import FIGURES_DIR, PLOT_FONT_SCALE

LEGEND_KW = {
    "frameon": True,
    "facecolor": "white",
    "edgecolor": "0.75",
    "framealpha": 1.0,
}

ANNOTATE_CASES = {
    "Evelyn Taylor": (8, 8),
    "Noah Lewis": (8, -12),
    "Lucas Wright": (8, 8),
}


def plot_real_example(df: pd.DataFrame, output_path: Path | None = None) -> Path:
    apply_helvetica_style(font_scale=PLOT_FONT_SCALE)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), gridspec_kw={"width_ratios": [1.35, 1]})

    ax_scatter, ax_bars = axes
    style_axes_frame(ax_scatter)
    style_axes_frame(ax_bars)

    # Panel (a): R vs Q scatter with highlighted cases
    ax_scatter.scatter(
        df["Q"],
        df["R"],
        s=55,
        c="0.55",
        edgecolors="white",
        linewidths=0.6,
        zorder=2,
        label="Cohort (n=20)",
    )
    for name, (dx, dy) in ANNOTATE_CASES.items():
        row = df.loc[df["case_id"] == name]
        if row.empty:
            continue
        x, y = float(row["Q"].iloc[0]), float(row["R"].iloc[0])
        ax_scatter.scatter([x], [y], s=90, c="#C31F5C", edgecolors="black", linewidths=0.6, zorder=4)
        ax_scatter.annotate(
            name.replace(" ", "\n"),
            (x, y),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=8,
            ha="left",
        )

    ax_scatter.set_xlabel(r"Context score $Q_i$ (Digital Transformation)")
    ax_scatter.set_ylabel(r"Cooperative risk $R_i$")
    ax_scatter.set_title(rf"(a) Risk--context plane ($\lambda={LAMBDA}$)")
    ax_scatter.set_xlim(-0.02, 0.78)
    ax_scatter.set_ylim(-0.02, 1.02)
    ax_scatter.legend(loc="upper left", **LEGEND_KW)

    # Panel (b): Top-K prioritization scores by operator
    x = np.arange(TOP_K)
    width = 0.24
    offsets = np.linspace(-width, width, len(OPERATORS))

    for offset, op in zip(offsets, OPERATORS):
        top = df.nsmallest(TOP_K, f"rank_{op}").sort_values(f"P_{op}", ascending=False)
        scores = top[f"P_{op}"].values
        labels = [n.split()[0] for n in top["case_id"]]
        color = OPERATOR_STYLE[op]["color"]
        bars = ax_bars.bar(
            x + offset,
            scores,
            width=width * 0.92,
            color=color,
            label=OPERATOR_LABELS[op],
            edgecolor="white",
            linewidth=0.5,
        )
        for bar, label in zip(bars, labels):
            ax_bars.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015,
                label,
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=0,
            )

    ax_bars.set_xticks(x)
    ax_bars.set_xticklabels([f"Rank {i + 1}" for i in range(TOP_K)])
    ax_bars.set_ylabel(r"Prioritization score $P_i$")
    ax_bars.set_title(rf"(b) Top-{TOP_K} cases by operator")
    ax_bars.set_ylim(0, 1.05)
    ax_bars.legend(loc="upper right", **LEGEND_KW)

    fig.tight_layout()
    out = output_path or (FIGURES_DIR / "real_example.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def results_table_latex(df: pd.DataFrame) -> str:
    """Generate a LaTeX tabular body for the manuscript."""
    cols = ["case_id", "R", "Q"] + [f"P_{op}" for op in OPERATORS] + [f"rank_{op}" for op in OPERATORS]
    sub = df.sort_values("P_linear", ascending=False)[cols]
    lines = []
    for _, row in sub.iterrows():
        name = row["case_id"].replace("_", r"\_")
        lines.append(
            f"{name} & {row['R']:.4f} & {row['Q']:.4f} & "
            f"{row['P_linear']:.4f} & {row['P_min']:.4f} & {row['P_geometric']:.4f} & "
            f"{int(row['rank_linear'])} & {int(row['rank_min'])} & {int(row['rank_geometric'])} \\\\"
        )
    return "\n".join(lines)
