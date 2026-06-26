"""Publication-quality plotting for CADEMAS simulation results."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import FIGURES_DIR, OPERATOR_LABELS, RQ2_LAMBDA_PANELS
from plot_style import OPERATOR_STYLE, apply_helvetica_style, plot_operator, style_axes_frame

LEGEND_KW = {
    "frameon": True,
    "facecolor": "white",
    "edgecolor": "0.75",
    "framealpha": 1.0,
}


def apply_publication_style() -> None:
    """Configure matplotlib for Q1 publication-quality figures."""
    apply_helvetica_style(
        **{
            "lines.linewidth": 2.0,
            "lines.markersize": 8,
        }
    )


def _ensure_figures_dir() -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURES_DIR


def plot_pareto_frontier(pareto: dict, output_path: Path | None = None) -> Path:
    """Plot RQ1 Pareto frontier with lambda-colored linear trace and zoomed compliance axis."""
    apply_publication_style()
    fig, (ax_main, ax_ref) = plt.subplots(
        1, 2, figsize=(10, 4.5), gridspec_kw={"width_ratios": [2.2, 1]}
    )

    linear = pareto["linear"]
    geometric = pareto["geometric"]

    plot_operator(
        ax_main,
        linear["tau"],
        linear["v_k"],
        "linear",
        label=OPERATOR_LABELS["linear"],
    )

    for lam_target in (0.86, 0.96):
        row = linear[np.isclose(linear["lambda"], lam_target)]
        if not row.empty:
            r = row.iloc[0]
            ax_main.annotate(
                f"$\\lambda={lam_target:.2f}$",
                (r["tau"], r["v_k"]),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=8,
                color=OPERATOR_STYLE["linear"]["color"],
            )

    plot_operator(
        ax_main,
        geometric["tau"],
        geometric["v_k"],
        "geometric",
        label=OPERATOR_LABELS["geometric"],
    )

    ax_main.set_xlabel("Predictive Rank Preservation (Kendall's $\\tau$)")
    ax_main.set_ylabel("Policy Violation Rate ($V_K$)")
    ax_main.set_title("(a) Trade-off regimes")
    ax_main.set_xlim(-0.05, 1.05)
    y_max = max(0.22, linear["v_k"].max() * 1.15)
    ax_main.set_ylim(-0.005, y_max)
    ax_main.legend(loc="upper left", **LEGEND_KW)

    ref_ops = [
        ("min", pareto["min"]),
        ("max", pareto["max"]),
    ]
    ref_labels = {
        "min": r"$A_T^{\min}$",
        "max": r"$A_S^{\max}$",
    }
    for i, (op, pt) in enumerate(ref_ops):
        style = OPERATOR_STYLE[op]
        ax_ref.plot([0, pt["tau"]], [i, i], color=style["color"], linewidth=2.5, alpha=0.55)
        ax_ref.scatter(
            [pt["tau"]],
            [i],
            marker=style["marker"],
            s=style["markersize"] ** 2 * 1.1,
            color=style["color"],
            edgecolors="black",
            linewidths=0.5,
            zorder=5,
        )
        ax_ref.text(
            pt["tau"] + 0.03,
            i,
            f"$\\tau={pt['tau']:.2f}$",
            va="center",
            fontsize=9,
        )

    ax_ref.set_yticks([0, 1])
    ax_ref.set_yticklabels([ref_labels["min"], ref_labels["max"]])
    ax_ref.set_xlabel("Kendall's $\\tau$")
    ax_ref.set_title("(b) Fixed operators")
    ax_ref.set_xlim(-0.05, 1.15)
    ax_ref.text(
        0.02,
        -0.35,
        r"$V_K = 0$ for both",
        transform=ax_ref.transAxes,
        fontsize=9,
        style="italic",
    )

    fig.suptitle(
        "Pareto Frontier: ML Utility vs. Institutional Compliance",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    for ax in (ax_main, ax_ref):
        style_axes_frame(ax)
    fig.tight_layout()

    out = output_path or (_ensure_figures_dir() / "pareto_frontier.pdf")
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_ml_overconfidence(
    overconfidence: pd.DataFrame, output_path: Path | None = None
) -> Path:
    """Plot RQ2 as faceted panels: V_K vs alpha for multiple lambda values."""
    apply_publication_style()
    n_panels = len(RQ2_LAMBDA_PANELS)
    fig, axes = plt.subplots(1, n_panels, figsize=(10, 3.8), sharey=True)
    if n_panels == 1:
        axes = [axes]

    y_max = max(0.25, overconfidence["v_k"].max() * 1.12)

    for ax, lam in zip(axes, RQ2_LAMBDA_PANELS):
        panel = overconfidence[np.isclose(overconfidence["lambda"], lam)]
        for operator in ("linear", "min", "geometric"):
            subset = panel[panel["operator"] == operator].sort_values("alpha", ascending=False)
            plot_operator(
                ax,
                subset["alpha"],
                subset["v_k"],
                operator,
                label=OPERATOR_LABELS[operator],
            )

        ax.set_title(f"$\\lambda = {lam:.2f}$")
        ax.set_xlim(0.05, 1.05)
        ax.set_ylim(-0.005, y_max)
        ax.invert_xaxis()
        if ax is axes[0]:
            ax.set_ylabel("Policy Violation Rate ($V_K$)")
        ax.set_xlabel("$\\alpha$ (calibrated $\\leftarrow$ overconfident)")

    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 1.08),
        **LEGEND_KW,
    )
    fig.suptitle(
        "Robustness to ML Overconfidence under Increasing Predictive Weight",
        fontsize=12,
        fontweight="bold",
        y=1.14,
    )
    for ax in axes:
        style_axes_frame(ax)
    fig.tight_layout()

    out = output_path or (_ensure_figures_dir() / "ml_overconfidence.pdf")
    fig.savefig(out)
    plt.close(fig)
    return out
