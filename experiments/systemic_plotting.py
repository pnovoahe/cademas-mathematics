"""Publication-quality figures for systemic CADEMAS evaluation."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from plot_style import apply_helvetica_style, plot_operator_mean_ci, style_axes_frame
from systemic_config import (
    FIGURES_DIR,
    NOISE_LAMBDA,
    PLOT_FONT_SCALE,
    SINGLE_PANEL_FIGSIZE,
    TWO_PANEL_FIGSIZE,
    SYSTEMIC_LABELS,
)

LEGEND_KW = {
    "frameon": True,
    "facecolor": "white",
    "edgecolor": "0.75",
    "framealpha": 1.0,
}


def _style() -> None:
    apply_helvetica_style(font_scale=PLOT_FONT_SCALE)


def _fig_dir() -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURES_DIR


def _plot_metric_panel(ax, df, operator, x_col, mean_col, lo_col, hi_col, label):
    sub = df[df["operator"] == operator].sort_values(x_col)
    plot_operator_mean_ci(
        ax,
        sub[x_col],
        sub[mean_col],
        sub[lo_col],
        sub[hi_col],
        operator,
        label=label,
    )


def plot_opportunity_cost(df: pd.DataFrame, output_path: Path | None = None) -> Path:
    """Two-panel layout: (a) V_K and (b) Predictive Efficiency vs lambda."""
    _style()
    fig, (ax_v, ax_e) = plt.subplots(1, 2, figsize=TWO_PANEL_FIGSIZE, sharex=True)

    for op in ("linear", "geometric"):
        label = SYSTEMIC_LABELS[op]
        _plot_metric_panel(ax_v, df, op, "lambda", "v_k_mean", "v_k_ci_low", "v_k_ci_high", label)
        _plot_metric_panel(
            ax_e, df, op, "lambda", "efficiency_mean", "efficiency_ci_low", "efficiency_ci_high", label
        )

    ax_v.set_xlabel(r"Trade-off parameter $\lambda$")
    ax_v.set_ylabel(r"Policy Violation Rate $V_K$")
    ax_v.set_title("(a) Institutional compliance", fontweight="bold")
    ax_v.set_xlim(0.0, 1.0)
    v_k_max = df["v_k_ci_high"].max()
    ax_v.set_ylim(-0.005, max(0.24, v_k_max * 1.08))

    ax_e.set_xlabel(r"Trade-off parameter $\lambda$")
    ax_e.set_ylabel(r"Predictive Efficiency ($\overline{R}_{\mathrm{std}} \in \mathrm{Top}\text{-}K$)")
    ax_e.set_title("(b) Predictive utility", fontweight="bold")
    eff_min = df["efficiency_ci_low"].min()
    eff_max = df["efficiency_ci_high"].max()
    eff_pad = (eff_max - eff_min) * 0.08
    ax_e.set_ylim(eff_min - eff_pad, eff_max + eff_pad)

    for ax in (ax_v, ax_e):
        ax.set_box_aspect(1)
        style_axes_frame(ax)

    ax_v.legend(loc="upper left", **LEGEND_KW)
    ax_e.legend(loc="lower right", **LEGEND_KW)
    fig.tight_layout()

    out = output_path or (_fig_dir() / "opportunity_cost.pdf")
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_algorithmic_firewall(
    df: pd.DataFrame, output_path: Path | None = None, lam: float = 0.5
) -> Path:
    """V_K vs local ML overconfidence on veto group."""
    _style()
    fig, ax = plt.subplots(figsize=SINGLE_PANEL_FIGSIZE)

    for op in ("linear", "min", "geometric"):
        _plot_metric_panel(
            ax, df, op, "mu_shift", "v_k_mean", "v_k_ci_low", "v_k_ci_high", SYSTEMIC_LABELS[op]
        )

    ax.set_xlabel(r"Local ML Overconfidence ($\mu_{\mathrm{shift}}$ on veto group)")
    ax.set_ylabel(r"Policy Violation Rate $V_K$")
    ax.set_title(rf"Algorithmic Firewall ($\lambda = {lam:.2f}$)", fontweight="bold")
    ax.set_xlim(0.48, 1.02)
    ax.set_ylim(-0.01, max(0.35, df["v_k_ci_high"].max() * 1.15))
    ax.legend(loc="upper left", **LEGEND_KW)
    style_axes_frame(ax)
    fig.tight_layout()

    out = output_path or (_fig_dir() / "algorithmic_firewall.pdf")
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_noise_propagation(df: pd.DataFrame, output_path: Path | None = None) -> Path:
    """Kendall tau vs contextual noise sigma."""
    _style()
    fig, ax = plt.subplots(figsize=SINGLE_PANEL_FIGSIZE)

    for op in ("linear", "min"):
        _plot_metric_panel(
            ax, df, op, "sigma", "tau_mean", "tau_ci_low", "tau_ci_high", SYSTEMIC_LABELS[op]
        )

    ax.set_xlabel(r"Contextual noise level $\sigma$")
    ax.set_ylabel(r"Rank stability (Kendall's $\tau$ vs. $\sigma=0$)")
    ax.set_title(
        rf"Noise Propagation under Contextual Uncertainty ($\lambda = {NOISE_LAMBDA:.2f}$)",
        fontweight="bold",
    )
    ax.set_xlim(-0.02, 0.52)

    tau_min = df["tau_ci_low"].min()
    tau_max = df["tau_ci_high"].max()
    y_pad = (tau_max - tau_min) * 0.12
    ax.set_ylim(tau_min - y_pad, tau_max + y_pad * 0.35)

    ax.legend(loc="lower left", **LEGEND_KW)
    style_axes_frame(ax)
    fig.tight_layout()

    out = output_path or (_fig_dir() / "noise_propagation.pdf")
    fig.savefig(out)
    plt.close(fig)
    return out
