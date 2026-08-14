"""Publication-quality figures for systemic CADEMAS evaluation."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from plot_style import (
    add_panel_letter,
    apply_helvetica_style,
    plot_operator_mean_ci,
    style_axes_frame,
)
from systemic_config import (
    FIGURES_DIR,
    NOISE_FIGSIZE,
    NOISE_LAMBDAS,
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


def _plot_metric_panel(ax, df, operator, x_col, mean_col, lo_col, hi_col, label, **style_overrides):
    sub = df[df["operator"] == operator].sort_values(x_col)
    plot_operator_mean_ci(
        ax,
        sub[x_col],
        sub[mean_col],
        sub[lo_col],
        sub[hi_col],
        operator,
        label=label,
        **style_overrides,
    )


def _aggregate_noise_for_plot(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Average MC summaries over population model and noise distribution."""
    mean_col = f"{metric}_mean"
    lo_col = f"{metric}_ci_low"
    hi_col = f"{metric}_ci_high"
    grouped = (
        df.groupby(["lambda", "sigma", "operator"], as_index=False)[[mean_col, lo_col, hi_col]]
        .mean()
        .sort_values(["lambda", "sigma", "operator"])
    )
    return grouped


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
    add_panel_letter(ax_v, "a")
    ax_v.set_xlim(0.0, 1.0)
    v_k_max = df["v_k_ci_high"].max()
    ax_v.set_ylim(-0.005, max(0.24, v_k_max * 1.08))

    ax_e.set_xlabel(r"Trade-off parameter $\lambda$")
    ax_e.set_ylabel(r"Predictive Efficiency ($\overline{R}_{\mathrm{std}} \in \mathrm{Top}\text{-}K$)")
    add_panel_letter(ax_e, "b")
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


def plot_predictive_overconfidence(
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
    ax.set_title(rf"Predictive Overconfidence ($\lambda = {lam:.2f}$)", fontweight="bold")
    ax.set_xlim(0.48, 1.02)
    ax.set_ylim(-0.01, max(0.35, df["v_k_ci_high"].max() * 1.15))
    ax.legend(loc="upper left", **LEGEND_KW)
    style_axes_frame(ax)
    fig.tight_layout()

    out = output_path or (_fig_dir() / "predictive_overconfidence.pdf")
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_noise_propagation(df: pd.DataFrame, output_path: Path | None = None) -> Path:
    """2x3 grid: Kendall tau and Top-K Jaccard vs sigma for each lambda."""
    _style()
    fig, axes = plt.subplots(2, 3, figsize=NOISE_FIGSIZE, sharex=True)

    tau_df = _aggregate_noise_for_plot(df, "tau")
    jacc_df = _aggregate_noise_for_plot(df, "jaccard")

    panel_letters = ["a", "b", "c", "d", "e", "f"]
    letter_idx = 0

    for col, lam in enumerate(NOISE_LAMBDAS):
        ax_tau = axes[0, col]
        ax_jac = axes[1, col]
        tau_sub = tau_df[tau_df["lambda"] == lam]
        jac_sub = jacc_df[jacc_df["lambda"] == lam]

        for op in ("linear", "min"):
            series_kw = {"markersize": 6.5 if op == "linear" else 4.8, "linewidth": 1.8}
            _plot_metric_panel(
                ax_tau,
                tau_sub,
                op,
                "sigma",
                "tau_mean",
                "tau_ci_low",
                "tau_ci_high",
                SYSTEMIC_LABELS[op],
                **series_kw,
            )
            _plot_metric_panel(
                ax_jac,
                jac_sub,
                op,
                "sigma",
                "jaccard_mean",
                "jaccard_ci_low",
                "jaccard_ci_high",
                SYSTEMIC_LABELS[op],
                **series_kw,
            )

        add_panel_letter(ax_tau, panel_letters[letter_idx])
        letter_idx += 1
        add_panel_letter(ax_jac, panel_letters[letter_idx])
        letter_idx += 1

        ax_tau.set_title(rf"$\lambda = {lam:.2f}$", fontsize=10, pad=4)
        ax_tau.set_ylabel(r"Kendall's $\tau$")
        ax_tau.set_xlim(-0.02, 0.52)
        ax_jac.set_ylabel(r"Top-$K$ Jaccard overlap")
        ax_jac.set_xlabel(r"Contextual noise level $\sigma$")
        ax_jac.set_xlim(-0.02, 0.52)

        for ax in (ax_tau, ax_jac):
            style_axes_frame(ax)

    axes[0, 0].legend(loc="upper right", **LEGEND_KW)
    fig.tight_layout()

    out = output_path or (_fig_dir() / "noise_propagation.pdf")
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_intermediate_context(df: pd.DataFrame, output_path: Path | None = None) -> Path:
    """Acceptance rate of intermediate-context candidates vs lambda."""
    _style()
    fig, ax = plt.subplots(figsize=SINGLE_PANEL_FIGSIZE)

    for op in ("linear", "geometric"):
        _plot_metric_panel(
            ax,
            df,
            op,
            "lambda",
            "acceptance_mean",
            "acceptance_ci_low",
            "acceptance_ci_high",
            SYSTEMIC_LABELS[op],
        )

    ax.set_xlabel(r"Trade-off parameter $\lambda$")
    ax.set_ylabel(r"Acceptance rate of $X_{\mathrm{inter}}$ in Top-$K$")
    ax.set_title(
        r"Intermediate context compensation ($Q=0.5$, $R\approx 0.9$)",
        fontweight="bold",
    )
    ax.set_xlim(0.05, 0.95)
    y_max = max(0.35, float(df["acceptance_ci_high"].max()) * 1.12)
    ax.set_ylim(-0.01, y_max)
    ax.legend(loc="upper left", **LEGEND_KW)
    style_axes_frame(ax)
    fig.tight_layout()

    out = output_path or (_fig_dir() / "intermediate_context.pdf")
    fig.savefig(out)
    plt.close(fig)
    return out
