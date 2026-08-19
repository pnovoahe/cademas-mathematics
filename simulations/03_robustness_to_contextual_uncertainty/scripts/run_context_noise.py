#!/usr/bin/env python3
"""Analysis 2 — Sensitivity to contextual noise σ_Q (Experiment 03).

σ_Q ∈ {0.00, …, 0.20}, λ = 0.75, baseline population.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
EXP_DIR = SCRIPTS_DIR.parent
SIM_DIR = EXP_DIR.parent
if str(SIM_DIR) not in sys.path:
    sys.path.insert(0, str(SIM_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

os.environ.setdefault("MPLCONFIGDIR", str(SIM_DIR / ".mplconfig"))
(SIM_DIR / ".mplconfig").mkdir(parents=True, exist_ok=True)

import pandas as pd

from _mc_utils import run_monte_carlo_cached  # noqa: E402
from common.config import (  # noqa: E402
    CI_LEVEL,
    CONTEXT_NOISE_PRIMARY_LAMBDA,
    MC_SEED_BASE,
    N_CASES,
    N_MONTE_CARLO,
    OPERATORS,
    Q_WEAK_THRESHOLD,
    SEEDS,
    SIGMA_Q_VALUES,
    TOP_K,
)
from common.plotting import (  # noqa: E402
    add_panel_letter,
    new_single_axes,
    new_square_two_panel,
    plot_line_mean_ci,
    save_figure,
    style_square_box,
)
from common.trial_runner import (  # noqa: E402
    PRIMARY_METRICS,
    PopulationConfig,
    run_sigma_q_grid_trial,
)
from common.utils import ensure_dirs, experiment_fingerprint, summarize_trials  # noqa: E402

LAM = CONTEXT_NOISE_PRIMARY_LAMBDA


def _fingerprint() -> str:
    pop = PopulationConfig.from_scenario("baseline")
    return experiment_fingerprint(
        "03_context_noise",
        {
            "analysis": "context_noise",
            "scenario": "baseline",
            "lambda": LAM,
            "sigma_q_values": list(SIGMA_Q_VALUES),
            "n_cases": N_CASES,
            "top_k": TOP_K,
            "n_mc": N_MONTE_CARLO,
            "seeds": SEEDS,
            "q_weak": Q_WEAK_THRESHOLD,
            "veto_fraction": pop.veto_fraction,
            "std_r_beta": pop.std_r_beta,
            "std_q_beta": pop.std_q_beta,
            "veto_r_beta": pop.veto_r_beta,
            "operators": list(OPERATORS),
        },
    )


def run_analysis(refresh: bool) -> pd.DataFrame:
    dirs = ensure_dirs(EXP_DIR)
    pop = PopulationConfig.from_scenario("baseline")
    return run_monte_carlo_cached(
        results_dir=dirs["results"],
        raw_filename="contextual_noise.csv",
        meta_filename="contextual_noise_metadata.json",
        fingerprint=_fingerprint(),
        experiment_name="03_robustness/context_noise",
        meta_extra={
            "analysis": "context_noise",
            "scenario": "baseline",
            "lambda": LAM,
            "sigma_q_values": list(SIGMA_Q_VALUES),
            "n_monte_carlo": N_MONTE_CARLO,
            "seed_base": MC_SEED_BASE,
            "ci_level": CI_LEVEL,
        },
        refresh=refresh,
        trial_fn=lambda i: run_sigma_q_grid_trial(
            i, pop, SIGMA_Q_VALUES, lam=LAM
        ),
    )


def write_figures(agg: pd.DataFrame, figures_dir: Path) -> list[Path]:
    written: list[Path] = []
    fig, ax_a, ax_b = new_square_two_panel()
    plot_line_mean_ci(
        ax_a,
        agg,
        x_col="sigma_q",
        group_col="operator",
        y_col="policy_violation_rate_mean",
        ci_low_col="policy_violation_rate_ci_low",
        ci_high_col="policy_violation_rate_ci_high",
        xlabel=r"Contextual noise $\sigma_Q$",
        ylabel=r"Policy violation rate $V$",
        ylim=(-0.015, 0.3),
    )
    plot_line_mean_ci(
        ax_b,
        agg,
        x_col="sigma_q",
        group_col="operator",
        y_col="predictive_utility_mean",
        ci_low_col="predictive_utility_ci_low",
        ci_high_col="predictive_utility_ci_high",
        xlabel=r"Contextual noise $\sigma_Q$",
        ylabel=r"Mean predictive score $\bar{R}$",
        ylim=None,
    )
    style_square_box(ax_a)
    style_square_box(ax_b)
    ax_b.get_legend().remove()
    add_panel_letter(ax_a, "a")
    add_panel_letter(ax_b, "b")
    written.extend(save_figure(fig, figures_dir / "contextual_noise", bbox_inches=None))

    baseline = agg[agg["sigma_q"] == 0.0][["operator", "policy_violation_rate_mean", "predictive_utility_mean"]]
    baseline = baseline.rename(
        columns={
            "policy_violation_rate_mean": "v_base",
            "predictive_utility_mean": "r_base",
        }
    )
    merged = agg.merge(baseline, on="operator")
    merged["delta_v"] = merged["policy_violation_rate_mean"] - merged["v_base"]
    merged["delta_r"] = merged["predictive_utility_mean"] - merged["r_base"]

    fig2, ax = new_single_axes()
    plot_line_mean_ci(
        ax,
        merged,
        x_col="sigma_q",
        group_col="operator",
        y_col="delta_v",
        ci_low_col="policy_violation_rate_ci_low",
        ci_high_col="policy_violation_rate_ci_high",
        xlabel=r"Contextual noise $\sigma_Q$",
        ylabel=r"$\Delta V$ vs. $\sigma_Q=0$",
        ylim=None,
    )
    ax.axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_title(r"Change in $V$ relative to noiseless baseline ($\lambda=0.75$)")
    written.extend(save_figure(fig2, figures_dir / "contextual_noise_delta_v", bbox_inches=None))

    linear = agg[agg["operator"] == "linear"].sort_values("sigma_q")
    fig3, ax3 = new_single_axes()
    colors = {"n_veto_topk": "#0072B2", "n_weak_nv_topk": "#D55E00", "n_normal_topk": "#009E73"}
    labels = {
        "n_veto_topk": "Vetoes",
        "n_weak_nv_topk": "Weak non-vetoes",
        "n_normal_topk": "Normal",
    }
    for col, color in colors.items():
        ax3.plot(
            linear["sigma_q"],
            linear[f"{col}_mean"],
            marker="o",
            color=color,
            label=labels[col],
        )
    ax3.set_xlabel(r"Contextual noise $\sigma_Q$")
    ax3.set_ylabel("Mean count in Top-$K$")
    ax3.set_title(r"Top-$K$ composition under $A_L$")
    ax3.legend(frameon=False)
    written.extend(save_figure(fig3, figures_dir / "contextual_noise_composition", bbox_inches=None))
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    dirs = ensure_dirs(EXP_DIR)
    raw = run_analysis(refresh=args.refresh)
    agg = summarize_trials(raw, ["sigma_q", "operator"], PRIMARY_METRICS)
    agg.to_csv(dirs["results"] / "contextual_noise_aggregated.csv", index=False)
    write_figures(agg, dirs["figures"])
    print("[done] context noise analysis complete")


if __name__ == "__main__":
    main()
