#!/usr/bin/env python3
"""Analysis 1 — Sensitivity to aggregation weight λ (Experiment 03).

λ ∈ {0.0, 0.1, …, 1.0}, σ_Q = 0, baseline population.
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

import numpy as np
import pandas as pd

from _mc_utils import run_monte_carlo_cached  # noqa: E402
from common.config import (  # noqa: E402
    CI_LEVEL,
    LAMBDA_BAR_VALUES,
    LAMBDA_ROBUSTNESS_VALUES,
    MC_SEED_BASE,
    N_CASES,
    N_MONTE_CARLO,
    OPERATORS,
    Q_WEAK_THRESHOLD,
    SEEDS,
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
from common.config import OPERATOR_FULL_LABELS  # noqa: E402
from common.trial_runner import (  # noqa: E402
    PRIMARY_METRICS,
    PopulationConfig,
    run_lambda_grid_trial,
)
from common.utils import ensure_dirs, experiment_fingerprint, summarize_trials  # noqa: E402

EXP01_AGG = SIM_DIR / "01_policy_compliance_under_contextual_vetoes" / "results" / "aggregated_dense.csv"


def _fingerprint() -> str:
    pop = PopulationConfig.from_scenario("baseline")
    return experiment_fingerprint(
        "03_lambda_sensitivity",
        {
            "analysis": "lambda_sensitivity",
            "scenario": "baseline",
            "sigma_q": 0.0,
            "lambda_values": list(LAMBDA_ROBUSTNESS_VALUES),
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
    fp = _fingerprint()

    raw = run_monte_carlo_cached(
        results_dir=dirs["results"],
        raw_filename="lambda_sensitivity.csv",
        meta_filename="lambda_sensitivity_metadata.json",
        fingerprint=fp,
        experiment_name="03_robustness/lambda_sensitivity",
        meta_extra={
            "analysis": "lambda_sensitivity",
            "scenario": "baseline",
            "sigma_q": 0.0,
            "lambda_values": list(LAMBDA_ROBUSTNESS_VALUES),
            "n_monte_carlo": N_MONTE_CARLO,
            "seed_base": MC_SEED_BASE,
            "ci_level": CI_LEVEL,
        },
        refresh=refresh,
        trial_fn=lambda i: run_lambda_grid_trial(
            i, pop, LAMBDA_ROBUSTNESS_VALUES, sigma_q=0.0
        ),
    )
    return raw


def aggregate(raw: pd.DataFrame) -> pd.DataFrame:
    return summarize_trials(raw, ["lambda", "operator"], PRIMARY_METRICS)


def validate_exp01(agg: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    if not EXP01_AGG.exists():
        lines.append("WARNING: Exp 01 aggregated_dense.csv not found; skip cross-validation.")
        return lines

    exp01 = pd.read_csv(EXP01_AGG)
    lines.append("## Cross-validation vs Experiment 01")
    lines.append("")
    lines.append("| λ | Operator | Exp01 V | Exp03 V | |ΔV| |")
    lines.append("|---|----------|---------|---------|------|")
    for lam in LAMBDA_BAR_VALUES:
        for op in OPERATORS:
            r03 = agg[(agg["operator"] == op) & np.isclose(agg["lambda"].to_numpy(), lam)]
            r01 = exp01[(exp01["operator"] == op) & np.isclose(exp01["lambda"].to_numpy(), lam)]
            if r03.empty or r01.empty:
                continue
            v01 = float(r01["policy_violation_rate_mean"].iloc[0])
            v03 = float(r03["policy_violation_rate_mean"].iloc[0])
            lines.append(
                f"| {lam:.2f} | {op} | {v01:.4f} | {v03:.4f} | {abs(v01 - v03):.6f} |"
            )
    lines.append("")
    return lines


def write_figures(agg: pd.DataFrame, figures_dir: Path) -> list[Path]:
    written: list[Path] = []
    fig, ax_a, ax_b = new_square_two_panel()
    plot_line_mean_ci(
        ax_a,
        agg,
        x_col="lambda",
        group_col="operator",
        y_col="policy_violation_rate_mean",
        ci_low_col="policy_violation_rate_ci_low",
        ci_high_col="policy_violation_rate_ci_high",
        xlabel=r"$\lambda$",
        ylabel=r"Policy violation rate $V$",
        ylim=(-0.05, 1.0),
    )
    plot_line_mean_ci(
        ax_b,
        agg,
        x_col="lambda",
        group_col="operator",
        y_col="predictive_utility_mean",
        ci_low_col="predictive_utility_ci_low",
        ci_high_col="predictive_utility_ci_high",
        xlabel=r"$\lambda$",
        ylabel=r"Mean predictive score $\bar{R}$",
        ylim=None,
    )
    style_square_box(ax_a)
    style_square_box(ax_b)
    add_panel_letter(ax_a, "a")
    add_panel_letter(ax_b, "b")
    written.extend(save_figure(fig, figures_dir / "lambda_sensitivity", bbox_inches=None))

    linear = agg[agg["operator"] == "linear"].sort_values("lambda").copy()
    linear["group"] = "linear"
    comp_rows = []
    for col, label in (
        ("n_veto_topk", "Vetoes"),
        ("n_weak_nv_topk", "Weak non-vetoes"),
        ("n_normal_topk", "Normal"),
    ):
        sub = linear[["lambda", f"{col}_mean", f"{col}_ci_low", f"{col}_ci_high"]].copy()
        sub = sub.rename(
            columns={
                f"{col}_mean": "y",
                f"{col}_ci_low": "ci_low",
                f"{col}_ci_high": "ci_high",
            }
        )
        sub["series"] = label
        comp_rows.append(sub)
    comp_df = pd.concat(comp_rows, ignore_index=True)

    fig2, ax = new_single_axes()
    colors = {"Vetoes": "#0072B2", "Weak non-vetoes": "#D55E00", "Normal": "#009E73"}
    markers = {"Vetoes": "v", "Weak non-vetoes": "s", "Normal": "o"}
    for series, sub in comp_df.groupby("series"):
        ax.fill_between(
            sub["lambda"], sub["ci_low"], sub["ci_high"],
            color=colors[series], alpha=0.2, linewidth=0,
        )
        ax.plot(
            sub["lambda"], sub["y"],
            color=colors[series], marker=markers[series], label=series,
            markeredgecolor="white", markeredgewidth=0.5,
        )
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel("Mean count in Top-$K$")
    ax.set_title(r"Top-$K$ composition under $A_L$ ($\sigma_Q=0$)")
    ax.legend(loc="best", frameon=False)
    written.extend(save_figure(fig2, figures_dir / "lambda_topk_composition", bbox_inches=None))
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="Recompute Monte Carlo trials")
    args = parser.parse_args()

    dirs = ensure_dirs(EXP_DIR)
    raw = run_analysis(refresh=args.refresh)
    agg = aggregate(raw)
    agg.to_csv(dirs["results"] / "lambda_sensitivity_aggregated.csv", index=False)
    write_figures(agg, dirs["figures"])

    validation = validate_exp01(agg)
    val_path = dirs["results"] / "lambda_exp01_validation.md"
    val_path.write_text("\n".join(validation) + "\n", encoding="utf-8")
    print(f"[done] wrote {val_path}")


if __name__ == "__main__":
    main()
