#!/usr/bin/env python3
"""Analysis 1 — λ perturbation sensitivity (Experiment 04).

Measures ranking stability (Kendall τ, Top-K Jaccard) when λ shifts
around the reference λ=0.75. Policy violation rate V is reported as
secondary information only — this is not a compliance experiment.

Grid: λ ∈ {0.70, 0.725, 0.75, 0.775, 0.80}, σ_R = 0, baseline population.
Reference: P_ref = A(R, Q, λ=0.75).
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

from _mc_utils import run_monte_carlo_cached
from common.aggregators import aggregate as agg_fn
from common.config import (
    CI_LEVEL,
    LAMBDA_PERTURBATION_VALUES,
    MC_SEED_BASE,
    N_CASES,
    N_MONTE_CARLO,
    OPERATORS,
    Q_WEAK_THRESHOLD,
    SEEDS,
    SENSITIVITY_PRIMARY_LAMBDA,
    TOP_K,
)
from common.metrics import (
    jaccard_top_k,
    kendall_tau,
    policy_violation_rate,
    predictive_utility,
)
from common.plotting import (
    add_panel_letter,
    new_square_two_panel,
    plot_line_mean_ci,
    save_figure,
    style_square_box,
)
from common.trial_runner import PopulationConfig, generate_scenario_population
from common.utils import ensure_dirs, experiment_fingerprint, summarize_trials, trial_seed

METRICS = (
    "kendall_tau",
    "jaccard_top_k",
    "delta_r_bar",
    "policy_violation_rate",
    "predictive_utility",
)


def _run_trial(trial_idx: int) -> list[dict]:
    pop_config = PopulationConfig.from_scenario("baseline")
    pop, _rng = generate_scenario_population(trial_idx, pop_config)

    records: list[dict] = []
    for operator in OPERATORS:
        p_ref = agg_fn(operator, pop.R, pop.Q, SENSITIVITY_PRIMARY_LAMBDA)
        r_bar_ref = predictive_utility(p_ref, pop.R, TOP_K, case_id=pop.case_id)

        for lam in LAMBDA_PERTURBATION_VALUES:
            p_pert = agg_fn(operator, pop.R, pop.Q, lam)
            tau = kendall_tau(p_ref, p_pert)
            jac = jaccard_top_k(
                p_ref, p_pert, TOP_K,
                R_ref=pop.R, R=pop.R, case_id=pop.case_id,
            )
            r_bar_pert = predictive_utility(p_pert, pop.R, TOP_K, case_id=pop.case_id)
            v = policy_violation_rate(p_pert, pop.Q, TOP_K, R=pop.R, case_id=pop.case_id)
            records.append({
                "trial": trial_idx,
                "seed": trial_seed(trial_idx),
                "analysis": "lambda_perturbation",
                "operator": operator,
                "lambda": lam,
                "kendall_tau": tau,
                "jaccard_top_k": jac,
                "delta_r_bar": r_bar_pert - r_bar_ref,
                "policy_violation_rate": v,
                "predictive_utility": r_bar_pert,
            })
    return records


def _fingerprint() -> str:
    pop = PopulationConfig.from_scenario("baseline")
    return experiment_fingerprint(
        "04_lambda_perturbation",
        {
            "analysis": "lambda_perturbation",
            "lambda_ref": SENSITIVITY_PRIMARY_LAMBDA,
            "lambda_values": list(LAMBDA_PERTURBATION_VALUES),
            "n_cases": N_CASES,
            "top_k": TOP_K,
            "n_mc": N_MONTE_CARLO,
            "seeds": SEEDS,
            "operators": list(OPERATORS),
            "veto_fraction": pop.veto_fraction,
        },
    )


def run_analysis(refresh: bool = False) -> pd.DataFrame:
    dirs = ensure_dirs(EXP_DIR)
    return run_monte_carlo_cached(
        results_dir=dirs["results"],
        raw_filename="lambda_perturbation.csv",
        meta_filename="lambda_perturbation_metadata.json",
        fingerprint=_fingerprint(),
        experiment_name="04_sensitivity/lambda_perturbation",
        meta_extra={
            "lambda_ref": SENSITIVITY_PRIMARY_LAMBDA,
            "lambda_values": list(LAMBDA_PERTURBATION_VALUES),
            "n_monte_carlo": N_MONTE_CARLO,
            "seed_base": MC_SEED_BASE,
            "ci_level": CI_LEVEL,
        },
        refresh=refresh,
        trial_fn=_run_trial,
    )


def write_figures(agg: pd.DataFrame, figures_dir: Path) -> list[Path]:
    written: list[Path] = []
    fig, ax_a, ax_b = new_square_two_panel()

    plot_line_mean_ci(
        ax_a, agg,
        x_col="lambda", group_col="operator",
        y_col="kendall_tau_mean",
        ci_low_col="kendall_tau_ci_low",
        ci_high_col="kendall_tau_ci_high",
        xlabel=r"$\lambda$",
        ylabel=r"Kendall $\tau$",
        ylim=(0.85, 1.01),
    )
    ax_a.axvline(SENSITIVITY_PRIMARY_LAMBDA, color="gray", linestyle="--",
                 linewidth=0.8, alpha=0.7, label=r"$\lambda_{\mathrm{ref}}=0.75$")

    plot_line_mean_ci(
        ax_b, agg,
        x_col="lambda", group_col="operator",
        y_col="jaccard_top_k_mean",
        ci_low_col="jaccard_top_k_ci_low",
        ci_high_col="jaccard_top_k_ci_high",
        xlabel=r"$\lambda$",
        ylabel=r"Top-$K$ Jaccard similarity",
        ylim=(0.70, 1.01),
    )
    ax_b.axvline(SENSITIVITY_PRIMARY_LAMBDA, color="gray", linestyle="--",
                 linewidth=0.8, alpha=0.7)
    ax_b.get_legend().remove()

    style_square_box(ax_a)
    style_square_box(ax_b)
    add_panel_letter(ax_a, "a")
    add_panel_letter(ax_b, "b")
    written.extend(save_figure(fig, figures_dir / "lambda_perturbation_stability",
                               bbox_inches=None))
    return written


def write_table(agg: pd.DataFrame, tables_dir: Path) -> Path:
    rows = []
    for lam in LAMBDA_PERTURBATION_VALUES:
        sub = agg[np.isclose(agg["lambda"].to_numpy(), lam)]
        for _, row in sub.iterrows():
            rows.append({
                "lambda": lam,
                "operator": row["operator"],
                "tau": row["kendall_tau_mean"],
                "jaccard": row["jaccard_top_k_mean"],
                "delta_r_bar": row["delta_r_bar_mean"],
            })

    lines = [
        r"\begin{table}[H]",
        r"\caption{Ranking stability under $\lambda$ perturbation at $\lambda_{\mathrm{ref}}=0.75$"
        r" ($\sigma_R=0$, baseline population). Kendall $\tau$ and Top-$K$ Jaccard similarity"
        r" are computed relative to the reference ranking at $\lambda=0.75$."
        r" Values are Monte Carlo means over $1000$ replications ($N=1000$, $K=100$).}",
        r"\label{tab:lambda-stability}",
        r"\centering\small\renewcommand{\arraystretch}{1.2}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"$\lambda$ & Operator & Kendall $\tau$ & Jaccard Top-$K$ \\",
        r"\midrule",
    ]
    prev_lam = None
    for r in rows:
        lam_str = f"${r['lambda']:.3f}$" if prev_lam != r["lambda"] else ""
        if prev_lam != r["lambda"] and prev_lam is not None:
            lines.append(r"\midrule")
        op_map = {"linear": r"$A_L$", "geometric": r"$A_G$", "min": r"$A_M$"}
        lines.append(
            f"{lam_str} & {op_map[r['operator']]} & {r['tau']:.4f} & {r['jaccard']:.4f} \\\\"
        )
        prev_lam = r["lambda"]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out = tables_dir / "table_lambda_stability.tex"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def main(refresh: bool = False) -> pd.DataFrame:
    dirs = ensure_dirs(EXP_DIR)
    raw = run_analysis(refresh=refresh)
    agg = summarize_trials(raw, ["lambda", "operator"], list(METRICS))
    agg.to_csv(dirs["results"] / "aggregated_lambda.csv", index=False)
    write_figures(agg, dirs["figures"])
    write_table(agg, dirs["tables"])
    print("[done] lambda perturbation analysis complete")
    return agg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    main(refresh=args.refresh)
