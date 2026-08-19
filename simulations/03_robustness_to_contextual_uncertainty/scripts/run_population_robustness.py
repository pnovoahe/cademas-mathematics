#!/usr/bin/env python3
"""Analysis 3 — Population robustness (Experiment 03).

Per scenario: full λ sweep at σ_Q=0 + full σ_Q sweep at λ=0.75.
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
    LAMBDA_ROBUSTNESS_VALUES,
    MC_SEED_BASE,
    N_CASES,
    N_MONTE_CARLO,
    OPERATORS,
    POPULATION_SCENARIOS,
    Q_WEAK_THRESHOLD,
    SEEDS,
    SIGMA_Q_VALUES,
    TOP_K,
)
from common.plotting import (  # noqa: E402
    OPERATOR_COLORS,
    apply_paper_style,
    new_single_axes,
    save_figure,
    style_axes_frame,
)
from common.trial_runner import (  # noqa: E402
    PRIMARY_METRICS,
    PopulationConfig,
    run_lambda_grid_trial,
    run_sigma_q_grid_trial,
)
from common.utils import ensure_dirs, experiment_fingerprint, summarize_trials  # noqa: E402

LAM = CONTEXT_NOISE_PRIMARY_LAMBDA


def _scenario_fingerprint(scenario: str, sweep: str) -> str:
    pop = PopulationConfig.from_scenario(scenario)
    params: dict = {
        "analysis": "population_robustness",
        "scenario": scenario,
        "sweep": sweep,
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
    }
    if sweep == "lambda":
        params["sigma_q"] = 0.0
        params["lambda_values"] = list(LAMBDA_ROBUSTNESS_VALUES)
    else:
        params["lambda"] = LAM
        params["sigma_q_values"] = list(SIGMA_Q_VALUES)
    return experiment_fingerprint(f"03_population_{scenario}_{sweep}", params)


def run_scenario(scenario: str, refresh: bool) -> pd.DataFrame:
    dirs = ensure_dirs(EXP_DIR)
    pop = PopulationConfig.from_scenario(scenario)
    records: list[pd.DataFrame] = []

    raw_lam = run_monte_carlo_cached(
        results_dir=dirs["results"],
        raw_filename=f"population_{scenario}_lambda.csv",
        meta_filename=f"population_{scenario}_lambda_metadata.json",
        fingerprint=_scenario_fingerprint(scenario, "lambda"),
        experiment_name=f"03_robustness/population/{scenario}/lambda",
        meta_extra={"scenario": scenario, "sweep": "lambda", "sigma_q": 0.0},
        refresh=refresh,
        trial_fn=lambda i: run_lambda_grid_trial(
            i, pop, LAMBDA_ROBUSTNESS_VALUES, sigma_q=0.0
        ),
    )
    raw_lam["sweep"] = "lambda"
    records.append(raw_lam)

    raw_sig = run_monte_carlo_cached(
        results_dir=dirs["results"],
        raw_filename=f"population_{scenario}_sigma.csv",
        meta_filename=f"population_{scenario}_sigma_metadata.json",
        fingerprint=_scenario_fingerprint(scenario, "sigma"),
        experiment_name=f"03_robustness/population/{scenario}/sigma",
        meta_extra={"scenario": scenario, "sweep": "sigma_q", "lambda": LAM},
        refresh=refresh,
        trial_fn=lambda i: run_sigma_q_grid_trial(
            i, pop, SIGMA_Q_VALUES, lam=LAM
        ),
    )
    raw_sig["sweep"] = "sigma_q"
    records.append(raw_sig)

    return pd.concat(records, ignore_index=True)


def run_all_scenarios(refresh: bool, scenarios: tuple[str, ...] | None = None) -> pd.DataFrame:
    if scenarios is None:
        scenarios = tuple(POPULATION_SCENARIOS.keys())
    parts = [run_scenario(s, refresh) for s in scenarios]
    combined = pd.concat(parts, ignore_index=True)
    out_path = ensure_dirs(EXP_DIR)["results"] / "population_variants.csv"
    combined.to_csv(out_path, index=False)
    print(f"[done] wrote {out_path} ({len(combined)} rows)")
    return combined


def write_figures(agg: pd.DataFrame, figures_dir: Path) -> list[Path]:
    written: list[Path] = []
    apply_paper_style()
    linear_lam = agg[(agg["operator"] == "linear") & (agg.get("sweep", "lambda") == "lambda")]
    if "sweep" not in agg.columns:
        linear_lam = agg[agg["operator"] == "linear"]

    fig, ax = new_single_axes()
    for scenario, sub in linear_lam.groupby("scenario"):
        sub = sub.sort_values("lambda")
        ax.plot(
            sub["lambda"],
            sub["policy_violation_rate_mean"],
            marker="o",
            markersize=4,
            label=scenario,
        )
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$V$ under $A_L$")
    ax.set_title(r"Population robustness: $V(\lambda)$ at $\sigma_Q=0$")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    style_axes_frame(ax)
    written.extend(save_figure(fig, figures_dir / "population_robustness_lambda", bbox_inches=None))

    linear_sig = agg[(agg["operator"] == "linear")]
    if "sweep" in agg.columns:
        linear_sig = linear_sig[linear_sig["sweep"] == "sigma_q"]
    elif "sigma_q" in agg.columns:
        pass

    fig2, ax2 = new_single_axes()
    for scenario, sub in linear_sig.groupby("scenario"):
        sub = sub.sort_values("sigma_q")
        ax2.plot(
            sub["sigma_q"],
            sub["policy_violation_rate_mean"],
            marker="s",
            markersize=4,
            label=scenario,
        )
    ax2.set_xlabel(r"$\sigma_Q$")
    ax2.set_ylabel(r"$V$ under $A_L$")
    ax2.set_title(rf"Population robustness: $V(\sigma_Q)$ at $\lambda={LAM}$")
    ax2.legend(frameon=False, fontsize=8, ncol=2)
    style_axes_frame(ax2)
    written.extend(save_figure(fig2, figures_dir / "population_robustness_noise", bbox_inches=None))
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument(
        "--scenario",
        default=None,
        help="Run one scenario only (default: all)",
    )
    args = parser.parse_args()

    dirs = ensure_dirs(EXP_DIR)
    scenarios = (args.scenario,) if args.scenario else tuple(POPULATION_SCENARIOS.keys())
    raw = run_all_scenarios(args.refresh, scenarios)

    agg_lam = summarize_trials(
        raw[raw["sweep"] == "lambda"],
        ["scenario", "lambda", "operator"],
        PRIMARY_METRICS,
    )
    agg_lam["sweep"] = "lambda"
    agg_sig = summarize_trials(
        raw[raw["sweep"] == "sigma_q"],
        ["scenario", "sigma_q", "operator"],
        PRIMARY_METRICS,
    )
    agg_sig["sweep"] = "sigma_q"
    agg = pd.concat([agg_lam, agg_sig], ignore_index=True)
    agg.to_csv(dirs["results"] / "population_variants_aggregated.csv", index=False)
    write_figures(agg, dirs["figures"])
    print("[done] population robustness complete")


if __name__ == "__main__":
    main()
