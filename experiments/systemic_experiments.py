"""Systemic evaluation experiments for CADEMAS."""

import numpy as np
import pandas as pd

from operators import aggregate
from systemic_config import (
    PREDICTIVE_OVERCONFIDENCE_LAMBDA,
    INTER_LAMBDA_MAX,
    INTER_LAMBDA_MIN,
    INTER_LAMBDA_POINTS,
    INTER_Q,
    INTER_R_MU,
    INTER_R_STD,
    K,
    MC_SEED_BASE,
    MU_SHIFT_MAX,
    MU_SHIFT_MIN,
    MU_SHIFT_POINTS,
    N_INTER,
    N_MC_TRIALS,
    N_VETO,
    NOISE_DISTRIBUTIONS,
    NOISE_LAMBDAS,
    NOISE_POPULATION_MODELS,
    NOISE_SIGMA_MAX,
    NOISE_SIGMA_MIN,
    NOISE_SIGMA_POINTS,
    OPP_LAMBDA_MAX,
    OPP_LAMBDA_MIN,
    OPP_LAMBDA_POINTS,
    SEED,
    VETO_R_STD,
)
from systemic_data import (
    assign_veto_r_from_normal,
    generate_intermediate_population,
    generate_rank_stability_population,
    generate_systemic_population,
    perturb_context,
    perturb_context_uniform,
)
from mc_cache import load_or_compute
from systemic_metrics import (
    group_acceptance_rate,
    kendall_tau_rankings,
    policy_violation_rate,
    predictive_efficiency,
    top_k_jaccard,
)


def _evaluate(df: pd.DataFrame, operator: str, lam: float, Q: np.ndarray | None = None):
    R = df["R"].to_numpy()
    Q_arr = df["Q"].to_numpy() if Q is None else Q
    groups = df["group"].to_numpy()
    P = aggregate(operator, R, Q_arr, lam)
    v_k = policy_violation_rate(P, groups, K)
    eff = predictive_efficiency(P, R, groups, K)
    return P, v_k, eff


def run_opportunity_cost_experiment(rng: np.random.Generator) -> pd.DataFrame:
    """
    Exp 1: Sweep lambda; veto R ~ Beta(8,2) (high ML risk on policy-violating cases).
    """
    veto_r = np.clip(rng.beta(8, 2, size=N_VETO), 0, 1)
    df = generate_systemic_population(rng, veto_r=veto_r)

    lambdas = np.linspace(OPP_LAMBDA_MIN, OPP_LAMBDA_MAX, OPP_LAMBDA_POINTS)
    records = []
    for lam in lambdas:
        for op in ("linear", "geometric"):
            _, v_k, eff = _evaluate(df, op, lam)
            records.append(
                {"lambda": lam, "operator": op, "v_k": v_k, "efficiency": eff}
            )
    return pd.DataFrame(records)


def run_predictive_overconfidence_experiment(rng: np.random.Generator) -> pd.DataFrame:
    """
    Exp 2: Fix lambda; inflate veto-group ML risk R ~ N(mu_shift, 0.05).
    Uses lambda=0.5 per spec; if violations are too rare at 0.5,
    PREDICTIVE_OVERCONFIDENCE_LAMBDA in systemic_config can be raised
    (see module docstring in run script output).
    """
    mu_vals = np.linspace(MU_SHIFT_MIN, MU_SHIFT_MAX, MU_SHIFT_POINTS)
    records = []

    std_r = rng.uniform(0.0, 1.0, 800)
    std_q = rng.uniform(0.0, 1.0, 800)

    for mu in mu_vals:
        veto_r = assign_veto_r_from_normal(rng, mu, VETO_R_STD)
        df = generate_systemic_population(rng, veto_r=veto_r)
        df.loc[df["group"] == "std", "R"] = std_r
        df.loc[df["group"] == "std", "Q"] = std_q

        for op in ("linear", "geometric", "min"):
            _, v_k, _ = _evaluate(df, op, PREDICTIVE_OVERCONFIDENCE_LAMBDA)
            records.append({"mu_shift": mu, "operator": op, "v_k": v_k})

    return pd.DataFrame(records)


def run_noise_propagation_experiment(rng: np.random.Generator) -> pd.DataFrame:
    """
    Exp 3: Contextual imprecision under multiple lambda, population, and noise settings.
    """
    sigmas = np.linspace(NOISE_SIGMA_MIN, NOISE_SIGMA_MAX, NOISE_SIGMA_POINTS)
    records = []

    for population in NOISE_POPULATION_MODELS:
        df = generate_rank_stability_population(rng, model=population)
        R = df["R"].to_numpy()
        Q = df["Q"].to_numpy()

        for noise_dist in NOISE_DISTRIBUTIONS:
            for lam in NOISE_LAMBDAS:
                baseline = {}
                for op in ("linear", "min"):
                    baseline[op] = aggregate(op, R, Q, lam)

                for sigma in sigmas:
                    if noise_dist == "gaussian":
                        Q_pert = perturb_context(Q, rng, sigma)
                    else:
                        Q_pert = perturb_context_uniform(Q, rng, sigma)

                    for op in ("linear", "min"):
                        P_pert = aggregate(op, R, Q_pert, lam)
                        tau = kendall_tau_rankings(baseline[op], P_pert)
                        jaccard = top_k_jaccard(baseline[op], P_pert, K)
                        records.append(
                            {
                                "population": population,
                                "noise_dist": noise_dist,
                                "lambda": lam,
                                "sigma": sigma,
                                "operator": op,
                                "tau": tau,
                                "jaccard": jaccard,
                            }
                        )

    return pd.DataFrame(records)


def run_intermediate_context_experiment(rng: np.random.Generator) -> pd.DataFrame:
    """
    Exp 4: Moderate context Q=0.5 with high predictive scores; sweep lambda.
    Reports the Top-K acceptance rate of the intermediate group.
    """
    inter_r = np.clip(rng.normal(INTER_R_MU, INTER_R_STD, size=N_INTER), 0.0, 1.0)
    df = generate_intermediate_population(rng, inter_r=inter_r, q_inter=INTER_Q)
    lambdas = np.linspace(INTER_LAMBDA_MIN, INTER_LAMBDA_MAX, INTER_LAMBDA_POINTS)
    records = []
    for lam in lambdas:
        for op in ("linear", "geometric"):
            R = df["R"].to_numpy()
            Q = df["Q"].to_numpy()
            groups = df["group"].to_numpy()
            P = aggregate(op, R, Q, lam)
            acc = group_acceptance_rate(P, groups, K, group_name="inter")
            records.append({"lambda": lam, "operator": op, "acceptance": acc})
    return pd.DataFrame(records)


def _aggregate_mc(df: pd.DataFrame, group_cols: list[str], metrics: list[str]) -> pd.DataFrame:
    """Aggregate Monte Carlo trials into mean and 95% CI per group."""
    rows = []
    for keys, sub in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        for metric in metrics:
            mean, lo, hi = _mean_ci(sub[metric].to_numpy())
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci_low"] = lo
            row[f"{metric}_ci_high"] = hi
        rows.append(row)
    return pd.DataFrame(rows)


def _mean_ci(values) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    return (
        float(np.mean(arr)),
        float(np.percentile(arr, 2.5)),
        float(np.percentile(arr, 97.5)),
    )


def run_opportunity_cost_mc(
    n_trials: int = N_MC_TRIALS,
    seed_base: int = MC_SEED_BASE,
    refresh: bool = False,
) -> pd.DataFrame:
    def compute() -> pd.DataFrame:
        frames = []
        for trial in range(n_trials):
            trial_df = run_opportunity_cost_experiment(
                np.random.default_rng(seed_base + trial)
            )
            trial_df["trial"] = trial
            frames.append(trial_df)
        return _aggregate_mc(
            pd.concat(frames, ignore_index=True),
            ["lambda", "operator"],
            ["v_k", "efficiency"],
        )

    return load_or_compute(
        "opportunity_cost",
        compute,
        n_trials=n_trials,
        seed_base=seed_base,
        refresh=refresh,
    )


def run_predictive_overconfidence_mc(
    n_trials: int = N_MC_TRIALS,
    seed_base: int = MC_SEED_BASE,
    refresh: bool = False,
) -> pd.DataFrame:
    def compute() -> pd.DataFrame:
        frames = []
        for trial in range(n_trials):
            trial_df = run_predictive_overconfidence_experiment(
                np.random.default_rng(seed_base + trial)
            )
            trial_df["trial"] = trial
            frames.append(trial_df)
        return _aggregate_mc(
            pd.concat(frames, ignore_index=True),
            ["mu_shift", "operator"],
            ["v_k"],
        )

    return load_or_compute(
        "overconfidence",
        compute,
        n_trials=n_trials,
        seed_base=seed_base,
        refresh=refresh,
    )


def run_noise_propagation_mc(
    n_trials: int = N_MC_TRIALS,
    seed_base: int = MC_SEED_BASE,
    refresh: bool = False,
) -> pd.DataFrame:
    def compute() -> pd.DataFrame:
        frames = []
        for trial in range(n_trials):
            trial_df = run_noise_propagation_experiment(
                np.random.default_rng(seed_base + trial)
            )
            trial_df["trial"] = trial
            frames.append(trial_df)
        return _aggregate_mc(
            pd.concat(frames, ignore_index=True),
            ["population", "noise_dist", "lambda", "sigma", "operator"],
            ["tau", "jaccard"],
        )

    return load_or_compute(
        "noise_propagation",
        compute,
        n_trials=n_trials,
        seed_base=seed_base,
        refresh=refresh,
    )


def run_intermediate_context_mc(
    n_trials: int = N_MC_TRIALS,
    seed_base: int = MC_SEED_BASE,
    refresh: bool = False,
) -> pd.DataFrame:
    def compute() -> pd.DataFrame:
        frames = []
        for trial in range(n_trials):
            trial_df = run_intermediate_context_experiment(
                np.random.default_rng(seed_base + trial)
            )
            trial_df["trial"] = trial
            frames.append(trial_df)
        return _aggregate_mc(
            pd.concat(frames, ignore_index=True),
            ["lambda", "operator"],
            ["acceptance"],
        )

    return load_or_compute(
        "intermediate_context",
        compute,
        n_trials=n_trials,
        seed_base=seed_base,
        refresh=refresh,
    )


def run_all_systemic_experiments_mc(
    n_trials: int = N_MC_TRIALS,
    seed_base: int = MC_SEED_BASE,
    refresh: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        run_opportunity_cost_mc(n_trials, seed_base, refresh=refresh),
        run_predictive_overconfidence_mc(n_trials, seed_base, refresh=refresh),
        run_noise_propagation_mc(n_trials, seed_base, refresh=refresh),
        run_intermediate_context_mc(n_trials, seed_base, refresh=refresh),
    )


def run_all_systemic_experiments() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    return (
        run_opportunity_cost_experiment(rng),
        run_predictive_overconfidence_experiment(np.random.default_rng(SEED + 1)),
        run_noise_propagation_experiment(np.random.default_rng(SEED + 2)),
        run_intermediate_context_experiment(np.random.default_rng(SEED + 3)),
    )
