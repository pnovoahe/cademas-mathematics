"""Systemic evaluation experiments for CADEMAS (three-figure study)."""

import numpy as np
import pandas as pd

from operators import aggregate
from systemic_config import (
    FIREWALL_LAMBDA,
    K,
    MC_SEED_BASE,
    MU_SHIFT_MAX,
    MU_SHIFT_MIN,
    MU_SHIFT_POINTS,
    N_MC_TRIALS,
    N_VETO,
    NOISE_LAMBDA,
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
    generate_systemic_population,
    generate_unbiased_population,
    perturb_context,
)
from systemic_metrics import (
    kendall_tau_rankings,
    policy_violation_rate,
    predictive_efficiency,
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


def run_algorithmic_firewall_experiment(rng: np.random.Generator) -> pd.DataFrame:
    """
    Exp 2: Fix lambda; inflate veto-group ML risk R ~ N(mu_shift, 0.05).
    Uses lambda=0.5 per spec; if violations are too rare at 0.5, FIREWALL_LAMBDA
    in systemic_config can be raised (see module docstring in run script output).
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
            _, v_k, _ = _evaluate(df, op, FIREWALL_LAMBDA)
            records.append({"mu_shift": mu, "operator": op, "v_k": v_k})

    return pd.DataFrame(records)


def run_noise_propagation_experiment(rng: np.random.Generator) -> pd.DataFrame:
    """
    Exp 3: Add Gaussian noise to Q; compare ranking stability vs baseline (sigma=0).
    """
    df = generate_unbiased_population(rng)
    sigmas = np.linspace(NOISE_SIGMA_MIN, NOISE_SIGMA_MAX, NOISE_SIGMA_POINTS)

    baseline = {}
    for op in ("linear", "min"):
        P0, _, _ = _evaluate(df, op, NOISE_LAMBDA)
        baseline[op] = P0

    records = []
    for sigma in sigmas:
        Q_pert = perturb_context(df["Q"].to_numpy(), rng, sigma)
        for op in ("linear", "min"):
            P_pert, _, _ = _evaluate(df, op, NOISE_LAMBDA, Q=Q_pert)
            tau = kendall_tau_rankings(baseline[op], P_pert)
            records.append({"sigma": sigma, "operator": op, "tau": tau})

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
    n_trials: int = N_MC_TRIALS, seed_base: int = MC_SEED_BASE
) -> pd.DataFrame:
    frames = []
    for trial in range(n_trials):
        trial_df = run_opportunity_cost_experiment(np.random.default_rng(seed_base + trial))
        trial_df["trial"] = trial
        frames.append(trial_df)
    return _aggregate_mc(pd.concat(frames, ignore_index=True), ["lambda", "operator"], ["v_k", "efficiency"])


def run_algorithmic_firewall_mc(
    n_trials: int = N_MC_TRIALS, seed_base: int = MC_SEED_BASE
) -> pd.DataFrame:
    frames = []
    for trial in range(n_trials):
        trial_df = run_algorithmic_firewall_experiment(np.random.default_rng(seed_base + trial))
        trial_df["trial"] = trial
        frames.append(trial_df)
    return _aggregate_mc(pd.concat(frames, ignore_index=True), ["mu_shift", "operator"], ["v_k"])


def run_noise_propagation_mc(
    n_trials: int = N_MC_TRIALS, seed_base: int = MC_SEED_BASE
) -> pd.DataFrame:
    frames = []
    for trial in range(n_trials):
        trial_df = run_noise_propagation_experiment(np.random.default_rng(seed_base + trial))
        trial_df["trial"] = trial
        frames.append(trial_df)
    return _aggregate_mc(pd.concat(frames, ignore_index=True), ["sigma", "operator"], ["tau"])


def run_all_systemic_experiments_mc(
    n_trials: int = N_MC_TRIALS, seed_base: int = MC_SEED_BASE
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        run_opportunity_cost_mc(n_trials, seed_base),
        run_algorithmic_firewall_mc(n_trials, seed_base),
        run_noise_propagation_mc(n_trials, seed_base),
    )


def run_all_systemic_experiments() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    return (
        run_opportunity_cost_experiment(rng),
        run_algorithmic_firewall_experiment(np.random.default_rng(SEED + 1)),
        run_noise_propagation_experiment(np.random.default_rng(SEED + 2)),
    )
