"""Experiment logic for CADEMAS simulation (RQ1, RQ2, RQ3)."""

import numpy as np
import pandas as pd

from config import (
    ALPHA_MAX,
    ALPHA_MIN,
    ALPHA_POINTS,
    HOMOGENEOUS_N,
    HOMOGENEOUS_Q_LEVELS,
    K,
    LAMBDA_FIXED,
    LAMBDA_MIN,
    LAMBDA_STEP,
    OPERATORS,
    RQ2_LAMBDA_PANELS,
    RQ2_OPERATORS,
    SEED,
)
from data import generate_dataset, generate_homogeneous_subpopulation
from metrics import kendall_tau_preservation, policy_violation_rate
from operators import aggregate


def _evaluate_operator(
    df: pd.DataFrame, operator: str, lam: float, R: np.ndarray | None = None
) -> tuple[float, float]:
    """Compute Kendall's tau and V_K for a given operator and lambda."""
    R_arr = df["R"].to_numpy() if R is None else R
    Q_arr = df["Q"].to_numpy()
    groups = df["group"].to_numpy()

    P = aggregate(operator, R_arr, Q_arr, lam)
    tau = kendall_tau_preservation(R_arr, P)
    v_k = policy_violation_rate(P, groups, K)
    return tau, v_k


def run_pareto_experiment(df: pd.DataFrame) -> dict:
    """
    RQ1: Sweep lambda from 0.01 to 0.99 and compute tau vs V_K
    for linear and geometric operators. Min/max are lambda-independent.
    """
    lambda_vals = np.arange(LAMBDA_MIN, 1.0, LAMBDA_STEP)

    linear_results = []
    geometric_results = []

    for lam in lambda_vals:
        tau_l, v_l = _evaluate_operator(df, "linear", lam)
        tau_g, v_g = _evaluate_operator(df, "geometric", lam)
        linear_results.append({"lambda": lam, "tau": tau_l, "v_k": v_l})
        geometric_results.append({"lambda": lam, "tau": tau_g, "v_k": v_g})

    tau_min, v_min = _evaluate_operator(df, "min", LAMBDA_FIXED)
    tau_max, v_max = _evaluate_operator(df, "max", LAMBDA_FIXED)

    return {
        "lambda_vals": lambda_vals,
        "linear": pd.DataFrame(linear_results),
        "geometric": pd.DataFrame(geometric_results),
        "min": {"tau": tau_min, "v_k": v_min},
        "max": {"tau": tau_max, "v_k": v_max},
    }


def run_overconfidence_experiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    RQ2: Distort risk scores R_new = R^alpha and measure V_K across lambda panels.
    Sweeps alpha in [1.0, 0.1] for each lambda in RQ2_LAMBDA_PANELS so that
    linear vulnerability emerges at high predictive weight.
    """
    alpha_vals = np.linspace(ALPHA_MAX, ALPHA_MIN, ALPHA_POINTS)
    records = []

    for lam in RQ2_LAMBDA_PANELS:
        for alpha in alpha_vals:
            R_distorted = np.power(df["R"].to_numpy(), alpha)
            for operator in RQ2_OPERATORS:
                _, v_k = _evaluate_operator(df, operator, lam, R=R_distorted)
                records.append(
                    {
                        "alpha": alpha,
                        "lambda": lam,
                        "operator": operator,
                        "v_k": v_k,
                    }
                )

    return pd.DataFrame(records)


def validate_proposition_2(rng: np.random.Generator) -> pd.DataFrame:
    """
    RQ3: Within homogeneous contextual strata, Kendall's tau must be 1.00
    for all strictly monotonic operators (empirical validation of Proposition 2).
    """
    records = []

    for q_level in HOMOGENEOUS_Q_LEVELS:
        for operator in OPERATORS:
            sub_df = generate_homogeneous_subpopulation(
                HOMOGENEOUS_N, q_level, rng, operator=operator
            )
            tau, _ = _evaluate_operator(sub_df, operator, LAMBDA_FIXED)
            records.append(
                {
                    "q_level": q_level,
                    "operator": operator,
                    "tau": tau,
                }
            )

    return pd.DataFrame(records)


def print_proposition_2_confirmation(results: pd.DataFrame) -> None:
    """Print console confirmation for Proposition 2 (Q = 0.5 stratum)."""
    q_mid = results[results["q_level"] == 0.5]

    print("\n" + "=" * 60)
    print("RQ3: Empirical Validation of Proposition 2")
    print("Contextual Homogeneity (Q_i = 0.5 for all alternatives)")
    print("=" * 60)

    for _, row in q_mid.iterrows():
        print(f"  {row['operator']:12s}  Kendall's tau = {row['tau']:.2f}")

    all_unity = np.allclose(q_mid["tau"].to_numpy(), 1.0)
    if all_unity:
        print(
            "\nCONFIRMED: Proposition 2 holds empirically. "
            "Under homogeneous context, all monotonic operators "
            "preserve the predictive rank ordering exactly (tau = 1.00)."
        )
    else:
        print("\nWARNING: Some operators did not achieve tau = 1.00.")

    print("=" * 60 + "\n")


def run_all_experiments() -> tuple[pd.DataFrame, dict, pd.DataFrame, pd.DataFrame]:
    """Run the full simulation pipeline and return all results."""
    rng = np.random.default_rng(SEED)
    df = generate_dataset(rng)

    pareto = run_pareto_experiment(df)
    overconfidence = run_overconfidence_experiment(df)
    homogeneity = validate_proposition_2(rng)

    return df, pareto, overconfidence, homogeneity
