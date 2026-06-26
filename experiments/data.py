"""Synthetic dataset generation for CADEMAS simulation."""

import numpy as np
import pandas as pd

from config import N_SAFE, N_STD, N_VETO


def generate_dataset(rng: np.random.Generator) -> pd.DataFrame:
    """
    Generate n alternatives with three sub-populations:
    - X_std: R, Q ~ U(0,1)
    - X_veto: R ~ Beta(8,2), Q = 0
    - X_safe: R ~ Beta(2,8), Q = 1
    """
    std = pd.DataFrame(
        {
            "R": rng.uniform(0.0, 1.0, N_STD),
            "Q": rng.uniform(0.0, 1.0, N_STD),
            "group": "std",
        }
    )

    veto = pd.DataFrame(
        {
            "R": rng.beta(8, 2, N_VETO),
            "Q": np.zeros(N_VETO),
            "group": "veto",
        }
    )

    safe = pd.DataFrame(
        {
            "R": rng.beta(2, 8, N_SAFE),
            "Q": np.ones(N_SAFE),
            "group": "safe",
        }
    )

    df = pd.concat([std, veto, safe], ignore_index=True)
    df = df.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1))).reset_index(
        drop=True
    )
    return df


def generate_homogeneous_subpopulation(
    n_sub: int, q_const: float, rng: np.random.Generator, operator: str | None = None
) -> pd.DataFrame:
    """
    Generate alternatives with identical contextual score Q = q_const.

    For min/max operators, R is sampled in the region where P = R
    (strict monotonicity regime required by Proposition 2):
    - min:      R in (0, q)  so that min(R, q) = R
    - max:      R in (q, 1)  so that max(R, q) = R
    - linear/geometric: R in (0, 1)
    """
    eps = 1e-4

    if operator == "min":
        upper = max(q_const - eps, eps)
        R = rng.uniform(eps, upper, n_sub)
    elif operator == "max":
        lower = min(q_const + eps, 1.0 - eps)
        R = rng.uniform(lower, 1.0 - eps, n_sub)
    else:
        R = rng.uniform(0.0, 1.0, n_sub)

    return pd.DataFrame(
        {
            "R": R,
            "Q": np.full(n_sub, q_const),
            "group": "homogeneous",
        }
    )
