"""Dataset generation for systemic CADEMAS evaluation."""

import numpy as np
import pandas as pd

from systemic_config import INTER_Q, N_INTER, N_STD, N_VETO


def generate_systemic_population(
    rng: np.random.Generator,
    veto_r: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    80% Standard (R, Q ~ U(0,1)), 20% Vetoed (Q=0, R set externally or ~ U(0,1)).
    """
    std = pd.DataFrame(
        {
            "R": rng.uniform(0.0, 1.0, N_STD),
            "Q": rng.uniform(0.0, 1.0, N_STD),
            "group": "std",
        }
    )

    if veto_r is None:
        veto_r = rng.uniform(0.0, 1.0, N_VETO)
    else:
        veto_r = np.clip(np.asarray(veto_r, dtype=float), 0.0, 1.0)

    veto = pd.DataFrame(
        {
            "R": veto_r,
            "Q": np.zeros(N_VETO),
            "group": "veto",
        }
    )

    df = pd.concat([std, veto], ignore_index=True)
    perm = rng.permutation(len(df))
    return df.iloc[perm].reset_index(drop=True)


def generate_unbiased_population(rng: np.random.Generator) -> pd.DataFrame:
    """n=1000 with R, Q ~ U(0,1) for noise-propagation experiment (no sub-groups)."""
    n = N_STD + N_VETO
    return pd.DataFrame(
        {
            "R": rng.uniform(0.0, 1.0, n),
            "Q": rng.uniform(0.0, 1.0, n),
            "group": "std",
        }
    )


def generate_rank_stability_population(
    rng: np.random.Generator, model: str = "uniform"
) -> pd.DataFrame:
    """Population for rank-stability experiment: uniform or Beta(2,2) scores."""
    n = N_STD + N_VETO
    if model == "uniform":
        return generate_unbiased_population(rng)
    if model == "beta":
        return pd.DataFrame(
            {
                "R": np.clip(rng.beta(2, 2, n), 0.0, 1.0),
                "Q": np.clip(rng.beta(2, 2, n), 0.0, 1.0),
                "group": "std",
            }
        )
    raise ValueError(f"Unknown population model: {model}")


def assign_veto_r_from_normal(
    rng: np.random.Generator, mu_shift: float, sigma: float = 0.05
) -> np.ndarray:
    """ML overconfidence on veto group only: R ~ N(mu_shift, sigma), clipped to [0, 1]."""
    return np.clip(rng.normal(mu_shift, sigma, N_VETO), 0.0, 1.0)


def generate_intermediate_population(
    rng: np.random.Generator,
    inter_r: np.ndarray | None = None,
    q_inter: float = INTER_Q,
) -> pd.DataFrame:
    """
    80% Standard (R, Q ~ U(0,1)), 20% Intermediate (Q fixed, high predictive R).
    """
    std = pd.DataFrame(
        {
            "R": rng.uniform(0.0, 1.0, N_STD),
            "Q": rng.uniform(0.0, 1.0, N_STD),
            "group": "std",
        }
    )

    if inter_r is None:
        inter_r = np.clip(rng.normal(0.9, 0.05, N_INTER), 0.0, 1.0)
    else:
        inter_r = np.clip(np.asarray(inter_r, dtype=float), 0.0, 1.0)

    inter = pd.DataFrame(
        {
            "R": inter_r,
            "Q": np.full(N_INTER, float(q_inter)),
            "group": "inter",
        }
    )

    df = pd.concat([std, inter], ignore_index=True)
    perm = rng.permutation(len(df))
    return df.iloc[perm].reset_index(drop=True)


def perturb_context(
    Q: np.ndarray, rng: np.random.Generator, sigma: float
) -> np.ndarray:
    """Q' = clip(Q + epsilon, 0, 1), epsilon ~ N(0, sigma)."""
    if sigma == 0.0:
        return Q.copy()
    noise = rng.normal(0.0, sigma, size=Q.shape)
    return np.clip(Q + noise, 0.0, 1.0)


def perturb_context_uniform(
    Q: np.ndarray, rng: np.random.Generator, sigma: float
) -> np.ndarray:
    """Q' = clip(Q + epsilon, 0, 1), epsilon ~ U(-sigma, sigma)."""
    if sigma == 0.0:
        return Q.copy()
    noise = rng.uniform(-sigma, sigma, size=Q.shape)
    return np.clip(Q + noise, 0.0, 1.0)
