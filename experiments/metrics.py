"""Ranking and policy compliance metrics."""

import numpy as np
from scipy.stats import kendalltau, rankdata


def kendall_tau_preservation(R: np.ndarray, P: np.ndarray) -> float:
    """
    Kendall's tau between the pure ML ranking (by R descending)
    and the aggregated ranking (by P descending).
    """
    rank_r = rankdata(-np.asarray(R, dtype=float), method="average")
    rank_p = rankdata(-np.asarray(P, dtype=float), method="average")
    tau, _ = kendalltau(rank_r, rank_p)
    return float(tau)


def policy_violation_rate(
    P: np.ndarray, groups: np.ndarray, K: int
) -> float:
    """
    Proportion of Top-K alternatives belonging to the veto sub-population.
    V_K = 0 indicates perfect institutional compliance.
    """
    P = np.asarray(P, dtype=float)
    groups = np.asarray(groups)
    top_k_idx = np.argsort(-P)[:K]
    return float(np.mean(groups[top_k_idx] == "veto"))
