"""Metrics for systemic CADEMAS evaluation."""

import numpy as np
from scipy.stats import kendalltau, rankdata


def top_k_indices(P: np.ndarray, K: int) -> np.ndarray:
    return np.argsort(-np.asarray(P, dtype=float))[:K]


def policy_violation_rate(P: np.ndarray, groups: np.ndarray, K: int) -> float:
    idx = top_k_indices(P, K)
    return float(np.mean(np.asarray(groups)[idx] == "veto"))


def predictive_efficiency(
    P: np.ndarray,
    R: np.ndarray,
    groups: np.ndarray,
    K: int,
) -> float:
    """
    Mean original risk R among Standard cases that entered the Top-K tier.
    Higher values indicate the Top-K cohort retains high-risk standard cases.
    """
    idx = top_k_indices(P, K)
    mask = np.asarray(groups)[idx] == "std"
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.asarray(R, dtype=float)[idx][mask]))


def kendall_tau_rankings(P_baseline: np.ndarray, P_perturbed: np.ndarray) -> float:
    """Rank correlation between two prioritization score vectors."""
    rank_base = rankdata(-np.asarray(P_baseline, dtype=float), method="average")
    rank_pert = rankdata(-np.asarray(P_perturbed, dtype=float), method="average")
    tau, _ = kendalltau(rank_base, rank_pert)
    return float(tau)
