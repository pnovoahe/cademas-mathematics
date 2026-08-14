"""Metrics for systemic CADEMAS evaluation."""

import numpy as np
from scipy.stats import kendalltau, rankdata


def top_k_indices(P: np.ndarray, K: int) -> np.ndarray:
    return np.argsort(-np.asarray(P, dtype=float))[:K]


def policy_violation_rate(P: np.ndarray, groups: np.ndarray, K: int) -> float:
    idx = top_k_indices(P, K)
    return float(np.mean(np.asarray(groups)[idx] == "veto"))


def group_acceptance_rate(
    P: np.ndarray, groups: np.ndarray, K: int, group_name: str = "inter"
) -> float:
    """Fraction of the Top-K tier belonging to ``group_name``."""
    idx = top_k_indices(P, K)
    return float(np.mean(np.asarray(groups)[idx] == group_name))


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


def top_k_jaccard(P_baseline: np.ndarray, P_perturbed: np.ndarray, K: int) -> float:
    """Jaccard overlap between Top-K tiers induced by baseline and perturbed scores."""
    idx_base = set(top_k_indices(P_baseline, K))
    idx_pert = set(top_k_indices(P_perturbed, K))
    union = idx_base | idx_pert
    if not union:
        return 1.0
    return len(idx_base & idx_pert) / len(union)
