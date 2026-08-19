"""Evaluation metrics for CADEMAS-ML simulations (manuscript R1V2, Section 5.4).

Tie-breaking for Top-K selection follows the manuscript protocol: primary key
is the aggregated score P, secondary key is the predictive score R, and
remaining ties are resolved by the case identifier.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import kendalltau, rankdata

FloatArray = NDArray[np.floating]
IntArray = NDArray[np.integer]


def top_k_indices(
    P: FloatArray,
    K: int,
    *,
    R: FloatArray | None = None,
    case_id: IntArray | None = None,
) -> IntArray:
    """Return indices of the Top-K alternatives after deterministic tie-breaks."""
    P = np.asarray(P, dtype=float)
    n = P.shape[0]
    if R is None:
        R = np.zeros(n, dtype=float)
    else:
        R = np.asarray(R, dtype=float)
    if case_id is None:
        case_id = np.arange(n, dtype=int)
    else:
        case_id = np.asarray(case_id, dtype=int)

    order = np.lexsort((case_id, -R, -P))
    return order[:K]


def policy_violation_rate(
    P: FloatArray,
    Q: FloatArray,
    K: int,
    *,
    R: FloatArray | None = None,
    case_id: IntArray | None = None,
) -> float:
    """V = |T_K ∩ V| / K, where V = {i : Q_i = 0}."""
    idx = top_k_indices(P, K, R=R, case_id=case_id)
    veto = np.asarray(Q, dtype=float)[idx] == 0.0
    return float(np.mean(veto))


def veto_preservation_rate(
    P: FloatArray,
    Q: FloatArray,
    K: int,
    *,
    R: FloatArray | None = None,
    case_id: IntArray | None = None,
) -> float:
    """VPR = 1 - V (contextual compliance)."""
    return 1.0 - policy_violation_rate(P, Q, K, R=R, case_id=case_id)


contextual_compliance = veto_preservation_rate


def predictive_utility(
    P: FloatArray,
    R: FloatArray,
    K: int,
    *,
    case_id: IntArray | None = None,
) -> float:
    """Mean predictive score of the selected Top-K alternatives."""
    idx = top_k_indices(P, K, R=R, case_id=case_id)
    return float(np.mean(np.asarray(R, dtype=float)[idx]))


def kendall_tau(P_ref: FloatArray, P: FloatArray) -> float:
    """Kendall rank correlation between two prioritization score vectors."""
    rank_ref = rankdata(-np.asarray(P_ref, dtype=float), method="average")
    rank_new = rankdata(-np.asarray(P, dtype=float), method="average")
    tau, _ = kendalltau(rank_ref, rank_new)
    return float(tau)


def topk_composition(
    Q_true: FloatArray,
    P: FloatArray,
    K: int,
    q_weak: float,
    *,
    R: FloatArray | None = None,
    case_id: IntArray | None = None,
) -> dict[str, float]:
    """Count and rate veto, weak-non-veto, and normal cases in Top-K.

    Classification uses **true** contextual scores ``Q_true``.
    """
    Q_true = np.asarray(Q_true, dtype=float)
    idx = top_k_indices(P, K, R=R, case_id=case_id)
    q_sel = Q_true[idx]
    is_veto = q_sel == 0.0
    is_weak_nv = (q_sel > 0.0) & (q_sel <= float(q_weak))
    is_normal = q_sel > float(q_weak)
    n_veto = int(is_veto.sum())
    n_weak_nv = int(is_weak_nv.sum())
    n_normal = int(is_normal.sum())
    n_weak = n_veto + n_weak_nv
    return {
        "n_veto_topk": float(n_veto),
        "n_weak_nv_topk": float(n_weak_nv),
        "n_normal_topk": float(n_normal),
        "frac_veto_topk": n_veto / K,
        "frac_weak_nv_topk": n_weak_nv / K,
        "frac_normal_topk": n_normal / K,
        "e_weak_context": n_weak / K,
    }


def observed_veto_rate(
    P: FloatArray,
    Q_observed: FloatArray,
    K: int,
    *,
    R: FloatArray | None = None,
    case_id: IntArray | None = None,
) -> float:
    """Diagnostic: fraction of Top-K with observed Q' = 0 (Class C only)."""
    return policy_violation_rate(P, Q_observed, K, R=R, case_id=case_id)


def false_negative_veto_rate(
    P: FloatArray,
    Q_true: FloatArray,
    Q_observed: FloatArray,
    K: int,
    *,
    R: FloatArray | None = None,
    case_id: IntArray | None = None,
) -> float:
    """Fraction of Top-K that are true vetoes (Q=0) but observed Q' > 0."""
    idx = top_k_indices(P, K, R=R, case_id=case_id)
    q_true = np.asarray(Q_true, dtype=float)[idx]
    q_obs = np.asarray(Q_observed, dtype=float)[idx]
    mask = (q_true == 0.0) & (q_obs > 0.0)
    return float(np.mean(mask))


def jaccard_top_k(
    P_ref: FloatArray,
    P: FloatArray,
    K: int,
    *,
    R_ref: FloatArray | None = None,
    R: FloatArray | None = None,
    case_id: IntArray | None = None,
) -> float:
    """Jaccard overlap between two Top-K sets."""
    idx_ref = set(top_k_indices(P_ref, K, R=R_ref, case_id=case_id).tolist())
    idx_new = set(top_k_indices(P, K, R=R, case_id=case_id).tolist())
    union = idx_ref | idx_new
    if not union:
        return 1.0
    return len(idx_ref & idx_new) / len(union)
