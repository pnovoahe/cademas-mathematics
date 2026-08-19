"""Synthetic population generators for CADEMAS-ML simulations.

Predictive and contextual scores are sampled from Beta distributions on
[0, 1], as specified in manuscript R1V2, Section 5.2. Contextual vetoes
are injected by setting Q_i = 0 for a designated subset of cases.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from common.config import (
    N_STD,
    N_VETO,
    STD_Q_BETA,
    STD_R_BETA,
    VETO_R_BETA,
)

FloatArray = NDArray[np.floating]
IntArray = NDArray[np.integer]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class Population:
    """A synthetic CADEMAS-ML decision population."""

    R: FloatArray
    Q: FloatArray
    case_id: IntArray
    is_veto: BoolArray

    @property
    def n(self) -> int:
        return int(self.R.shape[0])


def _sample_beta(
    rng: np.random.Generator,
    params: tuple[float, float],
    size: int,
) -> FloatArray:
    alpha, beta = params
    return np.clip(rng.beta(alpha, beta, size=size), 0.0, 1.0)


def generate_population(
    rng: np.random.Generator,
    *,
    n_std: int = N_STD,
    n_veto: int = N_VETO,
    std_r_beta: tuple[float, float] = STD_R_BETA,
    std_q_beta: tuple[float, float] = STD_Q_BETA,
    veto_r_beta: tuple[float, float] = VETO_R_BETA,
) -> Population:
    """Generate a mixed population of standard cases and contextual vetoes.

    Standard cases have independent R ~ Beta(α_R, β_R) and Q ~ Beta(α_Q, β_Q).
    Veto cases have Q = 0. By default their predictive scores follow a
    high-mean Beta (see ``VETO_R_BETA``), which is an adversarial design:
    predictive evidence strongly conflicts with the contextual constraint.
    Group sizes are typically derived from ``VETO_FRACTION`` in ``config``.
    Cases are randomly permuted so that group membership is not encoded by
    index order.
    """
    r_std = _sample_beta(rng, std_r_beta, n_std)
    q_std = _sample_beta(rng, std_q_beta, n_std)
    r_veto = _sample_beta(rng, veto_r_beta, n_veto)
    q_veto = np.zeros(n_veto, dtype=float)

    R = np.concatenate([r_std, r_veto])
    Q = np.concatenate([q_std, q_veto])
    is_veto = np.concatenate(
        [np.zeros(n_std, dtype=bool), np.ones(n_veto, dtype=bool)]
    )
    case_id = np.arange(n_std + n_veto, dtype=int)

    perm = rng.permutation(n_std + n_veto)
    return Population(
        R=R[perm],
        Q=Q[perm],
        case_id=case_id[perm],
        is_veto=is_veto[perm],
    )


def inject_contextual_vetoes(
    Q: FloatArray,
    indices: NDArray[np.integer] | None = None,
    *,
    n_veto: int | None = None,
    rng: np.random.Generator | None = None,
) -> FloatArray:
    """Return a copy of Q with selected entries set to zero.

    If ``indices`` is omitted, ``n_veto`` cases are chosen uniformly at random.
    """
    Q_out = np.asarray(Q, dtype=float).copy()
    if indices is None:
        if n_veto is None or rng is None:
            raise ValueError("Provide either indices, or both n_veto and rng.")
        indices = rng.choice(Q_out.size, size=n_veto, replace=False)
    Q_out[np.asarray(indices, dtype=int)] = 0.0
    return Q_out


def apply_contextual_noise(
    Q: FloatArray,
    rng: np.random.Generator,
    sigma_q: float,
) -> FloatArray:
    """Return Q' = clip(Q + ε, 0, 1) with ε ~ N(0, σ_Q²).

    When ``sigma_q`` is zero, returns a copy of ``Q`` unchanged.
    """
    Q_out = np.asarray(Q, dtype=float).copy()
    if float(sigma_q) <= 0.0:
        return Q_out
    eps = rng.normal(0.0, float(sigma_q), size=Q_out.shape)
    return np.clip(Q_out + eps, 0.0, 1.0)


def apply_score_uncertainty(
    R: FloatArray,
    rng: np.random.Generator,
    sigma_r: float,
) -> FloatArray:
    """Return R' = clip(R + ε, 0, 1) with ε ~ N(0, σ_R²). Unbiased noise.

    Applied to all cases (unlike predictive overconfidence in Exp 02 which
    targeted only contextually weak cases with a deterministic shift δ).
    When ``sigma_r`` is zero, returns a copy of ``R`` unchanged.
    """
    R_out = np.asarray(R, dtype=float).copy()
    if float(sigma_r) <= 0.0:
        return R_out
    eps = rng.normal(0.0, float(sigma_r), size=R_out.shape)
    return np.clip(R_out + eps, 0.0, 1.0)


def apply_predictive_overconfidence(
    R: FloatArray,
    Q: FloatArray,
    delta: float,
    *,
    q_threshold: float,
) -> FloatArray:
    """Return R' = clip(R + δ, 0, 1) on cases with Q <= ``q_threshold``.

    Cases with Q above the threshold are left unchanged. The shift ``delta``
    is deterministic (not random). A threshold of 0 includes only vetoes;
    the default ``Q_WEAK_THRESHOLD`` also covers contextually weak non-vetoes.
    """
    R_out = np.asarray(R, dtype=float).copy()
    weak = np.asarray(Q, dtype=float) <= float(q_threshold)
    R_out[weak] = np.clip(R_out[weak] + float(delta), 0.0, 1.0)
    return R_out
