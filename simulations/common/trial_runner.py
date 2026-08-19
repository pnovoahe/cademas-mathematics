"""Shared trial evaluation for Experiment 03 (contextual uncertainty).

Aggregation uses observed contextual scores ``Q_obs``; policy compliance
metrics use true ``Q_true`` unless noted otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from common.aggregators import aggregate
from common.config import (
    N_CASES,
    OPERATORS,
    POPULATION_SCENARIOS,
    Q_WEAK_THRESHOLD,
    TOP_K,
    n_std_from_fraction,
    n_veto_from_fraction,
)
from common.generators import Population, apply_contextual_noise, generate_population
from common.metrics import (
    false_negative_veto_rate,
    jaccard_top_k,
    kendall_tau,
    observed_veto_rate,
    policy_violation_rate,
    predictive_utility,
    topk_composition,
)
from common.utils import trial_seed

COMPOSITION_METRICS = (
    "n_veto_topk",
    "n_weak_nv_topk",
    "n_normal_topk",
    "frac_veto_topk",
    "frac_weak_nv_topk",
    "frac_normal_topk",
    "e_weak_context",
)

PRIMARY_METRICS = (
    "policy_violation_rate",
    "veto_preservation_rate",
    "predictive_utility",
    *COMPOSITION_METRICS,
    "observed_veto_rate",
    "false_negative_veto_rate",
    "kendall_tau",
    "jaccard_top_k",
)


@dataclass(frozen=True)
class PopulationConfig:
    """Population generator parameters for one scenario."""

    name: str = "baseline"
    veto_fraction: float = 0.20
    std_r_beta: tuple[float, float] = (2.0, 2.0)
    std_q_beta: tuple[float, float] = (2.0, 2.0)
    veto_r_beta: tuple[float, float] = (8.0, 2.0)

    @property
    def n_veto(self) -> int:
        return n_veto_from_fraction(self.veto_fraction, N_CASES)

    @property
    def n_std(self) -> int:
        return n_std_from_fraction(self.veto_fraction, N_CASES)

    @classmethod
    def from_scenario(cls, name: str) -> PopulationConfig:
        if name not in POPULATION_SCENARIOS:
            raise KeyError(f"Unknown scenario: {name}")
        spec = POPULATION_SCENARIOS[name]
        return cls(
            name=name,
            veto_fraction=float(spec["veto_fraction"]),
            std_r_beta=tuple(spec["std_r_beta"]),  # type: ignore[arg-type]
            std_q_beta=tuple(spec["std_q_beta"]),  # type: ignore[arg-type]
            veto_r_beta=tuple(spec["veto_r_beta"]),  # type: ignore[arg-type]
        )


def generate_scenario_population(
    trial_idx: int,
    pop_config: PopulationConfig,
) -> tuple[Population, np.random.Generator]:
    rng = np.random.default_rng(trial_seed(trial_idx))
    pop = generate_population(
        rng,
        n_std=pop_config.n_std,
        n_veto=pop_config.n_veto,
        std_r_beta=pop_config.std_r_beta,
        std_q_beta=pop_config.std_q_beta,
        veto_r_beta=pop_config.veto_r_beta,
    )
    return pop, rng


def evaluate_condition(
    pop: Population,
    q_obs: np.ndarray,
    *,
    operator: str,
    lam: float,
    q_weak: float = Q_WEAK_THRESHOLD,
    p_ref: np.ndarray | None = None,
) -> dict[str, float]:
    """Evaluate one operator at (λ, Q_obs) on a fixed population."""
    P = aggregate(operator, pop.R, q_obs, lam)

    v = policy_violation_rate(P, pop.Q, TOP_K, R=pop.R, case_id=pop.case_id)
    comp = topk_composition(
        pop.Q, P, TOP_K, q_weak, R=pop.R, case_id=pop.case_id
    )
    out: dict[str, float] = {
        "policy_violation_rate": v,
        "veto_preservation_rate": 1.0 - v,
        "predictive_utility": predictive_utility(P, pop.R, TOP_K, case_id=pop.case_id),
        **comp,
        "observed_veto_rate": observed_veto_rate(
            P, q_obs, TOP_K, R=pop.R, case_id=pop.case_id
        ),
        "false_negative_veto_rate": false_negative_veto_rate(
            P, pop.Q, q_obs, TOP_K, R=pop.R, case_id=pop.case_id
        ),
    }

    if p_ref is not None:
        out["kendall_tau"] = kendall_tau(p_ref, P)
        out["jaccard_top_k"] = jaccard_top_k(
            p_ref, P, TOP_K, R_ref=pop.R, R=pop.R, case_id=pop.case_id
        )
    else:
        out["kendall_tau"] = float("nan")
        out["jaccard_top_k"] = float("nan")

    return out


def run_lambda_grid_trial(
    trial_idx: int,
    pop_config: PopulationConfig,
    lambda_values: Iterable[float],
    *,
    sigma_q: float = 0.0,
    q_weak: float = Q_WEAK_THRESHOLD,
) -> list[dict]:
    """Analysis 1 / population λ sweep at fixed σ_Q."""
    pop, rng = generate_scenario_population(trial_idx, pop_config)
    q_obs = apply_contextual_noise(pop.Q, rng, sigma_q)

    p_ref_by_op: dict[str, np.ndarray] = {
        op: aggregate(op, pop.R, q_obs, 0.0) for op in OPERATORS
    }

    records: list[dict] = []
    for lam in lambda_values:
        for operator in OPERATORS:
            metrics = evaluate_condition(
                pop,
                q_obs,
                operator=operator,
                lam=float(lam),
                q_weak=q_weak,
                p_ref=p_ref_by_op[operator],
            )
            records.append(
                {
                    "trial": trial_idx,
                    "seed": trial_seed(trial_idx),
                    "analysis": "lambda_sensitivity",
                    "scenario": pop_config.name,
                    "sigma_q": sigma_q,
                    "lambda": float(lam),
                    "operator": operator,
                    **metrics,
                }
            )
    return records


def run_sigma_q_grid_trial(
    trial_idx: int,
    pop_config: PopulationConfig,
    sigma_q_values: Iterable[float],
    *,
    lam: float,
    q_weak: float = Q_WEAK_THRESHOLD,
) -> list[dict]:
    """Analysis 2 / population σ_Q sweep at fixed λ."""
    pop, _rng = generate_scenario_population(trial_idx, pop_config)

    q_clean = pop.Q.copy()
    p_ref_by_op: dict[str, np.ndarray] = {
        op: aggregate(op, pop.R, q_clean, lam) for op in OPERATORS
    }

    records: list[dict] = []
    for sigma_q in sigma_q_values:
        noise_rng = np.random.default_rng(
            trial_seed(trial_idx) + int(round(float(sigma_q) * 10000))
        )
        q_obs = apply_contextual_noise(pop.Q, noise_rng, float(sigma_q))
        for operator in OPERATORS:
            metrics = evaluate_condition(
                pop,
                q_obs,
                operator=operator,
                lam=lam,
                q_weak=q_weak,
                p_ref=p_ref_by_op[operator],
            )
            records.append(
                {
                    "trial": trial_idx,
                    "seed": trial_seed(trial_idx),
                    "analysis": "context_noise",
                    "scenario": pop_config.name,
                    "sigma_q": float(sigma_q),
                    "lambda": lam,
                    "operator": operator,
                    **metrics,
                }
            )
    return records
