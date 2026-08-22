# Shared simulation infrastructure

This package provides the reusable building blocks for the Monte Carlo
experiments reported in `first_round_v3/manuscriptR1V3.tex`.

Individual experiments live in numbered folders under `simulations/` and
**must import all parameters from `config.py`**. Do not hard-code population
sizes, seeds, λ grids, or figure settings in experiment scripts.

## Modules

| Module | Role |
|---|---|
| `config.py` | Population (`VETO_FRACTION` → group sizes), Monte Carlo, λ grids, σ_R sensitivity grid, figure export flags, operator colours |
| `aggregators.py` | Linear, weighted geometric, and minimum operators with a common interface |
| `generators.py` | Beta-distributed $(R_i,Q_i)$ populations, contextual-veto injection, score uncertainty on $R$, contextual noise on $Q$ |
| `metrics.py` | Policy violation, VPR/compliance, predictive utility, Top-$K$ composition, Kendall τ (full / Top-$K$ union), Jaccard Top-$K$ |
| `plotting.py` | Publication style (Okabe–Ito / Helvetica) plus bar, line, box, overview, and sensitivity helpers |
| `utils.py` | Seeds, mean/std/CI aggregation, fingerprints, LaTeX table writer |

## Experiments

| Folder | Focus |
|---|---|
| `01_policy_compliance_under_contextual_vetoes` | Veto compliance and $V(\lambda)$; ranking stability vs $\lambda=0.5$ |
| `02_sensitivity_analysis` | Gaussian noise on $R$ only; $V$, τ, Jaccard vs clean; pairwise operator agreement |

## Operator definitions

- Linear: $A_L(R,Q)=\lambda R+(1-\lambda)Q$
- Weighted geometric: $A_G(R,Q)=R^\lambda Q^{1-\lambda}$, with $A_G=0$ if $R=0$ or $Q=0$
- Minimum: $A_M(R,Q)=\min(R,Q)$

## Metric definitions

- Policy violation rate: $V=|\mathcal{T}_K\cap\mathcal{V}|/K$, $\mathcal{V}=\{i:Q_i=0\}$
- Veto preservation / contextual compliance: $\mathrm{VPR}=1-V$
- Predictive utility: $\bar{R}=\frac{1}{K}\sum_{i\in\mathcal{T}_K}R_i$
- Kendall $\tau$ / Jaccard Top-$K$ for ranking stability

Top-$K$ ties are broken by $(-P,-R,\mathrm{case\_id})$.
