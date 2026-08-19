# Shared simulation infrastructure

This package provides the reusable building blocks for the Monte Carlo
experiments reported in `first_round_v2/manuscriptR1V2.tex`.

Individual experiments live in numbered folders under `simulations/` and
**must import all parameters from `config.py`**. Do not hard-code population
sizes, seeds, λ grids, or figure settings in experiment scripts.

## Modules

| Module | Role |
|---|---|
| `config.py` | Population (`VETO_FRACTION` → group sizes), Monte Carlo, λ grids, overconfidence ($\delta$, $q_{\mathrm{weak}}$), figure export flags, operator colours |
| `aggregators.py` | Linear, weighted geometric, and minimum operators with a common interface |
| `generators.py` | Beta-distributed $(R_i,Q_i)$ populations, contextual-veto injection, predictive overconfidence, and contextual noise $Q'=\mathrm{clip}(Q+\epsilon,0,1)$ |
| `metrics.py` | Policy violation, VPR/compliance, predictive utility, Top-$K$ composition, Kendall τ, Jaccard Top-$K$ |
| `trial_runner.py` | Shared trial loops for Experiment 03 (λ and $\sigma_Q$ grids) |
| `plotting.py` | Publication style (Okabe–Ito / Helvetica) plus bar, line, box, and square two-panel helpers |
| `utils.py` | Seeds, mean/std/CI aggregation, fingerprints, LaTeX table writer |

## Operator definitions

- Linear: $A_L(R,Q)=\lambda R+(1-\lambda)Q$
- Weighted geometric: $A_G(R,Q)=R^\lambda Q^{1-\lambda}$, with $A_G=0$ if $R=0$ or $Q=0$
- Minimum: $A_M(R,Q)=\min(R,Q)$

## Metric definitions (manuscript Section 5.4)

- Policy violation rate: $V=|\mathcal{T}_K\cap\mathcal{V}|/K$, $\mathcal{V}=\{i:Q_i=0\}$
- Veto preservation / contextual compliance: $\mathrm{VPR}=1-V$
- Predictive utility: $\bar{R}=\frac{1}{K}\sum_{i\in\mathcal{T}_K}R_i$
- Kendall $\tau$ and Jaccard Top-$K$ for ranking stability

Top-$K$ ties are broken by $(-P,-R,\mathrm{case\_id})$.

## Population sizes

`VETO_FRACTION` (default $0.20$) determines how many cases have $Q_i=0$:

```python
N_VETO = n_veto_from_fraction(VETO_FRACTION)
N_STD = N_CASES - N_VETO
```

Supported fractions for later checks: $0.05$, $0.10$, $0.20$. Experiments may override the fraction from the command line (for example `--veto-fraction 0.10`) without editing `config.py`. The default $0.20$ is the setting reported in Section 6.1.

The default veto predictive scores $R\sim\mathrm{Beta}(8,2)$ are an **adversarial** choice: high predictive evidence against a hard contextual constraint.

## Predictive overconfidence (Experiment 02)

Deterministic upward bias applied only to contextually weak cases:

```python
R_prime = apply_predictive_overconfidence(R, Q, delta, q_threshold=Q_WEAK_THRESHOLD)
# R'_i = clip(R_i + δ, 0, 1)  if Q_i <= q_threshold, else R_i
```

Defaults in `config.py`:

| Parameter | Default |
|---|---|
| `DELTA_VALUES` | $(0.00, 0.05, \\ldots, 0.30)$ |
| `DELTA_TABLE_VALUES` | $(0.00, 0.10, 0.20, 0.30)$ |
| `Q_WEAK_THRESHOLD` | $0.25$ |
| `OVERCONFIDENCE_LAMBDAS` | $\\{0.50, 0.75, 0.90\\}$ |
| `OVERCONFIDENCE_PRIMARY_LAMBDA` | $0.75$ |

Experiment 02 lives in `simulations/02_effects_of_predictive_overconfidence/`. Monte Carlo trials are cached by fingerprint; figure-style reruns do not recompute the 1000 replications.

## Contextual uncertainty (Experiment 03)

Stochastic contextual perturbation (measurement error, not adversarial):

```python
Q_prime = apply_contextual_noise(Q, rng, sigma_q)
# Q'_i = clip(Q_i + ε_i, 0, 1),  ε_i ~ N(0, σ_Q²)
```

**Convention:** aggregation uses $Q'$; policy violation $V$ uses **true** $Q$ (ground-truth vetoes in Top-$K$).

| Parameter | Default |
|---|---|
| `LAMBDA_ROBUSTNESS_VALUES` | $(0.0, 0.1, \ldots, 1.0)$ |
| `SIGMA_Q_VALUES` | $(0.00, 0.05, 0.10, 0.15, 0.20)$ |
| `CONTEXT_NOISE_PRIMARY_LAMBDA` | $0.75$ |
| `POPULATION_SCENARIOS` | baseline + 6 robustness variants |

Experiment 03 lives in `simulations/03_robustness_to_contextual_uncertainty/`.

## Adding a new experiment

1. Create `simulations/NN_short_name/` with `run.py`, `results/`, `figures/`, `tables/`.
2. Import generators, aggregators, metrics, and config from `common`.
3. Store both trial-level and aggregated outputs.
4. Generate captions and manuscript snippets from the numerical results only.

## Installation

```bash
pip install -r simulations/requirements.txt
```
