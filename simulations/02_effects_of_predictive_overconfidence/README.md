# Experiment 02 — Effects of predictive overconfidence

Manuscript reference: `first_round_v2/manuscriptR1V2.tex`, Section 6.2
(`\subsection{Effects of Predictive Overconfidence}`).

This folder contains the simulation, artifacts, and an LLM-facing report.
It does **not** draft manuscript text.

## Scientific objective

This subsection is **not** a comparison of which operator is better. It
demonstrates the **population-level consequences** of artificial predictive
overconfidence under different aggregation semantics (manuscript Section 4.4,
compensation / rank-reversal thresholds):

- how a systematic upward bias in $R$ on contextually weak cases changes
  Top-$K$ policy violations;
- whether that bias also changes the true predictive quality of the selected
  set (mean **original** $R$);
- how much the ranking moves relative to the unperturbed ranking.

## Overconfidence mechanism

For each Monte Carlo population, scores of contextually weak cases are shifted

$$
R_i' = \mathrm{clip}(R_i + \delta, 0, 1)
\qquad\text{iff}\qquad
Q_i \le q_{\mathrm{weak}}.
$$

$\delta$ is deterministic (not random). Cases with $Q_i > q_{\mathrm{weak}}$
are left unchanged. The default threshold $q_{\mathrm{weak}}=0.25$
includes all vetoes ($Q_i=0$) and standard cases with weak context.

The baseline population is the same generator as Experiment 01.

## Experimental design

For each Monte Carlo replication:

1. Generate a synthetic population of $N$ cases (same as Experiment 01).
2. For each $\lambda\in\{0.50, 0.75, 0.90\}$
   and each operator, store the unperturbed ranking $P_{\mathrm{ref}}=A(R,Q)$.
3. For each $\delta$, form $R'$ and $P=A(R',Q)$.
4. Select Top-$K$ with ties $(-P,-R',\mathrm{case\_id})$.
5. Compute $V$ on $\{Q_i=0\}$, $\bar{R}$ as the mean **original** $R$ of
   the selected set, Kendall $\tau(P_{\mathrm{ref}},P)$, and Jaccard Top-$K$.

Primary figures use $\lambda=0.75$. The other
$\lambda$ values are stored for robustness checks.

## Parameters

All defaults live in `simulations/common/config.py`.

| Parameter | Value |
|---|---|
| $N$ | 1000 |
| `veto_fraction` | 0.20 |
| Standard cases | 800 |
| Veto cases ($Q_i=0$) | 200 |
| Top-$K$ | 100 |
| $q_{\mathrm{weak}}$ | 0.25 |
| $\delta$ grid | (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3) |
| $\lambda$ (stored) | (0.5, 0.75, 0.9) |
| $\lambda$ (primary figures) | 0.75 |
| $R$ (standard) | Beta(2.0, 2.0) |
| $Q$ (standard) | Beta(2.0, 2.0) |
| $R$ (veto, adversarial) | Beta(8.0, 2.0) |
| Monte Carlo replications | 1000 |
| Seeds | 42 … 1041 |
| Confidence intervals | 95% percentiles |

## Metrics

- **Policy violation rate** $V=|\mathcal{T}_K\cap\mathcal{V}|/K$, $\mathcal{V}=\{i:Q_i=0\}$.
- **Predictive utility** $\bar{R}$ = mean original $R_i$ of the Top-$K$ selected from $P=A(R',Q)$.
- **Kendall $\tau$** between $P_{\mathrm{ref}}$ and $P$ (same trial, operator, $\lambda$).
- **Jaccard Top-$K$** between the $\delta=0$ set and the perturbed set.

## How to run

```bash
pip install -r simulations/requirements.txt
cd simulations/02_effects_of_predictive_overconfidence
python run.py
python run.py --refresh
python run.py --q-threshold 0.25
```

Monte Carlo trials are cached in `results/trials_raw.csv` keyed by a
fingerprint of the scientific parameters. Re-running without `--refresh`
rebuilds figures, tables, and documentation from the cache when the
fingerprint matches.

## Generated outputs

- `results/trials_raw.csv` — one row per (trial, $\lambda$, $\delta$, operator)
- `results/aggregated.csv` — mean, std, CI95% over all stored $\lambda$
- `results/aggregated_primary.csv` — subset $\lambda=0.75$
- `results/run_metadata.json` — seeds, threshold, fingerprint
- `figures/overconfidence_v_utility.{pdf,png}` — primary two-panel figure
- `figures/overconfidence_kendall_tau.{pdf,png}` — Kendall $\tau$ vs $\delta$
- Jaccard figure omitted (redundant with Kendall $\tau$, or no separation)
- `tables/table_predictive_overconfidence.tex` — $V$ and $\bar{R}$
- `tables/table_overconfidence_kendall.tex` — $\tau$ and Jaccard at representative $\delta$
- `captions.md`
- `results_narrative.md`

## Main conclusions from the results

Observed Monte Carlo behaviour (no ranking of operators as better/worse):

Policy violation rate $V$ at $\lambda=0.75$:
- $A_L$: 0.077 (95% CI [0.020, 0.150]) at $\delta=0.00$; 0.022 (95% CI [0.000, 0.130]) at $\delta=0.30$; interior maximum 0.251 (95% CI [0.080, 0.390]) at $\delta=0.10$; first mean increase after $\delta=0$ at $\delta=0.05$.
- $A_G$: 0.000 (95% CI [0.000, 0.000]) at $\delta=0.00$; 0.000 (95% CI [0.000, 0.000]) at $\delta=0.30$; no change relative to $\delta=0$ on this grid.
- $A_M$: 0.000 (95% CI [0.000, 0.000]) at $\delta=0.00$; 0.000 (95% CI [0.000, 0.000]) at $\delta=0.30$; no change relative to $\delta=0$ on this grid.

Predictive utility $\bar{R}$ (original $R$ of Top-$K$):
- $A_L$: 0.850 (95% CI [0.835, 0.866]) at $\delta=0.00$; 0.830 (95% CI [0.812, 0.847]) at $\delta=0.30$; interior maximum 0.880 (95% CI [0.862, 0.896]) at $\delta=0.10$; first mean increase after $\delta=0$ at $\delta=0.05$.
- $A_G$: 0.818 (95% CI [0.798, 0.837]) at $\delta=0.00$; 0.819 (95% CI [0.802, 0.837]) at $\delta=0.30$.
- $A_M$: 0.744 (95% CI [0.721, 0.766]) at $\delta=0.00$; 0.744 (95% CI [0.721, 0.766]) at $\delta=0.30$; no change relative to $\delta=0$ on this grid.

Kendall $\tau$ vs the $\delta=0$ ranking:
- $A_L$: 1.000 (95% CI [1.000, 1.000]) at $\delta=0.00$; 0.736 (95% CI [0.717, 0.757]) at $\delta=0.30$; mean decreases over the $\delta$ grid.
- $A_G$: 1.000 (95% CI [1.000, 1.000]) at $\delta=0.00$; 0.914 (95% CI [0.900, 0.928]) at $\delta=0.30$; mean decreases over the $\delta$ grid.
- $A_M$: 1.000 (95% CI [1.000, 1.000]) at $\delta=0.00$; 0.997 (95% CI [0.995, 0.999]) at $\delta=0.30$; mean decreases over the $\delta$ grid.
