# Experiment 01 — Policy compliance under contextual vetoes

Manuscript reference: `first_round_v2/manuscriptR1V2.tex`, Section 6.1
(`\subsection{Policy Compliance under Contextual Vetoes}`).

## Scientific objective

This subsection is **not** a comparison of which operator is better. It
demonstrates the **population-level consequences** of the theoretical
contextual-veto property (manuscript Section 4 / zero absorption):

- when policy violations emerge under compensatory (linear) aggregation;
- when contextual vetoes remain preserved;
- how those outcomes depend on $\lambda$.

## Adversarial veto population

The veto group is intentionally assigned high predictive scores
($R_i\sim\mathrm{Beta}(8.0, 2.0)$, $Q_i=0$) in order to create the most
challenging scenario for policy preservation. This allows the aggregation
semantics to be evaluated under conditions where predictive evidence strongly
conflicts with contextual constraints.

## Experimental design

For each Monte Carlo replication:

1. Generate a synthetic population of $N$ cases.
2. Set a fraction `veto_fraction` of cases to $Q_i=0$ with high $R_i$.
3. Apply the three aggregation operators on the **same** population.
4. Select the Top-$K$ set with deterministic tie-breaking $(-P,-R,\mathrm{case\_id})$.
5. Compute $V$, $\mathrm{VPR}=1-V$, and predictive utility $\bar{R}$
   (the last two are stored for reuse; only $V$ is presented in Section 6.1).

The manuscript uses a single three-panel figure:

- **Panel (a).** Illustrative $(R,Q)$ scatter (trial~0).
- **Panel (b).** Boxplots of $P_i$ by operator at $\lambda\in\{0.50,0.75,0.90\}$.
- **Panel (c).** Dense sweep $V(\lambda)$ on $[0,1]$, which shows the
  transition from no violations to systematic violations.

## Parameters

All defaults live in `simulations/common/config.py`. Group sizes are derived
from `veto_fraction` (default $0.20$). Supported values for
later checks: (0.05, 0.1, 0.2). This run used
`veto_fraction=0.20`.

| Parameter | Value |
|---|---|
| $N$ | 1000 |
| `veto_fraction` | 0.20 |
| Standard cases | 800 |
| Veto cases ($Q_i=0$) | 200 |
| Top-$K$ | 100 |
| $R$ (standard) | Beta(2.0, 2.0) |
| $Q$ (standard) | Beta(2.0, 2.0) |
| $R$ (veto, adversarial) | Beta(8.0, 2.0) |
| Monte Carlo replications | 1000 |
| Seeds | 42 … 1041 |
| $\lambda$ (Figure 6) | (0.5, 0.75, 0.9) |
| $\lambda$ (Figure 7) | 21 points on $[0.0,1.0]$ |
| Confidence intervals | 95% percentiles |

## Metrics

- **Policy violation rate** $V=|\mathcal{T}_K\cap\mathcal{V}|/K$, $\mathcal{V}=\{i:Q_i=0\}$ — reported in Section 6.1.
- **Contextual compliance / VPR** $1-V$ — computed, mentioned in text, not plotted.
- **Predictive utility** $\bar{R}$ — computed and stored; **not** discussed in Section 6.1.

## How to run

```bash
pip install -r simulations/requirements.txt
cd simulations/01_policy_compliance_under_contextual_vetoes
python run.py
python run.py --refresh
python run.py --veto-fraction 0.10   # optional extra scenario; not used in the paper by default
```

## Generated outputs

- `results/trials_raw.csv` — one row per (trial, $\lambda$, operator); includes $V$, VPR, $\bar{R}$
- `results/aggregated_dense.csv` — mean, std, CI95% over the dense $\lambda$ grid
- `results/aggregated_barplot.csv` — subset $\lambda\in\{0.50,0.75,0.90\}$
- `results/run_metadata.json` — seeds, `veto_fraction`, fingerprint
- `figures/policy_compliance_overview.{pdf,png}` — Section 6.1 composite figure
- `captions.md`
- `manuscript_snippets.md`
- `results_narrative.md`

## Main conclusions from the results

Observed Monte Carlo behaviour (no ranking of operators as better/worse):

Linear operator — violations emerge as $\lambda$ increases:
- $\lambda=0.50$: $V=0.000 (95% CI [0.000, 0.000])$
- $\lambda=0.75$: $V=0.077 (95% CI [0.020, 0.150])$
- $\lambda=0.90$: $V=0.462 (95% CI [0.370, 0.550])$
Linear mean $V$ first becomes strictly positive at $\lambda=0.70$.

Weighted geometric operator: $V=0$ at the representative $\lambda$ values and throughout the dense sweep, consistent with zero absorption.
Minimum operator: $V=0$ at the representative $\lambda$ values and throughout the dense sweep, consistent with zero absorption.

Contextual compliance is $\mathrm{VPR}=1-V$ and is not plotted.
Predictive utility $\bar{R}$ is stored in the CSV files for later sections and is not discussed here.
