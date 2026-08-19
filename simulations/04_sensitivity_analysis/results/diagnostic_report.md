# Experiment 04 — Diagnostic Report

Sensitivity analysis. Internal analysis for Section 6.4.

## 1. Methods recap

- **Analysis 1** — λ perturbation: λ ∈ {0.70, 0.725, 0.75, 0.775, 0.80}, σ_R=0, baseline population. Reference: P_ref at λ=0.75.
- **Analysis 2** — Score uncertainty: σ_R ∈ {0.00, 0.02, 0.05, 0.10, 0.20}, λ=0.75, baseline population. R'=clip(R+ε,0,1), ε~N(0,σ_R²), all cases.
- Primary metrics: Kendall τ, Top-K Jaccard (vs noiseless reference).
- Secondary metrics: ΔR̄, V (compliance, not primary focus).
- N=1000, K=100, 1000 Monte Carlo replications, seeds 42–1041.

## Baseline sanity checks

- linear: τ at λ_ref=1.000000 (expected ≈1.0), τ at σ_R=0: 1.000000 (expected ≈1.0), Jaccard at λ_ref=1.000000, Jaccard at σ_R=0: 1.000000
- geometric: τ at λ_ref=1.000000 (expected ≈1.0), τ at σ_R=0: 1.000000 (expected ≈1.0), Jaccard at λ_ref=1.000000, Jaccard at σ_R=0: 1.000000
- min: τ at λ_ref=1.000000 (expected ≈1.0), τ at σ_R=0: 1.000000 (expected ≈1.0), Jaccard at λ_ref=1.000000, Jaccard at σ_R=0: 1.000000

## 2. Analysis 1 — λ perturbation results

| λ | Operator | τ | Jaccard |
|---|---|---|---|
| 0.700 | linear | 0.9255 | 0.8064 |
| 0.700 | geometric | 0.9652 | 0.8945 |
| 0.700 | min | 1.0000 | 1.0000 |
| 0.725 | linear | 0.9639 | 0.8760 |
| 0.725 | geometric | 0.9827 | 0.9441 |
| 0.725 | min | 1.0000 | 1.0000 |
| 0.750 | linear | 1.0000 | 1.0000 |
| 0.750 | geometric | 1.0000 | 1.0000 |
| 0.750 | min | 1.0000 | 1.0000 |
| 0.775 | linear | 0.9662 | 0.8609 |
| 0.775 | geometric | 0.9830 | 0.9416 |
| 0.775 | min | 1.0000 | 1.0000 |
| 0.800 | linear | 0.9347 | 0.7425 |
| 0.800 | geometric | 0.9663 | 0.8861 |
| 0.800 | min | 1.0000 | 1.0000 |

## 3. Analysis 2 — Score uncertainty results

| σ_R | Operator | τ | Jaccard |
|---|---|---|---|
| 0.00 | linear | 1.0000 | 1.0000 |
| 0.00 | geometric | 1.0000 | 1.0000 |
| 0.00 | min | 1.0000 | 1.0000 |
| 0.02 | linear | 0.9442 | 0.8373 |
| 0.02 | geometric | 0.9663 | 0.8948 |
| 0.02 | min | 0.9727 | 0.9274 |
| 0.05 | linear | 0.8629 | 0.6589 |
| 0.05 | geometric | 0.9162 | 0.7620 |
| 0.05 | min | 0.9321 | 0.8382 |
| 0.10 | linear | 0.7390 | 0.4781 |
| 0.10 | geometric | 0.8371 | 0.5960 |
| 0.10 | min | 0.8677 | 0.7148 |
| 0.20 | linear | 0.5454 | 0.3356 |
| 0.20 | geometric | 0.7049 | 0.4021 |
| 0.20 | min | 0.7572 | 0.5405 |

## 4. Analysis 3 — Stability summary (mean and worst-case τ)

Averages exclude the reference point (λ=0.75 and σ_R=0) where τ=1 by construction.

| Operator | Mean τ (λ) | Worst τ (λ) | Mean τ (σ_R) | Worst τ (σ_R) |
|---|---|---|---|---|
| linear | 0.9476 | 0.9255 | 0.7729 | 0.5454 |
| geometric | 0.9743 | 0.9652 | 0.8561 | 0.7049 |
| min | 1.0000 | 1.0000 | 0.8824 | 0.7572 |

## 5. Research questions

### Q1: Which operators produce more stable rankings?

Under λ perturbation, highest mean τ: **min**.
Under score uncertainty, highest mean τ: **min**.

### Q2: Do compliance-preserving operators (A_G, A_M) sacrifice or improve stability?

Under λ perturbation: A_G has higher τ than A_L; A_M has higher τ than A_L.
Under score uncertainty: A_G has higher τ than A_L; A_M has higher τ than A_L.

### Q3: Is instability concentrated in ranking positions (τ) or Top-K membership (Jaccard)?

Compare τ and Jaccard at σ_R=0.20 (most extreme noise):

| Operator | τ at σ_R=0.20 | Jaccard at σ_R=0.20 |
|---|---|---|
| linear | 0.5454 | 0.3356 |
| geometric | 0.7049 | 0.4021 |
| min | 0.7572 | 0.5405 |

If Jaccard drops more than τ, instability is concentrated in Top-K membership.
If τ drops more (relative to 1.0), instability is spread across the full ranking.

## 6. Finding classification

### A — Essential for manuscript narrative

- Operator ordering by ranking stability under λ perturbation.
- Operator ordering by ranking stability under score uncertainty.
- Whether compliance-preserving operators (A_G, A_M) are more or less stable than A_L.

### B — Supporting evidence (appendix / brief mention)

- Worst-case τ and Jaccard at extreme perturbation values.
- Comparison of τ vs Jaccard degradation (ranking positions vs Top-K membership).
- ΔR̄ under both perturbation types.
- Secondary V values (compliance not primary focus).

### C — Internal diagnostic only

- Per-trial raw CSVs.
- σ_R × λ interaction (not computed; out of scope for this experiment).

**Guardrails:**
- Do NOT claim global operator superiority.
- V is secondary information; do not frame §6.4 as a compliance experiment.
- ΔR̄ values reflect noise, not systematic score inflation.

## 7. Artifacts

- `results/lambda_perturbation.csv`, `score_uncertainty.csv`
- `results/aggregated_lambda.csv`, `aggregated_score.csv`, `aggregated_stability_summary.csv`
- `figures/lambda_perturbation_stability.pdf`
- `figures/score_uncertainty_stability.pdf`
- `figures/stability_summary.pdf`
- `tables/table_lambda_stability.tex`, `table_score_stability.tex`
