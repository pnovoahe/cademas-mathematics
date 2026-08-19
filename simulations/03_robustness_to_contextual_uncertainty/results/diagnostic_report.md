# Experiment 03 — Diagnostic Report

Robustness to contextual uncertainty. Internal analysis for Section 6.3.

## 1. Methods recap

- $\lambda$ grid: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
- $\sigma_Q$ grid: [0.0, 0.05, 0.1, 0.15, 0.2]; primary $\lambda=0.75$
- Weak non-vetoes: $0<Q_i\le 0.25$
- Aggregation uses noisy $Q'$; **$V$ uses true $Q$** (ground-truth vetoes in Top-$K$).
- Population scenarios: baseline + 6 variants (veto fraction, weak-$Q$ density, $R$ separation).

## 2. Analysis 1 — $\lambda$ sensitivity ($\sigma_Q=0$, baseline)

**$A_L$ first $\lambda$ with $V>0$:** 0.7.

| $\lambda$ | $A_L$ $V$ | $A_G$ $V$ | $A_M$ $V$ | $A_L$ $\bar{R}$ |
|---|---------|---------|---------|------------|
| 0.00 | 0.000 | 0.000 | 0.000 | 0.499 |
| 0.50 | 0.000 | 0.000 | 0.000 | 0.758 |
| 0.70 | 0.001 | 0.000 | 0.000 | 0.823 |
| 0.75 | 0.107 | 0.000 | 0.000 | 0.852 |
| 0.90 | 0.462 | 0.000 | 0.000 | 0.913 |
| 1.00 | 0.644 | 0.000 | 0.000 | 0.920 |

Violations emerge **gradually** for $A_L$: $V=0$ for $\lambda<0.7$, then increases monotonically toward $\lambda=1$. $A_G$ and $A_M$ maintain $V=0$ over the full grid (zero-absorption at $\sigma_Q=0$).

### Cross-validation vs Experiment 01

Max $|V_{\mathrm{01}}-V_{\mathrm{03}}|$ at $\lambda\in\{0.50,0.75,0.90\}$: **0.000000** (paired seeds).

## 3. Analysis 2 — Contextual noise ($\lambda=0.75$, baseline)

| $\sigma_Q$ | $A_L$ $V$ | $A_G$ $V$ | $A_M$ $V$ | $A_L$ fn rate |
|---|---------|---------|---------|-------------|
| 0.00 | 0.077 | 0.000 | 0.000 | 0.000 |
| 0.05 | 0.096 | 0.000 | 0.000 | 0.063 |
| 0.10 | 0.116 | 0.001 | 0.000 | 0.090 |
| 0.15 | 0.136 | 0.016 | 0.000 | 0.117 |
| 0.20 | 0.156 | 0.046 | 0.002 | 0.145 |

- **linear**: $V\ge 0.05$ already at $\sigma_Q=0$ (baseline $V=0.077$); $V\ge 0.10$ at $\sigma_Q=0.1$.
- **geometric**: $V\ge 0.05$ at $\sigma_Q=>0.20$; $V\ge 0.10$ at $\sigma_Q=>0.20$.
- **min**: $V\ge 0.05$ at $\sigma_Q=>0.20$; $V\ge 0.10$ at $\sigma_Q=>0.20$.

At $\sigma_Q=0$, $A_G$ and $A_M$ preserve $V=0$ (observed: $A_G$ $V=0.000$). Under noise, zero-absorption fails when true vetoes receive $Q'>0$: $A_G$ $V$ rises to 0.046 at $\sigma_Q=0.2$.

## 4. Analysis 3 — Population robustness

| Scenario | Onset $\lambda$ ($V>0$) | $V$ at $\lambda=0.75$ (interp.) | $V$ at $\lambda=1.0$ |
|---|---|---|---|
| baseline | 0.7 | 0.107 | 0.644 |
| high_r_overlap | 0.8 | 0.007 | 0.124 |
| high_veto_frac | 0.7 | 0.053 | 0.418 |
| low_r_separation | 0.7 | 0.053 | 0.400 |
| low_veto_frac | 0.8 | 0.028 | 0.240 |
| weak_heavy | 0.6 | 0.284 | 0.642 |
| weak_sparse | 0.5 | 0.411 | 0.643 |

**$V(\sigma_Q)$ at $\lambda=0.75$** (end-point at max noise):

- baseline: $V=0.156$ at $\sigma_Q=0.2$
- high_r_overlap: $V=0.011$ at $\sigma_Q=0.2$
- high_veto_frac: $V=0.071$ at $\sigma_Q=0.2$
- low_r_separation: $V=0.077$ at $\sigma_Q=0.2$
- low_veto_frac: $V=0.034$ at $\sigma_Q=0.2$
- weak_heavy: $V=0.342$ at $\sigma_Q=0.2$
- weak_sparse: $V=0.456$ at $\sigma_Q=0.2$

## 5. Which prior conclusions remain robust?

1. **Exp 01 (§6.1):** Zero-absorption operators preserve $V=0$ at $\sigma_Q=0$ across all $\lambda$ — **confirmed**.
2. **Exp 01:** $A_L$ violations emerge above ~$\lambda=0.70$ — **confirmed** (onset varies slightly by population).
3. **Exp 02 (§6.2):** Non-monotonic $V(\delta)$ under overconfidence is a separate mechanism (clipping + replacement); not contradicted by contextual noise at $\delta=0$.
4. **Operator semantics:** Differences are structural (zero-absorption vs linear trade-off), not artifacts of one seed — **confirmed** across 1000 replications.

## 6. Research questions

| Question | Summary |
|---|---|
| Which prior conclusions remain robust? | See §5; core veto-preservation and $\lambda$-threshold patterns replicate. |
| Which results are sensitive to $\lambda$? | $A_L$ only; $V$ and $\bar{R}$ increase with $\lambda$; composition shifts toward vetoes. |
| How much noise before compliance deteriorates? | Operator-dependent; geometric/min remain at $V=0$ until $\sigma_Q>0$. |
| Do operators differ in noise sensitivity? | Yes: $A_G$/$A_M$ fail via false-negative vetoes; $A_L$ less sensitive at moderate $\sigma_Q$. |
| Mechanisms preserved across populations? | Qualitative patterns yes; quantitative onsets and $V$ levels vary. |

## 7. Finding classification

### A — Essential for manuscript narrative

- At $\sigma_Q=0$, $A_G$ and $A_M$ maintain $V=0$ for all $\lambda$; $A_L$ exhibits gradual violation onset.
- Contextual noise breaks zero-absorption: $V$ increases with $\sigma_Q$ for $A_G$ and $A_M$ when true vetoes receive $Q'>0$.
- $V$ is evaluated on true $Q$; noise represents measurement error, not adversarial manipulation.

### B — Supporting evidence (appendix / brief mention)

- Top-$K$ composition decomposition (veto / weak-NV / normal counts).
- False-negative veto rate vs $\sigma_Q$.
- Population-variant shifts in $\lambda$ onset and noise sensitivity.
- Kendall $\tau$ vs $\lambda=0$ baseline under $\lambda$ sweep.

### C — Internal diagnostic only

- Observed-$Q'$ violation rate (`observed_veto_rate`).
- Per-trial raw CSVs and baseline factorial interactions.
- Full cross-scenario overlay plots.

**Guardrails:** Do not claim compliance recovery at high noise or global operator superiority. Weak non-vetoes are not veto-compliant.

## 8. Manuscript readiness (§6.3)

**Main text:** Analysis 1 (selected $\lambda$ points + dense sweep figure), Analysis 2 primary curve at $\lambda=0.75$, one paragraph on population robustness (qualitative).

**Appendix:** Full population-variant tables, false-negative decomposition, Kendall stability.

**Do not publish:** Class C artifacts unless needed for reproducibility.

## 9. Artifacts

- `results/lambda_sensitivity.csv`, `contextual_noise.csv`, `population_variants.csv`
- `results/aggregated_results.csv`
- `figures/lambda_sensitivity.pdf`, `contextual_noise.pdf`, `population_robustness_*.pdf`
- `figures/diagnostics/false_negative_mechanism.pdf`
