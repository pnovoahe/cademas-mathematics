# §6.3 Robustness to Contextual Uncertainty — Summary for LLM

This document provides a self-contained description of Experiment 03 and its
integration into the manuscript, intended as context for an LLM agent working on
this section.

---

## Manuscript location

- **Section:** §6.3 `\subsection{Robustness to Contextual Uncertainty}`
  `\label{subsec:simulation-contextual-uncertainty}`
- **File:** `first_round_v2/manuscriptR1V2.tex` (lines ~1742–1828)
- **Appendix:** `\subsection{Population robustness variants}`
  `\label{app:population-robustness-variants}` (lines ~1997–2007)

---

## Experimental design

| Element | Value |
|---|---|
| Monte Carlo replications | 1000 |
| Population size | N = 1000 |
| Top-K | K = 100 |
| Seeds | 42–1041 (fixed, reproducible) |
| Operators | A_L (linear), A_G (weighted geometric), A_M (minimum) |
| Primary λ (noise sweep) | 0.75 |
| λ grid (sensitivity sweep) | {0.0, 0.1, …, 1.0} — 11 points |
| σ_Q grid (noise sweep) | {0.00, 0.05, 0.10, 0.15, 0.20} |
| Weak non-vetoes | 0 < Q_i ≤ 0.25 |
| Veto definition | Q_i = 0 |
| Contextual noise model | Q'_i = clip(Q_i + ε_i, 0, 1), ε_i ~ N(0, σ_Q²) |
| Aggregation input | Noisy Q'_i |
| Policy violation V evaluated on | True Q_i (ground-truth) |
| R̄ | Mean original predictive score R of Top-K |

**Key design choice:** V is evaluated against true Q, not observed Q'. This means
V measures real compliance failure caused by measurement error in the contextual
subsystem. This is distinct from §6.2 (overconfidence), which affected predictive
scores R systematically.

---

## Analysis 1 — λ sensitivity at σ_Q = 0 (baseline cross-check)

**Purpose:** Confirm that Experiment 03 at σ_Q = 0 reproduces Experiment 01 (§6.1) exactly.

**Result:** Cross-validation vs Experiment 01 at λ ∈ {0.50, 0.75, 0.90}:
max |V_01 − V_03| = 0.000000 (paired seeds). Perfect match.

**Selected results (baseline population, σ_Q = 0):**

| λ    | A_L V | A_G V | A_M V | A_L R̄ |
|------|-------|-------|-------|--------|
| 0.00 | 0.000 | 0.000 | 0.000 | 0.499  |
| 0.50 | 0.000 | 0.000 | 0.000 | 0.758  |
| 0.70 | 0.001 | 0.000 | 0.000 | 0.823  |
| 0.75 | 0.107 | 0.000 | 0.000 | 0.852  |
| 0.90 | 0.462 | 0.000 | 0.000 | 0.913  |
| 1.00 | 0.644 | 0.000 | 0.000 | 0.920  |

- A_G and A_M maintain V = 0 for all λ (zero-absorption property holds at σ_Q = 0).
- A_L: first V > 0 at λ ≈ 0.70; monotonically increasing thereafter.
- **Manuscript treatment:** One cross-reference sentence only. No figure or table in
  main text (would duplicate §6.1 panel b). Full λ-sweep figure
  (`figures/lambda_sensitivity.pdf`) demoted to appendix reference.

---

## Analysis 2 — Contextual noise sweep at λ = 0.75 (main text)

**Purpose:** Study how imperfect contextual observation degrades policy compliance.

**Mechanism — false-negative contextual errors:**
When ε_i > 0 pushes a true veto (Q_i = 0) to Q'_i > 0, the aggregation operator
no longer sees it as a veto, allowing it into Top-K. This is called a
**false-negative contextual error**. The result is that zero-absorption operators
(A_G, A_M), which guarantee V = 0 under exact Q, lose that guarantee under noise.

**Results table (λ = 0.75, baseline population):**

| σ_Q  | A_L V | A_G V | A_M V | A_L FN rate |
|------|-------|-------|-------|-------------|
| 0.00 | 0.077 | 0.000 | 0.000 | 0.000       |
| 0.05 | 0.096 | 0.000 | 0.000 | 0.063       |
| 0.10 | 0.116 | 0.001 | 0.000 | 0.090       |
| 0.15 | 0.136 | 0.016 | 0.000 | 0.117       |
| 0.20 | 0.156 | 0.046 | 0.002 | 0.145       |

FN rate = fraction of Top-K positions occupied by true vetoes (Q_i = 0) observed
with Q'_i > 0. For A_G, FN rate ≈ V (all violations are false-negative errors).
For A_M, FN rate stays near zero: even small positive Q'_i does not guarantee
a positive prioritization score because the minimum over all components remains
near zero.

**Noise threshold summary:**
- A_L: V ≥ 0.05 already at σ_Q = 0 (baseline V = 0.077); V ≥ 0.10 at σ_Q = 0.10.
- A_G: V > 0 first appears at σ_Q ≈ 0.10; reaches 0.046 at σ_Q = 0.20.
- A_M: essentially V = 0 across all tested σ_Q; reaches only 0.002 at σ_Q = 0.20.

**Manuscript artifacts (main text):**
- Figure: `figures/contextual_noise.pdf`
  - Panel (a): V(σ_Q) for A_L, A_G, A_M. y-axis: [−0.015, 0.3]. Legend only in (a).
  - Panel (b): R̄(σ_Q) for all operators. No legend (shared with panel a).
- Figure: `figures/false_negative_mechanism.pdf`
  - Single panel: false-negative veto rate in Top-K vs σ_Q at λ = 0.75.
  - y-axis: [−0.015, 0.3].
- Table: `tables/table_contextual_noise.tex` (label: `tab:contextual-noise`)
  - Columns: σ_Q, Operator, V, FN rate. Values: Monte Carlo means, 1000 reps.

---

## Analysis 3 — Population robustness (appendix)

**Purpose:** Test whether qualitative patterns hold under alternative synthetic
populations (different veto fractions, weak-context prevalence, R score overlap).

**Population variants (6 + baseline):**

| Scenario | Description |
|---|---|
| baseline | veto_fraction=0.20, standard betas |
| low_veto_frac | veto_fraction=0.05 |
| high_veto_frac | veto_fraction=0.10 |
| weak_heavy | More 0 < Q ≤ 0.25 cases (std_q_beta skewed) |
| weak_sparse | Fewer weak-context cases |
| low_r_separation | Veto cases have lower R separation |
| high_r_overlap | Veto cases strongly overlap with non-veto R distribution |

**λ-sweep results at σ_Q = 0 (A_L onset and end-point):**

| Scenario       | Onset λ (V > 0) | V at λ = 0.75 | V at λ = 1.0 |
|---|---|---|---|
| baseline       | 0.7 | 0.107 | 0.644 |
| high_r_overlap | 0.8 | 0.007 | 0.124 |
| high_veto_frac | 0.7 | 0.053 | 0.418 |
| low_r_separation | 0.7 | 0.053 | 0.400 |
| low_veto_frac  | 0.8 | 0.028 | 0.240 |
| weak_heavy     | 0.6 | 0.284 | 0.642 |
| weak_sparse    | 0.5 | 0.411 | 0.643 |

**σ_Q-sweep results at λ = 0.75 (V at σ_Q = 0.20):**

| Scenario       | V at σ_Q = 0.20 |
|---|---|
| baseline       | 0.156 |
| high_r_overlap | 0.011 |
| high_veto_frac | 0.071 |
| low_r_separation | 0.077 |
| low_veto_frac  | 0.034 |
| weak_heavy     | 0.342 |
| weak_sparse    | 0.456 |

**Key insight:** The qualitative ordering (A_M most robust → A_G intermediate →
A_L highest V) is preserved across all scenarios. Quantitative levels depend on
population: weak_heavy and weak_sparse produce substantially higher V because
weak non-vetoes (0 < Q_i ≤ 0.25) are at elevated risk of receiving noise-inflated
Q' that satisfies operator thresholds.

**Manuscript treatment:** Appendix `\ref{app:population-robustness-variants}`.
Text cross-references `figures/lambda_sensitivity.pdf` and
`figures/population_robustness_lambda.pdf` without embedding them as full figures.

---

## Prior conclusions and robustness

| Prior conclusion (Exp 01/02) | Status under Exp 03 |
|---|---|
| A_G, A_M maintain V = 0 for all λ at σ_Q = 0 | Confirmed (exact match with Exp 01) |
| A_L violations emerge above λ ≈ 0.70 | Confirmed (onset varies slightly by population) |
| Non-monotonic V(δ) under overconfidence is clipping + replacement | Not contradicted; different mechanism |
| Operator differences are structural, not seed artifacts | Confirmed across 1000 replications |

---

## Finding classification

### A — Essential (main text §6.3)
- At σ_Q = 0, A_G and A_M maintain V = 0 for all λ; A_L shows gradual violation onset.
- Contextual noise breaks zero-absorption: V increases with σ_Q for A_G and A_M
  via false-negative contextual errors (true vetoes with Q' > 0).
- V is evaluated on true Q; noise is stochastic measurement error, not adversarial.

### B — Supporting evidence (appendix / brief mention)
- Top-K composition decomposition (veto / weak-NV / normal counts).
- False-negative veto rate vs σ_Q (now in main text as second figure).
- Population-variant shifts in λ onset and noise sensitivity.
- Kendall τ vs λ = 0 baseline under λ sweep.

### C — Internal diagnostic only (not published)
- Observed-Q' violation rate (observed_veto_rate).
- Per-trial raw CSVs and baseline factorial interactions.
- Full cross-scenario overlay plots.

**Guardrails (enforced in manuscript text):**
- Do NOT claim compliance recovery at high noise.
- Do NOT claim global operator superiority.
- Weak non-vetoes (0 < Q_i ≤ q_weak) are not veto-compliant.

---

## Artifact index

| Artifact | Location | In manuscript |
|---|---|---|
| `contextual_noise.pdf` | `first_round_v2/figures/` | Yes — Fig. `fig:contextual-noise` |
| `false_negative_mechanism.pdf` | `first_round_v2/figures/` | Yes — Fig. `fig:false-negative-mechanism` |
| `table_contextual_noise.tex` | `first_round_v2/tables/` | Yes — Table `tab:contextual-noise` |
| `lambda_sensitivity.pdf` | `first_round_v2/figures/` (copy); `simulations/03_.../figures/` | Appendix ref only |
| `table_lambda_robustness.tex` | `first_round_v2/tables/` | Not in main text |
| `population_robustness_lambda.pdf` | `simulations/03_.../figures/` | Appendix ref only |
| `lambda_sensitivity.csv` | `simulations/03_.../results/` | Internal |
| `contextual_noise.csv` | `simulations/03_.../results/` | Internal |
| `population_variants.csv` | `simulations/03_.../results/` | Internal |
| `aggregated_results.csv` | `simulations/03_.../results/` | Internal |
| `diagnostic_report.md` | `simulations/03_.../results/` | Internal |

---

## Simulation scripts

| Script | Purpose |
|---|---|
| `scripts/run_lambda_sensitivity.py` | Analysis 1: λ sweep at σ_Q = 0 |
| `scripts/run_context_noise.py` | Analysis 2: σ_Q sweep at λ = 0.75 |
| `scripts/run_population_robustness.py` | Analysis 3: population variants |
| `scripts/run_all.py` | Orchestrates all three |
| `scripts/deep_validation.py` | Generates diagnostic figures and report |

All scripts use `.venv` at `simulations/.venv/`. Run from `simulations/`:
```bash
.venv/bin/python 03_robustness_to_contextual_uncertainty/scripts/run_all.py
.venv/bin/python 03_robustness_to_contextual_uncertainty/scripts/deep_validation.py
```

Figure y-axis limits (as of latest revision):
- `contextual_noise.pdf` panel (a) V(σ_Q): ylim = (−0.015, 0.3)
- `contextual_noise.pdf` panel (b) R̄(σ_Q): ylim = None (auto); no legend in panel (b)
- `false_negative_mechanism.pdf`: ylim = (−0.015, 0.3)
