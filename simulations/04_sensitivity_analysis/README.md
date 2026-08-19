# Experiment 04 — Sensitivity Analysis

Companion to manuscript §6.4.

## Research question
How stable are the resulting prioritization decisions under small
perturbations of model parameters (λ) and input scores (R)?

## Analyses
- **Analysis 1** — λ perturbation: ranking stability (Kendall τ, Jaccard)
  when λ shifts by ±0.025–0.05 around the reference λ=0.75.
- **Analysis 2** — Score uncertainty: ranking stability under unbiased
  Gaussian noise on R (σ_R ∈ {0, 0.02, 0.05, 0.10, 0.20}) at λ=0.75.
- **Analysis 3** — Stability summary: worst-case and mean stability per
  operator across both perturbation types.

## Usage
```bash
cd simulations
.venv/bin/python 04_sensitivity_analysis/scripts/run_all.py
```

## Output
- `results/diagnostic_report.md` — primary deliverable
- `figures/` — PDF + PNG plots
- `tables/` — LaTeX tables
