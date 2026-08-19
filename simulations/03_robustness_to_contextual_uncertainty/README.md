# Experiment 03 — Robustness to Contextual Uncertainty

Monte Carlo evaluation of CADEMAS-ML aggregation semantics under:
1. **λ sensitivity** — balance between predictive and contextual evidence
2. **Contextual noise** — $Q'=\mathrm{clip}(Q+\epsilon,0,1)$, $\epsilon\sim\mathcal{N}(0,\sigma_Q^2)$
3. **Population robustness** — alternative synthetic population configurations

## Convention

- Aggregation uses observed $Q'$; policy violation $V$ uses **true** $Q$.
- Same seeds (42–1041), $N=1000$, $K=100$, 1000 replications as Experiments 01–02.

## Run

```bash
cd simulations/03_robustness_to_contextual_uncertainty
python scripts/run_all.py           # all analyses
python scripts/run_all.py --refresh # recompute trials
python scripts/deep_validation.py   # diagnostic report
```

Individual scripts: `run_lambda_sensitivity.py`, `run_context_noise.py`, `run_population_robustness.py`.

## Outputs

| Path | Description |
|------|-------------|
| `results/lambda_sensitivity.csv` | Analysis 1 raw trials |
| `results/contextual_noise.csv` | Analysis 2 raw trials |
| `results/population_variants.csv` | Analysis 3 combined raw |
| `results/aggregated_results.csv` | Merged summaries |
| `results/diagnostic_report.md` | Deep validation report |
| `figures/` | Primary plots |
