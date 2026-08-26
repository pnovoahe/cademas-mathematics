# Experiment 02 — Sensitivity to predictive-score noise

Gaussian noise on $R$ only; $Q$ fixed as in Experiment 01.

Figures:

- `sensitivity_r_noise_overview` — $(R',Q)$ scatters and $V$/τ/Jaccard bars
- `sensitivity_operator_agreement` — pairwise Kendall / Jaccard heatmaps
- `sensitivity_v_by_k` — $V(K)$ for $K=10,20,\ldots,100$ at four $\sigma_R$ (2×2 + panel e)

## Run

```bash
cd simulations && .venv/bin/python 02_sensitivity_analysis/run.py
```

Use `--refresh` to ignore the Monte Carlo cache.
