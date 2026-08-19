# Diagnostics for Experiment 02

Internal analysis of the non-monotonic $V(\delta)$ curve for $A_L$ at
$\lambda=0.75$. Does not change the formal experiment or the manuscript.

```bash
python analyze_non_monotonicity.py
python analyze_non_monotonicity.py --refresh
```

Uses the same seeds and `generate_population` as `../run.py`. Monte Carlo
records are cached in `results/`.

Read `non_monotonicity_analysis.md` first.
