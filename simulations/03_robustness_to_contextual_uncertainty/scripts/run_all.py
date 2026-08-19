#!/usr/bin/env python3
"""Orchestrate all Experiment 03 analyses and merge aggregated results."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
EXP_DIR = SCRIPTS_DIR.parent
SIM_DIR = EXP_DIR.parent
if str(SIM_DIR) not in sys.path:
    sys.path.insert(0, str(SIM_DIR))

import pandas as pd

from common.utils import ensure_dirs, utc_now_iso, write_json  # noqa: E402


def _run(script: str, refresh: bool) -> None:
    cmd = [sys.executable, str(SCRIPTS_DIR / script)]
    if refresh:
        cmd.append("--refresh")
    print(f"\n=== Running {script} ===")
    subprocess.run(cmd, check=True)


def merge_aggregated(results_dir: Path) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for path, analysis in (
        (results_dir / "lambda_sensitivity_aggregated.csv", "lambda_sensitivity"),
        (results_dir / "contextual_noise_aggregated.csv", "context_noise"),
        (results_dir / "population_variants_aggregated.csv", "population_robustness"),
    ):
        if path.exists():
            df = pd.read_csv(path)
            df["analysis_source"] = analysis
            parts.append(df)
    if not parts:
        raise FileNotFoundError("No aggregated CSVs found; run analyses first.")
    merged = pd.concat(parts, ignore_index=True, sort=False)
    merged.to_csv(results_dir / "aggregated_results.csv", index=False)
    return merged


def write_readme(exp_dir: Path) -> None:
    text = """# Experiment 03 — Robustness to Contextual Uncertainty

Monte Carlo evaluation of CADEMAS-ML aggregation semantics under:
1. **λ sensitivity** — balance between predictive and contextual evidence
2. **Contextual noise** — $Q'=\\mathrm{clip}(Q+\\epsilon,0,1)$, $\\epsilon\\sim\\mathcal{N}(0,\\sigma_Q^2)$
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
"""
    (exp_dir / "README.md").write_text(text, encoding="utf-8")


def write_captions(exp_dir: Path) -> None:
    text = """# Figure captions (Experiment 03)

## lambda_sensitivity
Policy violation rate $V$ and mean original predictive score $\\bar{R}$ versus aggregation weight $\\lambda$ at $\\sigma_Q=0$ ($N=1000$, $K=100$, 1000 replications).

## contextual_noise
$V$ and $\\bar{R}$ versus contextual noise $\\sigma_Q$ at $\\lambda=0.75$.

## population_robustness_lambda / population_robustness_noise
$A_L$ policy violation across population scenarios: full $\\lambda$ sweep ($\\sigma_Q=0$) and full $\\sigma_Q$ sweep ($\\lambda=0.75$).
"""
    (exp_dir / "captions.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--skip-population", action="store_true")
    args = parser.parse_args()

    _run("run_lambda_sensitivity.py", args.refresh)
    _run("run_context_noise.py", args.refresh)
    if not args.skip_population:
        _run("run_population_robustness.py", args.refresh)

    dirs = ensure_dirs(EXP_DIR)
    merged = merge_aggregated(dirs["results"])
    write_json(
        dirs["results"] / "run_metadata.json",
        {
            "experiment": "03_robustness_to_contextual_uncertainty",
            "timestamp_utc": utc_now_iso(),
            "n_aggregated_rows": int(len(merged)),
        },
    )
    write_readme(EXP_DIR)
    write_captions(EXP_DIR)
    print(f"\n[done] merged {len(merged)} aggregated rows")


if __name__ == "__main__":
    main()
