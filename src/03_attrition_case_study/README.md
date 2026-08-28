# Experiment 03 — Attrition case-study scoring

This experiment reproduces the CADEMAS-ML case-study scoring pipeline on
100 employee cases and exports prioritization scores for the seven operator
configurations used in the manuscript:

- `A_L(0.10)`, `A_L(0.50)`, `A_L(0.90)`
- `A_G(0.10)`, `A_G(0.50)`, `A_G(0.90)`
- `A_M`

## Inputs

- `data/cases_attrition_100.csv` (feature-level dataset; `;` separator)
- `models/*.zip` (H2O MOJOs for cw/fin/od/hr units)
- `models/model_definitions.json` (AUC values for weighting)
- `context/context_digital_transformation.json` (fuzzy rules)

## Context aggregation choice

The context file uses:

- `upskilling_capacity`: `AVERAGE`
- `career_scalability`: `PRODUCT`
- `investment_horizon`: `PRODUCT`
- final `logic.op`: `AND` (Gödel min of the four high-level criteria)

A zero on any high-level criterion yields \(Q=0\) (contextual veto).

## Outputs

Generated under `results/`:

- `cases_attrition_100_operator_scores.csv`
- `run_metadata.json`

Generated under `figures/`:

- `attrition_case_overview.{pdf,png}` — (a) $(R,Q)$ scatter; (b) seven-operator violins with Top-$K=10$ cutoffs.
- `attrition_rank_bump.{pdf,png}` — rank trajectories of the Top-10 under $A_L(0.5)$ across the seven operators.
- `attrition_rank_bump_delta.{pdf,png}` — same trajectories; non-baseline tracks show $+/-$ rank change vs $A_L(0.5)$ (solid series ball if unchanged).

Main columns in the CSV:

- `Case_ID`, `attrition`
- `R`, `Q`
- `P_AL_0.10`, `P_AL_0.50`, `P_AL_0.90`
- `P_AG_0.10`, `P_AG_0.50`, `P_AG_0.90`
- `P_AM`

Plus audit columns: `cw_prob`, `fin_prob`, `od_prob`, `hr_prob`.

## Run (using simulations venv)

From `src/`:

```bash
.venv/bin/python 03_attrition_case_study/run.py
.venv/bin/python 03_attrition_case_study/run.py --figures-only
```

## Runtime requirements

- `h2o` Python package in `src/.venv`
- Java 17+ installed on the system
- Sibling repository path:
  - `../cademas-app/app/fuzzy_context.py`
