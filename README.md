# Companion Repository for “Aggregation Semantics in Cooperative Automated Decision-Making”

This repository provides reproducible code for the simulation studies and case study reported in the paper *Aggregation Semantics for Prioritization in Cooperative Automated Decision Making*.

## Contents

| Path | Description |
|------|-------------|
| `src/common/` | Shared configuration, aggregation operators, population generators, metrics, and publication plotting |
| `src/01_policy_compliance_under_contextual_vetoes/` | Monte Carlo study of contextual veto compliance and ranking agreement |
| `src/02_sensitivity_analysis/` | Sensitivity to Gaussian noise on predictive scores; intervention-capacity sweep |
| `src/03_attrition_case_study/` | Employee-attrition case study using H2O MOJO models and fuzzy contextual rules |

Each experiment folder contains a `run.py` entry point, precomputed results (where applicable), and generated figures.

## Requirements

- Python 3.11+
- See [`src/requirements.txt`](src/requirements.txt) for Python dependencies
- Experiment 03 additionally requires Java (for H2O) and the MOJO model artifacts under `src/03_attrition_case_study/models/`

## Setup

```bash
python -m venv src/.venv
src/.venv/bin/pip install -r src/requirements.txt
```

## Run

From the repository root:

```bash
cd src

# Experiment 01 — policy compliance under contextual vetoes
.venv/bin/python 01_policy_compliance_under_contextual_vetoes/run.py

# Experiment 02 — sensitivity to predictive-score noise
.venv/bin/python 02_sensitivity_analysis/run.py

# Experiment 03 — attrition case study (full pipeline; requires H2O)
.venv/bin/python 03_attrition_case_study/run.py

# Experiment 03 — figures only (uses cached operator scores)
.venv/bin/python 03_attrition_case_study/run.py --figures-only
```

Use `--refresh` on Experiments 01 and 02 to ignore the Monte Carlo cache and regenerate trials.

## Reproducibility

- Monte Carlo experiments use 1000 registered random seeds (`42`–`1041`); defaults are in `src/common/config.py`.
- Experiment-specific README files document outputs, figures, and parameters.
- Regenerated aggregated CSVs and figures are written under each experiment’s `results/` and `figures/` directories.

## License

See [`LICENSE`](LICENSE).
