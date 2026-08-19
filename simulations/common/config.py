"""Global configuration for CADEMAS-ML simulations (manuscript R1V2).

All experiment scripts must import parameters from this module. Do not
hard-code population sizes, seeds, lambda grids, or figure settings in
individual ``run.py`` files.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SIMULATIONS_DIR = Path(__file__).resolve().parents[1]
COMMON_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Population
# ---------------------------------------------------------------------------
N_CASES = 1000
TOP_K = 100

# Fraction of cases with Q_i = 0. The manuscript reports 0.20; 0.05 and 0.10
# can be selected from experiment CLIs without changing this module's logic.
VETO_FRACTION = 0.20
VETO_FRACTIONS_SUPPORTED: tuple[float, ...] = (0.05, 0.10, 0.20)


def n_veto_from_fraction(fraction: float, n_cases: int = N_CASES) -> int:
    """Return the veto-group size implied by ``fraction`` of ``n_cases``."""
    if fraction < 0.0 or fraction > 1.0:
        raise ValueError(f"veto_fraction must be in [0, 1], got {fraction}.")
    return int(round(n_cases * fraction))


def n_std_from_fraction(fraction: float, n_cases: int = N_CASES) -> int:
    return n_cases - n_veto_from_fraction(fraction, n_cases)


N_VETO = n_veto_from_fraction(VETO_FRACTION)
N_STD = n_std_from_fraction(VETO_FRACTION)

if N_STD + N_VETO != N_CASES:
    raise ValueError("N_STD + N_VETO must equal N_CASES.")

# ---------------------------------------------------------------------------
# Score distributions (Beta, unit interval)
# ---------------------------------------------------------------------------
# Standard cases: independent R and Q, as in manuscript Section 5.2.
STD_R_BETA: tuple[float, float] = (2.0, 2.0)
STD_Q_BETA: tuple[float, float] = (2.0, 2.0)
# Veto group: Q_i = 0 with intentionally high predictive scores so that
# prediction strongly conflicts with the contextual constraint (adversarial).
VETO_R_BETA: tuple[float, float] = (8.0, 2.0)

# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------
MC_SEED_BASE = 42
N_MONTE_CARLO = 1000
SEEDS: list[int] = list(range(MC_SEED_BASE, MC_SEED_BASE + N_MONTE_CARLO))
CI_LEVEL = 0.95
CI_PERCENTILES: tuple[float, float] = (2.5, 97.5)

# ---------------------------------------------------------------------------
# Aggregation operators
# ---------------------------------------------------------------------------
OPERATORS: tuple[str, ...] = ("linear", "geometric", "min")
OPERATOR_LABELS: dict[str, str] = {
    "linear": r"$A_L$",
    "geometric": r"$A_G$",
    "min": r"$A_M$",
}
OPERATOR_FULL_LABELS: dict[str, str] = {
    "linear": r"$A_L$ (Linear)",
    "geometric": r"$A_G$ (Geometric)",
    "min": r"$A_M$ (Minimum)",
}

# ---------------------------------------------------------------------------
# Lambda grids
# ---------------------------------------------------------------------------
# Representative settings for grouped barplots in Section 6.1:
# 0.50 balanced, 0.75 prediction-dominant, 0.90 strongly prediction-dominant.
LAMBDA_BAR_VALUES: tuple[float, ...] = (0.50, 0.75, 0.90)

# Dense sweep stored for line plots and reuse in later experiments.
LAMBDA_DENSE_MIN = 0.0
LAMBDA_DENSE_MAX = 1.0
LAMBDA_DENSE_POINTS = 21
LAMBDA_DENSE_VALUES: tuple[float, ...] = tuple(
    float(x) for x in np.linspace(LAMBDA_DENSE_MIN, LAMBDA_DENSE_MAX, LAMBDA_DENSE_POINTS)
)

# ---------------------------------------------------------------------------
# Predictive overconfidence (Experiment 02 / Section 6.2)
# ---------------------------------------------------------------------------
# Deterministic upward bias R' = clip(R + δ, 0, 1), applied only to
# contextually weak cases with Q_i <= Q_WEAK_THRESHOLD (includes vetoes).
DELTA_VALUES: tuple[float, ...] = (0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
DELTA_TABLE_VALUES: tuple[float, ...] = (0.00, 0.10, 0.20, 0.30)
Q_WEAK_THRESHOLD = 0.25
OVERCONFIDENCE_LAMBDAS: tuple[float, ...] = LAMBDA_BAR_VALUES
OVERCONFIDENCE_PRIMARY_LAMBDA = 0.75

# ---------------------------------------------------------------------------
# Contextual uncertainty (Experiment 03 / Section 6.3)
# ---------------------------------------------------------------------------
LAMBDA_ROBUSTNESS_VALUES: tuple[float, ...] = tuple(round(i * 0.1, 1) for i in range(11))

SIGMA_Q_VALUES: tuple[float, ...] = (0.00, 0.05, 0.10, 0.15, 0.20)
CONTEXT_NOISE_PRIMARY_LAMBDA = 0.75

# Named population scenarios for Analysis 3 (population robustness).
# Each scenario overrides baseline fields where specified.
_BASE_POP: dict[str, object] = {
    "veto_fraction": VETO_FRACTION,
    "std_r_beta": STD_R_BETA,
    "std_q_beta": STD_Q_BETA,
    "veto_r_beta": VETO_R_BETA,
}

POPULATION_SCENARIOS: dict[str, dict[str, object]] = {
    "baseline": dict(_BASE_POP),
    "low_veto_frac": {**_BASE_POP, "veto_fraction": 0.05},
    "high_veto_frac": {**_BASE_POP, "veto_fraction": 0.10},
    "weak_heavy": {**_BASE_POP, "std_q_beta": (0.8, 2.0)},
    "weak_sparse": {**_BASE_POP, "std_q_beta": (2.0, 8.0)},
    "low_r_separation": {**_BASE_POP, "veto_r_beta": (4.0, 2.0)},
    "high_r_overlap": {**_BASE_POP, "veto_r_beta": (3.0, 3.0)},
}

# ---------------------------------------------------------------------------
# Sensitivity analysis (Experiment 04 / Section 6.4)
# ---------------------------------------------------------------------------
SENSITIVITY_PRIMARY_LAMBDA: float = 0.75
LAMBDA_PERTURBATION_VALUES: tuple[float, ...] = (0.70, 0.725, 0.75, 0.775, 0.80)
SIGMA_R_VALUES: tuple[float, ...] = (0.00, 0.02, 0.05, 0.10, 0.20)

# ---------------------------------------------------------------------------
# Figures (ported from experiments/plot_style.py and systemic_config.py)
# ---------------------------------------------------------------------------
FIG_DPI = 300
SAVE_PDF = True
SAVE_PNG = True
SAVE_SVG = False
FONT_SIZE = 11
FIG_SCALE = 0.85
PLOT_FONT_SCALE = 1.22
GROUPED_BAR_FIGSIZE = (8.5 * FIG_SCALE, 4.6 * FIG_SCALE)
SINGLE_PANEL_FIGSIZE = (8.5 * FIG_SCALE, 4.6 * FIG_SCALE)
TWO_PANEL_FIGSIZE = (9.0 * FIG_SCALE, 4.6 * FIG_SCALE)

# Okabe–Ito palette (colorblind-safe), matching experiments/plot_style.py
OKABE_ITO = (
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
    "#000000",
)
OPERATOR_COLORS: dict[str, str] = {
    "linear": OKABE_ITO[4],  # #0072B2
    "geometric": OKABE_ITO[5],  # #D55E00
    "min": OKABE_ITO[2],  # #009E73
}

OPERATOR_MARKERS: dict[str, str] = {
    "linear": "v",
    "geometric": "s",
    "min": "o",
}
OPERATOR_MARKERSIZES: dict[str, float] = {
    "linear": 13.0,
    "geometric": 9.0,
    "min": 6.5,
}
