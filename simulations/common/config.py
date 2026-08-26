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
# Sensitivity analysis (Experiment 02) — Gaussian noise on R only
# ---------------------------------------------------------------------------
# Q is held fixed (same population design as Experiment 01). Predictive scores
# are perturbed as R' = clip(R + ε, 0, 1), ε ~ N(0, σ_R²).
SENSITIVITY_PRIMARY_LAMBDA: float = 0.75
SIGMA_R_VALUES: tuple[float, ...] = (0.00, 0.05, 0.10, 0.20)
SIGMA_R_FIGURE_VALUES: tuple[float, ...] = (0.05, 0.10, 0.20)
SIGMA_R_V_BY_K_VALUES: tuple[float, ...] = (0.00, 0.05, 0.10, 0.20)
# Policy-violation sweep over selection size for Exp 02 (V vs K figure).
TOP_K_SWEEP_VALUES: tuple[int, ...] = tuple(range(10, TOP_K + 1, 10))
# Seven ranking configurations used for pairwise Top-K agreement heatmaps
# (and Exp 01 panels f–g): A_L and A_G at three λ values, plus A_M.
AGREEMENT_LAMBDA_VALUES: tuple[float, ...] = (0.10, 0.50, 0.90)
# Barplot alpha: lightest at λ=0.10, most intense at λ=0.90.
AGREEMENT_LAMBDA_ALPHAS: dict[float, float] = {
    0.10: 0.40,
    0.50: 0.70,
    0.90: 1.00,
}
# Line-plot marker scale: largest at λ=0.10, decreasing toward A_M.
AGREEMENT_LAMBDA_MARKER_SCALES: dict[float, float] = {
    0.10: 1.22,
    0.50: 1.00,
    0.90: 0.82,
}
AGREEMENT_MIN_MARKER_SCALE: float = 0.68


def agreement_operator_specs() -> tuple[tuple[str, str, float | None, str], ...]:
    """Return ``(config_id, operator, lambda_or_None, tick_label)`` for 7 configs."""
    specs: list[tuple[str, str, float | None, str]] = []
    letter = {"linear": "L", "geometric": "G"}
    for op in ("linear", "geometric"):
        for lam in AGREEMENT_LAMBDA_VALUES:
            cid = f"{op}@{lam:.2f}"
            label = rf"$A_{{{letter[op]}}}({lam:.2g})$"
            specs.append((cid, op, float(lam), label))
    specs.append(("min", "min", None, r"$A_M$"))
    return tuple(specs)


# Kept for shared Top-K composition helpers (weak non-veto band).
Q_WEAK_THRESHOLD = 0.25

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
