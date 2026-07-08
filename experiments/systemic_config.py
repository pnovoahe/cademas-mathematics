"""Configuration for systemic CADEMAS evaluation (three-figure study)."""

from pathlib import Path

N = 1000
K = 100
N_STD = 800
N_VETO = 200
SEED = 42
N_MC_TRIALS = 30
MC_SEED_BASE = 0

LAMBDA_FIXED = 0.5
OPP_LAMBDA_MIN = 0.05
OPP_LAMBDA_MAX = 0.95
OPP_LAMBDA_POINTS = 19

# Exp 2: lambda=0.5 caps veto P at 0.5 while std reach 1.0; 0.85 exposes linear failure.
PREDICTIVE_OVERCONFIDENCE_LAMBDA = 0.85

# Exp 3: lower lambda weights Q more, amplifying linear sensitivity to Q noise.
NOISE_LAMBDA = 0.35

MU_SHIFT_MIN = 0.5
MU_SHIFT_MAX = 1.0
MU_SHIFT_POINTS = 11
VETO_R_STD = 0.05

NOISE_SIGMA_MIN = 0.0
NOISE_SIGMA_MAX = 0.5
NOISE_SIGMA_POINTS = 21

FIGURES_DIR = Path(__file__).resolve().parents[1] / "paper_final" / "figures"

# Figure dimensions (inches); aspect ratio preserved vs. original layout
FIG_SCALE = 0.85
TWO_PANEL_FIGSIZE = (9.0 * FIG_SCALE, 4.6 * FIG_SCALE)
SINGLE_PANEL_FIGSIZE = (8.5 * FIG_SCALE, 4.6 * FIG_SCALE)
PLOT_FONT_SCALE = 1.22

SYSTEMIC_LABELS = {
    "linear": r"$A_L$ (Linear)",
    "geometric": r"$A_G$ (Geometric)",
    "min": r"$A_T^{\min}$ (Min)",
}
