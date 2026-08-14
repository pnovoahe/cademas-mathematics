"""Configuration for systemic CADEMAS evaluation."""

from pathlib import Path

N = 1000
K = 100
N_STD = 800
N_VETO = 200
N_INTER = 200
SEED = 42
N_MC_TRIALS = 1000
MC_SEED_BASE = 42

LAMBDA_FIXED = 0.5
OPP_LAMBDA_MIN = 0.05
OPP_LAMBDA_MAX = 0.95
OPP_LAMBDA_POINTS = 19

# Exp 2: lambda=0.5 caps veto P at 0.5 while std reach 1.0; 0.85 exposes linear failure.
PREDICTIVE_OVERCONFIDENCE_LAMBDA = 0.85

# Exp 3: rank stability under contextual imprecision (multi-lambda, multi-model).
NOISE_LAMBDAS = (0.25, 0.50, 0.75)
NOISE_POPULATION_MODELS = ("uniform", "beta")
NOISE_DISTRIBUTIONS = ("gaussian", "uniform")

MU_SHIFT_MIN = 0.5
MU_SHIFT_MAX = 1.0
MU_SHIFT_POINTS = 11
VETO_R_STD = 0.05

NOISE_SIGMA_MIN = 0.0
NOISE_SIGMA_MAX = 0.5
NOISE_SIGMA_POINTS = 21

# Exp 4: intermediate context (Q=0.5) with high predictive scores.
INTER_Q = 0.5
INTER_R_MU = 0.9
INTER_R_STD = 0.05
INTER_LAMBDA_MIN = 0.1
INTER_LAMBDA_MAX = 0.9
INTER_LAMBDA_POINTS = 17

FIGURES_DIR = Path(__file__).resolve().parents[1] / "first_round" / "figures"
MC_CACHE_DIR = Path(__file__).resolve().parent / ".cache" / "mc"

# Figure dimensions (inches); aspect ratio preserved vs. original layout
FIG_SCALE = 0.85
TWO_PANEL_FIGSIZE = (9.0 * FIG_SCALE, 4.6 * FIG_SCALE)
SINGLE_PANEL_FIGSIZE = (8.5 * FIG_SCALE, 4.6 * FIG_SCALE)
NOISE_FIGSIZE = (10.5 * FIG_SCALE, 6.8 * FIG_SCALE)
PLOT_FONT_SCALE = 1.22

SYSTEMIC_LABELS = {
    "linear": r"$A_L$ (Linear)",
    "geometric": r"$A_G$ (Geometric)",
    "min": r"$A_C^{\min}$ (Min)",
}
