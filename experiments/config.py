"""Reproducible configuration for CADEMAS simulation experiments."""

from pathlib import Path

# Dataset size and resource constraint
N = 1000
K = 100

# Sub-population sizes (must sum to N)
N_STD = 800
N_VETO = 100
N_SAFE = 100

# Reproducibility
SEED = 42

# Experiment parameters
LAMBDA_MIN = 0.01
LAMBDA_MAX = 0.99
LAMBDA_STEP = 0.05
LAMBDA_FIXED = 0.5

# RQ2: lambda panels where linear vulnerability becomes visible under R^alpha distortion
RQ2_LAMBDA_PANELS = (0.5, 0.85, 0.96)
# Operators highlighted in RQ2 (linear vs veto-safe regimes)
RQ2_OPERATORS = ("linear", "min", "geometric")

ALPHA_MIN = 0.1
ALPHA_MAX = 1.0
ALPHA_POINTS = 19

# RQ3 homogeneous sub-population
HOMOGENEOUS_N = 500
HOMOGENEOUS_Q_LEVELS = (0.8, 0.5, 0.2)

# Output paths
FIGURES_DIR = Path(__file__).resolve().parents[1] / "paper" / "figures"

OPERATORS = ("linear", "min", "max", "geometric")

OPERATOR_LABELS = {
    "linear": r"$A_L$ (Linear)",
    "min": r"$A_T^{\min}$ (Min)",
    "max": r"$A_S^{\max}$ (Max)",
    "geometric": r"$A_G$ (Geometric)",
}
