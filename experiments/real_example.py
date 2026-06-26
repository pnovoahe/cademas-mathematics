"""Real attrition example: operator comparison on CADEMAS-ML cohort."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from operators import aggregate

DATA_DIR = Path(__file__).resolve().parent / "data" / "attrition"
SCORES_CSV = DATA_DIR / "attrition_scores.csv"

LAMBDA = 0.5
TOP_K = 2
OPERATORS = ("linear", "min", "geometric")
OPERATOR_LABELS = {
    "linear": r"$A_L$",
    "min": r"$A_T^{\min}$",
    "geometric": r"$A_G$",
}


def load_attrition_scores(path: Path | None = None) -> pd.DataFrame:
    """Load precomputed cooperative risk R and fuzzy context Q."""
    path = path or SCORES_CSV
    df = pd.read_csv(path)
    required = {"case_id", "attrition", "R", "Q"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    return df.sort_values("case_id").reset_index(drop=True)


def apply_operators(df: pd.DataFrame, lam: float = LAMBDA) -> pd.DataFrame:
    """Add prioritization scores and ranks for each aggregation operator."""
    out = df.copy()
    R = out["R"].to_numpy(dtype=float)
    Q = out["Q"].to_numpy(dtype=float)

    for op in OPERATORS:
        P = aggregate(op, R, Q, lam)
        out[f"P_{op}"] = np.round(P, 4)
        out[f"rank_{op}"] = pd.Series(P).rank(ascending=False, method="min").astype(int)

    return out


def top_k_cases(df: pd.DataFrame, operator: str, k: int = TOP_K) -> pd.DataFrame:
    col = f"P_{operator}"
    return df.nlargest(k, col)[["case_id", "R", "Q", col, "attrition", f"rank_{operator}"]]


def summarize_example(lam: float = LAMBDA, k: int = TOP_K) -> pd.DataFrame:
    """Return full cohort with scores and ranks."""
    df = apply_operators(load_attrition_scores(), lam=lam)
    return df


def print_summary(df: pd.DataFrame, k: int = TOP_K) -> None:
    print(f"Attrition real example (n={len(df)}, lambda={LAMBDA}, Top-K={k})")
    print("-" * 60)
    for op in OPERATORS:
        top = top_k_cases(df, op, k=k)
        names = ", ".join(top["case_id"].tolist())
        print(f"  {OPERATOR_LABELS[op]:12s} Top-{k}: {names}")
    print()
    highlight = ["Evelyn Taylor", "Noah Lewis", "Lucas Wright", "Theodore Adams"]
    sub = df[df["case_id"].isin(highlight)].sort_values("R", ascending=False)
    cols = ["case_id", "R", "Q"] + [f"P_{op}" for op in OPERATORS] + [f"rank_{op}" for op in OPERATORS]
    print(sub[cols].to_string(index=False))
