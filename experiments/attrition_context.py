"""Fuzzy context scores for the CADEMAS-ML attrition example (Digital Transformation).

The analysis pipeline loads authoritative Q_i values from
``data/attrition/attrition_scores.csv`` (exported from CADEMAS-ML).
``compute_digital_context_scores`` implements the closed-form appendix
specification for audit; small numerical differences vs. the app engine are
expected because CADEMAS-ML aggregates all fuzzy audit columns when the UI
aggregation mode is ``average``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data" / "attrition"
CASES_CSV = DATA_DIR / "cases_atttrition.csv"

ROLE_ADAPTABILITY = {
    "Research Scientist": 0.9,
    "Research Director": 0.9,
    "Manager": 0.8,
    "Manufacturing Director": 0.7,
    "Healthcare Representative": 0.7,
    "Sales Executive": 0.6,
    "Human Resources": 0.5,
    "Laboratory Technician": 0.4,
    "Sales Representative": 0.4,
}


def linear_increasing(x: np.ndarray, a: float, b: float) -> np.ndarray:
    if a == b:
        return (x >= a).astype(float)
    y = (x - a) / (b - a)
    return np.clip(y, 0.0, 1.0)


def triangular(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    if a == b or b == c:
        return np.zeros_like(x, dtype=float)
    term1 = (x - a) / (b - a)
    term2 = (c - x) / (c - b)
    return np.maximum(0.0, np.minimum(term1, term2))


def trapezoidal(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    term1 = (x - a) / (b - a) if b > a else np.ones_like(x)
    term3 = (d - x) / (d - c) if d > c else np.ones_like(x)
    return np.maximum(0.0, np.minimum(np.minimum(term1, 1.0), term3))


def _col(df: pd.DataFrame, name: str) -> np.ndarray:
    return df[name].astype(float).values


def compute_digital_context_scores(df: pd.DataFrame) -> pd.Series:
    """Return Q_i for the Digital / Technological Transformation context."""
    mu_train = linear_increasing(_col(df, "training_times_last_year"), 1.0, 6.0)
    mu_edu = linear_increasing(_col(df, "education"), 2.0, 5.0)
    mu_yrs_co = triangular(_col(df, "years_at_company"), 3.0, 10.0, 20.0)
    mu_yrs_role = triangular(_col(df, "years_in_current_role"), 1.0, 6.0, 11.0)
    mu_age = trapezoidal(_col(df, "age"), 22.0, 28.0, 50.0, 57.0)
    mu_exp = linear_increasing(_col(df, "total_working_years"), 2.0, 10.0)
    mu_role = df["job_role"].map(ROLE_ADAPTABILITY).fillna(0.0).astype(float).values

    mu_upskill = 0.5 * (mu_train + mu_edu)
    mu_career = mu_yrs_co * mu_yrs_role
    mu_invest = mu_age * mu_exp

    scores = 0.25 * (mu_upskill + mu_role + mu_career + mu_invest)
    return pd.Series(np.clip(scores, 0.0, 1.0), index=df.index, name="Q")


def load_cases_csv(path: Path | None = None) -> pd.DataFrame:
    path = path or CASES_CSV
    for sep in (";", ",", "\t"):
        try:
            df = pd.read_csv(path, sep=sep)
            if len(df.columns) > 5:
                return df
        except (pd.errors.ParserError, ValueError):
            continue
    raise ValueError(f"Could not parse {path}")
