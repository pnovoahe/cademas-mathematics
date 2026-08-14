"""Real attrition example: operator comparison on the CADEMAS-ML 100-case cohort."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from operators import aggregate

DATA_DIR = Path(__file__).resolve().parent / "data" / "attrition100"
RAW_CSV = DATA_DIR / "cases_attrition_100.csv"
FINAL_CSV = DATA_DIR / "cases_attrition_100_final.csv"

LAMBDA = 0.5
LAMBDA_LOW = 0.1
LAMBDA_HIGH = 0.9
TOP_K = 10

# Each entry: (column key, aggregation operator, lambda).
COMPARISONS: tuple[tuple[str, str, float], ...] = (
    ("linear_01", "linear", LAMBDA_LOW),
    ("linear_05", "linear", LAMBDA),
    ("linear_90", "linear", LAMBDA_HIGH),
    ("min_05", "min", LAMBDA),
    ("geometric_05", "geometric", LAMBDA),
)

OPERATORS = tuple(key for key, _, _ in COMPARISONS)
OPERATOR_LABELS = {
    "linear_01": r"$A_L$ ($\lambda=0.1$)",
    "linear_05": r"$A_L$ ($\lambda=0.5$)",
    "linear_90": r"$A_L$ ($\lambda=0.9$)",
    "min_05": r"$A_C^{\min}$",
    "geometric_05": r"$A_G$ ($\lambda=0.5$)",
}
# Bump chart tracks: aggregation operators only.
BUMP_TRACKS = OPERATORS
BUMP_LABELS = OPERATOR_LABELS
BASELINE_KEY = "linear_05"


def load_attrition_scores(path: Path | None = None) -> pd.DataFrame:
    """Load predictive score R and fuzzy context Q from the final cohort CSV."""
    path = path or FINAL_CSV
    df = pd.read_csv(path, sep=";")
    df = df.rename(
        columns={
            "Case_ID": "case_id",
            "Ri_Global_Risk": "R",
            "Ci_Context_Score": "Q",
        }
    )
    required = {"case_id", "attrition", "R", "Q"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    return df.sort_values("case_id").reset_index(drop=True)


def describe_cohort(raw_path: Path | None = None) -> dict:
    """Summarize cohort composition from the raw feature CSV."""
    raw_path = raw_path or RAW_CSV
    raw = pd.read_csv(raw_path, sep=";")
    attrition = raw["attrition"].value_counts().to_dict()
    departments = raw["department"].value_counts().to_dict()
    scores = load_attrition_scores()
    q_zero = int((scores["Q"] == 0).sum())
    return {
        "n": len(raw),
        "attrition": attrition,
        "departments": departments,
        "r_min": float(scores["R"].min()),
        "r_max": float(scores["R"].max()),
        "q_min": float(scores["Q"].min()),
        "q_max": float(scores["Q"].max()),
        "q_zero": q_zero,
    }


def apply_operators(df: pd.DataFrame) -> pd.DataFrame:
    """Add prioritization scores and ranks for each comparison setting."""
    out = df.copy().reset_index(drop=True)
    R = out["R"].to_numpy(dtype=float)
    Q = out["Q"].to_numpy(dtype=float)

    out["rank_R"] = out["R"].rank(ascending=False, method="min").astype(int)
    out["rank_Q"] = out["Q"].rank(ascending=False, method="min").astype(int)

    for key, op, lam in COMPARISONS:
        P = aggregate(op, R, Q, lam)
        out[f"P_{key}"] = np.round(P, 4)
        out[f"rank_{key}"] = (
            pd.Series(P, index=out.index).rank(ascending=False, method="min").astype(int)
        )

    out["P_linear"] = out[f"P_{BASELINE_KEY}"]
    out["rank_linear"] = out[f"rank_{BASELINE_KEY}"]
    out["P_min"] = out["P_min_05"]
    out["rank_min"] = out["rank_min_05"]
    out["P_geometric"] = out["P_geometric_05"]
    out["rank_geometric"] = out["rank_geometric_05"]

    return out


def top_k_union(
    df: pd.DataFrame,
    k: int = TOP_K,
    comparisons: tuple[str, ...] = OPERATORS,
) -> list[str]:
    """Return case IDs in the union of Top-K tiers across comparison settings."""
    ids: set[str] = set()
    for key in comparisons:
        top = df.nsmallest(k, f"rank_{key}")
        ids.update(top["case_id"].tolist())
    return sorted(ids, key=lambda cid: int(df.loc[df["case_id"] == cid, f"rank_{BASELINE_KEY}"].iloc[0]))


def top_k_cases(df: pd.DataFrame, key: str, k: int = TOP_K) -> pd.DataFrame:
    col = f"P_{key}"
    return df.nlargest(k, col)[["case_id", "R", "Q", col, "attrition", f"rank_{key}"]]


def illustrative_cases(df: pd.DataFrame) -> list[str]:
    """Select cases for scatter annotations: high R/low Q, low R/high Q, Q=0 with high R."""
    picks: list[str] = []
    high_r = df.nlargest(1, "R").iloc[0]
    if high_r["Q"] < df["Q"].median():
        picks.append(high_r["case_id"])

    high_q = df[df["Q"] > 0].nlargest(1, "Q").iloc[0]
    if high_q["R"] < df["R"].median():
        picks.append(high_q["case_id"])

    veto_high_r = df[(df["Q"] == 0) & (df["R"] > 0.5)].nlargest(1, "R")
    if len(veto_high_r):
        picks.append(veto_high_r.iloc[0]["case_id"])

    seen: set[str] = set()
    unique: list[str] = []
    for name in picks:
        if name not in seen:
            seen.add(name)
            unique.append(name)
    return unique[:3]


def rank_displacement(
    df: pd.DataFrame,
    case_id: str,
    tracks: tuple[str, ...] = BUMP_TRACKS,
) -> int:
    """Max rank spread across bump-chart tracks for one case."""
    ranks = [int(df.loc[df["case_id"] == case_id, f"rank_{key}"].iloc[0]) for key in tracks]
    return max(ranks) - min(ranks)


def summarize_example() -> pd.DataFrame:
    """Return full cohort with scores and ranks."""
    return apply_operators(load_attrition_scores())


def print_summary(df: pd.DataFrame, k: int = TOP_K) -> None:
    stats = describe_cohort()
    print(f"Attrition real example (n={len(df)}, Top-K={k})")
    print(f"  Linear lambda: {LAMBDA_LOW} (context), {LAMBDA} (baseline), {LAMBDA_HIGH} (predictive)")
    print(f"  Attrition: {stats['attrition']}")
    print(f"  Departments: {stats['departments']}")
    print(f"  R range: [{stats['r_min']:.4f}, {stats['r_max']:.4f}]")
    print(f"  Q range: [{stats['q_min']:.4f}, {stats['q_max']:.4f}]  (Q=0: {stats['q_zero']})")
    print("-" * 60)
    for key in OPERATORS:
        top = top_k_cases(df, key, k=k)
        names = ", ".join(top["case_id"].head(3).tolist()) + ", ..."
        print(f"  {OPERATOR_LABELS[key]:22s} Top-{k} (first 3): {names}")
    print(f"\n  Union Top-{k} size: {len(top_k_union(df, k))}")
    print()
    highlight = illustrative_cases(df)
    sub = df[df["case_id"].isin(highlight)]
    cols = ["case_id", "R", "Q"] + [f"P_{key}" for key in OPERATORS] + [f"rank_{key}" for key in OPERATORS]
    print(sub[cols].to_string(index=False))
