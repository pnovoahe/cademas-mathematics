"""One-off export of cooperative R and fuzzy Q scores from CADEMAS-ML example bundle."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import h2o
import numpy as np
import pandas as pd

CADEMAS_APP = Path(__file__).resolve().parents[2].parent / "cademas-app"
if not CADEMAS_APP.exists():
    CADEMAS_APP = Path(os.environ.get("CADEMAS_APP", "/Users/pavelnovoahernandez/Cursor/cademas-app"))

sys.path.insert(0, str(CADEMAS_APP / "app"))
from fuzzy_context import calculate_context_score  # noqa: E402

EXAMPLE_DIR = CADEMAS_APP / "example_attrition"
OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "attrition"


def _read_cases_csv(path: Path) -> pd.DataFrame:
    for sep in (";", ",", "\t"):
        try:
            df = pd.read_csv(path, sep=sep)
            if len(df.columns) > 5:
                return df
        except (pd.errors.ParserError, ValueError):
            continue
    raise ValueError(f"Could not parse {path}")


def _prepare_case_ids(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Case_ID" in df.columns:
        df.insert(0, "CaseID", df["Case_ID"].values)
        return df.drop(columns=["Case_ID"])
    if "CaseID" not in df.columns:
        df.insert(0, "CaseID", np.arange(1, len(df) + 1))
    return df


def export_scores() -> pd.DataFrame:
    h2o.init(max_mem_size="700m", nthreads=1)

    cases_path = EXAMPLE_DIR / "data" / "cases_atttrition.csv"
    model_defs_path = EXAMPLE_DIR / "model_definitions.json"
    context_path = EXAMPLE_DIR / "context" / "context_digital_transformation.json"
    models_dir = EXAMPLE_DIR / "models"

    master_df = _prepare_case_ids(_read_cases_csv(cases_path))
    model_df = master_df.drop(columns=["CaseID"])

    with open(model_defs_path, encoding="utf-8") as f:
        feature_config = json.load(f)

    metric = "auc"
    metrics_vals = {
        m: feature_config[m]["performance"].get(metric, 0.0)
        for m in feature_config
    }
    total = sum(metrics_vals.values())
    weights = {m: (v / total if total > 0 else 1 / len(metrics_vals)) for m, v in metrics_vals.items()}

    risk_accum = np.zeros(len(master_df))
    for model_name, weight in weights.items():
        mojo_path = models_dir / model_name
        mojo = h2o.import_mojo(str(mojo_path))
        hf = h2o.H2OFrame(model_df)
        try:
            preds = mojo.predict(hf).as_data_frame()
            p_col = "p1" if "p1" in preds.columns else preds.columns[-1]
            risk_accum += preds[p_col].values * weight
        finally:
            h2o.remove(hf)

    with open(context_path, encoding="utf-8") as f:
        context_config = json.load(f)

    ci_scores, _ = calculate_context_score(master_df, context_config, "average")

    out = pd.DataFrame(
        {
            "case_id": master_df["CaseID"].values,
            "attrition": master_df["attrition"].values,
            "R": np.round(risk_accum, 4),
            "Q": np.round(ci_scores.values, 4),
        }
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "attrition_scores.csv"
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")
    print(out.sort_values("R", ascending=False).to_string(index=False))
    return out


if __name__ == "__main__":
    export_scores()
